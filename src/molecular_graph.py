"""
Phase 2: SMILES -> hierarchical molecular graph (atom -> substructure -> molecule),
following HDN-DDI's design (Sun & Zheng, 2025, BMC Bioinformatics).

Design choices made where the paper's main text doesn't fully specify details
(exact rules are in their Additional File 1, which isn't publicly available to us):
  - Atom features: the 55-dim scheme documented explicitly in SSI-DDI (Nyamabo et al. 2021),
    which DSN-DDI and HDN-DDI's lineage all build on.
  - Large-ring fallback decomposition: BRICS sometimes leaves a big fused-ring system as one
    fragment (e.g. steroid scaffolds, fused aromatics). We split any fragment larger than
    RING_SPLIT_THRESHOLD atoms into its individual SSSR (smallest set of smallest rings) via
    RDKit's ring perception, approximating the "additional decomposition rules" HDN-DDI cites
    from Zang et al.'s HiMol paper without full access to their exact ruleset.
  - Substructure-level and molecule-level node features are initialized as the mean of their
    member atoms' feature vectors (a standard bootstrap for hierarchical graphs; the GNN layers
    refine this during message passing).
"""
from rdkit import Chem
from rdkit.Chem import BRICS
import torch
from torch_geometric.data import Data


class HierGraphData(Data):
    """A Data subclass that knows how to correctly shift mol_node_idx when many of these
    graphs get merged into one big batched graph (needed for Phase 3's mini-batch training —
    plain Data doesn't know this field is a node index that needs offsetting)."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "mol_node_idx":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

RING_SPLIT_THRESHOLD = 12  # fragments larger than this get split into individual rings

ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
    'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge',
    'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown',
]  # 44 entries, matches SSI-DDI's documented atomic-symbol one-hot
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]  # 5 entries


def _one_hot(value, choices):
    vec = [0] * len(choices)
    if value in choices:
        vec[choices.index(value)] = 1
    else:
        vec[-1] = 1  # last slot = "unknown/other" bucket
    return vec


def atom_features(atom):
    """55-dim atom feature vector: 44 (symbol) + 1 (degree) + 1 (implicit valence)
    + 1 (formal charge) + 1 (radical electrons) + 5 (hybridization) + 1 (aromatic) + 1 (total Hs)"""
    return torch.tensor(
        _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
        + [atom.GetDegree()]
        + [atom.GetValence(Chem.ValenceType.IMPLICIT)]
        + [atom.GetFormalCharge()]
        + [atom.GetNumRadicalElectrons()]
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic())]
        + [atom.GetTotalNumHs()],
        dtype=torch.float,
    )


def decompose_fragments(mol):
    """Return a list of atom-index tuples, one per chemical substructure."""
    n_atoms = mol.GetNumAtoms()
    bonds = list(BRICS.FindBRICSBonds(mol))
    if not bonds:
        frag_groups = [tuple(range(n_atoms))]
    else:
        frag_mol = Chem.FragmentOnBRICSBonds(mol)
        raw_groups = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
        # drop dummy atoms RDKit appends at cut points (indices >= original atom count)
        frag_groups = [tuple(i for i in g if i < n_atoms) for g in raw_groups]

    # split any oversized fragment into its individual rings (fallback for fused-ring systems)
    final_groups = []
    ring_info = mol.GetRingInfo()
    for group in frag_groups:
        if len(group) <= RING_SPLIT_THRESHOLD:
            final_groups.append(group)
            continue
        group_set = set(group)
        rings = [tuple(r) for r in ring_info.AtomRings() if group_set.issuperset(r)]
        covered = set()
        for r in rings:
            final_groups.append(r)
            covered.update(r)
        leftover = tuple(i for i in group if i not in covered)
        if leftover:
            final_groups.append(leftover)
    return final_groups


def smiles_to_hierarchical_graph(smiles):
    """Build a 3-level hierarchical graph (atom / substructure / molecule) as a PyG Data object.

    Returns None if the SMILES fails to parse.
    Node layout: [atoms 0..n-1] [substructures n..n+m-1] [molecule node n+m]
    Data fields:
      x            - node features (atoms: 55-dim RDKit features; substructure/molecule: mean of members)
      edge_index   - bidirectional edges, all levels combined
      node_type    - 0=atom, 1=substructure, 2=molecule
      edge_type    - 0=atom-atom bond, 1=atom-substructure membership, 2=substructure-molecule
      frag_groups  - raw atom-index groups per substructure (kept for interpretability/case studies)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]

    src, dst, edge_type = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
        edge_type += [0, 0]

    frag_groups = decompose_fragments(mol)
    n_frags = len(frag_groups)
    frag_feats = []
    for f_idx, group in enumerate(frag_groups):
        frag_node_id = n_atoms + f_idx
        member_feats = torch.stack([atom_feats[a] for a in group])
        frag_feats.append(member_feats.mean(dim=0))
        for a in group:
            src += [frag_node_id, a]
            dst += [a, frag_node_id]
            edge_type += [1, 1]

    mol_node_id = n_atoms + n_frags
    mol_feat = torch.stack(frag_feats).mean(dim=0) if frag_feats else torch.zeros(55)
    for f_idx in range(n_frags):
        frag_node_id = n_atoms + f_idx
        src += [mol_node_id, frag_node_id]
        dst += [frag_node_id, mol_node_id]
        edge_type += [2, 2]

    x = torch.stack(atom_feats + frag_feats + [mol_feat])
    node_type = torch.tensor([0] * n_atoms + [1] * n_frags + [2], dtype=torch.long)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    return HierGraphData(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
        frag_groups=frag_groups,
        mol_node_idx=mol_node_id,
        num_atoms=n_atoms,
    )
