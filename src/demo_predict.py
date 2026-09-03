"""
Interactive demo: give two drugs (DrugBank ID or raw SMILES), get a DDI prediction back.

This is the actual "use the system" demonstration -- unlike full_evaluate.py (which
scores a whole held-out test set), this takes two arbitrary drugs and returns the
model's predicted interaction, computed live.

Usage:
    python src/demo_predict.py DB00460 DB04571
    python src/demo_predict.py "CC(=O)OC1=CC=CC=C1C(=O)O" "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    python src/demo_predict.py            # runs with two example DrugBank IDs

Interaction-type descriptions come from the original DeepDDI dataset release (Ryu et al.
2018, PNAS) -- see data/raw/DrugBank/interaction_types.csv and PROGRESS.md for how the
type-ID mapping was cross-checked against this project's own type-frequency distribution.
"""
import sys
import time

import pandas as pd
import torch
from torch_geometric.data import Batch

sys.path.insert(0, "src")
from molecular_graph import smiles_to_hierarchical_graph
from model import HDN_DDI

DEVICE = torch.device("cpu")
CHECKPOINT = "results/hdn_ddi_warmstart_fold0_edgeaware.pt"  # bond-aware model, 89.80% ACC
N_REL_TYPES = 86

DEFAULT_PAIR = ("DB00460", "DB04571")  # two real DrugBank drugs, for a no-argument demo run


def resolve_drug(token, smiles_lookup, name_lookup):
    """token is either a DrugBank ID (looked up) or a raw SMILES string.
    Returns (smiles, id_or_input, display_label, name_for_sentences)."""
    token = token.strip()
    if token in smiles_lookup:
        name = name_lookup.get(token)
        label = f"{name} ({token})" if name else f"{token} (name not in reference dataset)"
        return smiles_lookup[token], token, label, name or token
    preview = token if len(token) <= 24 else token[:21] + "..."
    return token, token, f"Custom SMILES: {preview}", None


def main():
    args = sys.argv[1:]
    if len(args) == 2:
        drug_a_in, drug_b_in = args
    else:
        drug_a_in, drug_b_in = DEFAULT_PAIR
        print(f"(no arguments given -- using example pair {DEFAULT_PAIR})\n")

    smiles_df = pd.read_csv("data/raw/DrugBank/drug_smiles.csv")
    smiles_lookup = dict(zip(smiles_df["drug_id"], smiles_df["smiles"]))
    names_df = pd.read_csv("data/raw/DrugBank/drug_names.csv")
    name_lookup = dict(zip(names_df["drugbank_id"], names_df["name"]))
    types_df = pd.read_csv("data/raw/DrugBank/interaction_types.csv")
    type_desc = dict(zip(types_df["type_id"], types_df["description"]))

    smiles_a, id_a, label_a, sentence_a = resolve_drug(drug_a_in, smiles_lookup, name_lookup)
    smiles_b, id_b, label_b, sentence_b = resolve_drug(drug_b_in, smiles_lookup, name_lookup)
    sentence_a = sentence_a or "Drug A"
    sentence_b = sentence_b or "Drug B"

    def describe(type_id):
        text = type_desc.get(type_id, f"(no description found for type {type_id})")
        return text.replace("Drug A", sentence_a).replace("Drug B", sentence_b)
    print(f"Drug A: {label_a}  (SMILES: {smiles_a})")
    print(f"Drug B: {label_b}  (SMILES: {smiles_b})")

    graph_a = smiles_to_hierarchical_graph(smiles_a)
    graph_b = smiles_to_hierarchical_graph(smiles_b)
    if graph_a is None or graph_b is None:
        print("\nCould not parse one of the SMILES strings -- check the input.")
        sys.exit(1)

    batch_a = Batch.from_data_list([graph_a])
    batch_b = Batch.from_data_list([graph_b])

    print("\nLoading trained bond-aware model...", flush=True)
    model = HDN_DDI(in_dim=55, hidden_dim=64, n_blocks=6, heads=2, n_rel_types=N_REL_TYPES,
                     use_edge_features=True).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f"Scoring all {N_REL_TYPES} DrugBank interaction-type codes for this pair...", flush=True)
    start = time.time()
    scores = torch.zeros(N_REL_TYPES)
    with torch.no_grad():
        for rel in range(N_REL_TYPES):
            rel_ids = torch.tensor([rel])
            # batch objects get mutated in place by the encoder, so rebuild per relation
            bx = Batch.from_data_list([graph_a])
            by = Batch.from_data_list([graph_b])
            scores[rel] = model(bx, by, rel_ids).item()
    elapsed = time.time() - start

    top5 = torch.topk(scores, 5)
    print(f"\nDone in {elapsed:.1f}s. Top predicted interaction types:")
    for score, rel in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {score*100:5.2f}%  (type {rel:2d})  {describe(rel)}")

    best_score = top5.values[0].item()
    best_type = top5.indices[0].item()
    verdict = "an interaction IS predicted" if best_score > 0.5 else "no significant interaction is predicted"
    print(f"\nSummary: {verdict} between these two drugs — {describe(best_type)} "
          f"({best_score*100:.2f}% confidence, type {best_type}).")


if __name__ == "__main__":
    main()
