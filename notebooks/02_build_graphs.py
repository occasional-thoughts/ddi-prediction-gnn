"""
Phase 2: convert every drug's SMILES into a hierarchical molecular graph and cache
the results to disk, so Phase 3 (model training) doesn't need to rebuild them every run.
"""
import sys
import time

import pandas as pd
import torch
from rdkit import RDLogger

sys.path.insert(0, "src")
from molecular_graph import smiles_to_hierarchical_graph

RDLogger.DisableLog("rdApp.*")

RAW = "data/raw"
PROCESSED = "data/processed"


def build_graph_cache(smiles_csv, out_path, label=""):
    df = pd.read_csv(smiles_csv)
    graphs = {}
    failures = []
    n_atoms_list, n_frags_list = [], []

    start = time.time()
    for _, row in df.iterrows():
        g = smiles_to_hierarchical_graph(row["smiles"])
        if g is None:
            failures.append(row["drug_id"])
            continue
        graphs[row["drug_id"]] = g
        n_atoms_list.append(g.num_atoms)
        n_frags_list.append((g.node_type == 1).sum().item())
    elapsed = time.time() - start

    torch.save(graphs, out_path)

    print(f"[{label}] {len(graphs)}/{len(df)} drugs converted successfully in {elapsed:.1f}s")
    if failures:
        print(f"[{label}] {len(failures)} failures: {failures}")
    if n_atoms_list:
        s_atoms = pd.Series(n_atoms_list)
        s_frags = pd.Series(n_frags_list)
        print(f"[{label}] atoms/drug -> min {s_atoms.min()}, max {s_atoms.max()}, mean {s_atoms.mean():.1f}")
        print(f"[{label}] fragments/drug -> min {s_frags.min()}, max {s_frags.max()}, mean {s_frags.mean():.1f}")
    print(f"[{label}] cached to {out_path}\n")
    return graphs


if __name__ == "__main__":
    build_graph_cache(f"{RAW}/DrugBank/drug_smiles.csv", f"{PROCESSED}/drugbank_graphs.pt", label="DrugBank")
    build_graph_cache(f"{RAW}/Twosides/drug_smiles.csv", f"{PROCESSED}/twosides_graphs.pt", label="Twosides")
