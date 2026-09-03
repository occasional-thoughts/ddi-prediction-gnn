"""
Phase 3c: turns DrugBank's train/test CSV rows into ready-to-train batches.

Each CSV row is one documented positive interaction (d1, d2, type) plus one
pre-generated negative sample ("Neg samples" column, format `<drug_id>$h` or
`<drug_id>$t` -- $h means replace d1 (head) with that drug, $t means replace
d2 (tail)), matching the paper's 1:1 positive:negative sampling and the
protocol shared by SSI-DDI/DSN-DDI/HDN-DDI.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class DrugPairDataset(Dataset):
    def __init__(self, csv_path, graphs):
        self.df = pd.read_csv(csv_path)
        self.graphs = graphs
        # HDN-DDI's released warm-start files carry the negative sample directly in a
        # "Neg samples" column; the cold-start files (same dataset, different split
        # script) instead leave "Neg samples" empty and put that same "<drug_id>$h/$t"
        # string in a column called "split" -- just an inconsistency in their released
        # data, not a different sampling scheme. Detect which one this file actually uses.
        if "Neg samples" in self.df.columns and self.df["Neg samples"].notna().any():
            self.neg_col = "Neg samples"
        else:
            self.neg_col = "split"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        d1, d2, rel = row["d1"], row["d2"], int(row["type"])
        neg_id, side = row[self.neg_col].split("$")

        if side == "h":
            neg_d1, neg_d2 = neg_id, d2
        else:  # side == "t"
            neg_d1, neg_d2 = d1, neg_id

        return {
            "pos_x": d1, "pos_y": d2,
            "neg_x": neg_d1, "neg_y": neg_d2,
            "rel": rel,
        }


def make_collate_fn(graphs):
    """Builds one batch containing both the positive and negative sample for every
    row (so a `rows_per_batch=256` DataLoader yields 512 total drug pairs per step)."""

    def collate(batch):
        x_graphs, y_graphs, rels, labels = [], [], [], []
        for item in batch:
            x_graphs.append(graphs[item["pos_x"]])
            y_graphs.append(graphs[item["pos_y"]])
            rels.append(item["rel"])
            labels.append(1.0)
        for item in batch:
            x_graphs.append(graphs[item["neg_x"]])
            y_graphs.append(graphs[item["neg_y"]])
            rels.append(item["rel"])
            labels.append(0.0)

        return (
            Batch.from_data_list(x_graphs),
            Batch.from_data_list(y_graphs),
            torch.tensor(rels, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float),
        )

    return collate
