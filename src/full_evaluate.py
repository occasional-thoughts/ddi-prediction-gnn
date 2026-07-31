"""
Phase 4: proper full evaluation on the untouched DrugBank warm-start test set
(38,362 rows -> 76,724 pairs with negatives), reporting the same four metrics
HDN-DDI's paper reports: ACC, AUROC, AUPR, F1.
"""
import sys
import time

import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from dataset import DrugPairDataset, make_collate_fn
from model import HDN_DDI

DEVICE = torch.device("cpu")


def full_evaluate(test_csv, graphs_path, checkpoint_path, batch_size=512, n_rel_types=86):
    graphs = torch.load(graphs_path, weights_only=False)
    dataset = DrugPairDataset(test_csv, graphs)
    loader = DataLoader(dataset, batch_size=batch_size // 2, shuffle=False, collate_fn=make_collate_fn(graphs))
    print(f"test rows: {len(dataset)} -> {len(dataset)*2} pairs with negatives, {len(loader)} batches")

    model = HDN_DDI(in_dim=55, hidden_dim=64, n_blocks=6, heads=2, n_rel_types=n_rel_types).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()

    all_scores, all_labels = [], []
    start = time.time()
    with torch.no_grad():
        for i, (batch_x, batch_y, rels, labels) in enumerate(loader):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            rels = rels.to(DEVICE)
            scores = model(batch_x, batch_y, rels)
            all_scores.append(scores.cpu())
            all_labels.append(labels)
            if (i + 1) % 20 == 0:
                print(f"  batch {i+1}/{len(loader)} ({time.time()-start:.0f}s elapsed)")
    elapsed = time.time() - start

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (scores > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    f1 = f1_score(labels, preds)

    print(f"\nFull test-set evaluation ({elapsed:.1f}s):")
    print(f"  ACC:   {acc*100:.2f}%")
    print(f"  AUROC: {auroc*100:.2f}%")
    print(f"  AUPR:  {aupr*100:.2f}%")
    print(f"  F1:    {f1*100:.2f}%")
    print(f"\nHDN-DDI paper's published warm-start numbers (for comparison):")
    print(f"  ACC: 97.93%  AUROC: 99.73%  AUPR: 99.72%  F1: 98.03%")
    return acc, auroc, aupr, f1


if __name__ == "__main__":
    full_evaluate(
        test_csv="data/raw/DrugBank/warm_start/fold0/test.csv",
        graphs_path="data/processed/drugbank_graphs.pt",
        checkpoint_path="results/hdn_ddi_warmstart_fold0.pt",
    )
