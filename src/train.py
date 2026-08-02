"""
Phase 3d: training loop for HDN-DDI, matching the paper's reported hyperparameters
(Adam, lr=0.001, batch size 512, 6 blocks, 2 attention heads, 86 DrugBank interaction types).

Reverted back to this simple configuration after two later attempts -- (1) matching the
paper's equations more literally in the interactive-view layer, and (2) matching the
authors' own exact training hyperparameters (weight decay, LR decay, 1024-batch,
200-epoch early stopping) extracted from their real run logs -- both measured WORSE than
this version. This is the exact configuration (paired with model.py's current
architecture) that produced our best verified result: 89.40% test accuracy / 95.23%
AUROC / 94.00% AUPR / 89.71% F1 on DrugBank warm-start. See project notes for the full
comparison across attempts.
"""
import sys
import time

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "src")
from dataset import DrugPairDataset, make_collate_fn
from model import HDN_DDI

DEVICE = torch.device("cpu")  # see project notes: MPS measured slower for this workload


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for i, (batch_x, batch_y, rels, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            rels, labels = rels.to(device), labels.to(device)
            scores = model(batch_x, batch_y, rels)
            loss_sum += torch.nn.functional.binary_cross_entropy(scores, labels).item()
            correct += ((scores > 0.5).float() == labels).sum().item()
            total += len(labels)
    return loss_sum / (i + 1), correct / total


def train(
    train_csv, graphs_path, rows_per_batch=256, lr=1e-3, n_epochs=30,
    n_rel_types=86, val_fraction=0.2, max_batches_per_epoch=None, checkpoint_path=None,
    use_edge_features=False,
):
    graphs = torch.load(graphs_path, weights_only=False)
    full_dataset = DrugPairDataset(train_csv, graphs)

    idx_train, idx_val = train_test_split(range(len(full_dataset)), test_size=val_fraction, random_state=42)
    collate = make_collate_fn(graphs)
    train_loader = DataLoader(Subset(full_dataset, idx_train), batch_size=rows_per_batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Subset(full_dataset, idx_val), batch_size=rows_per_batch, shuffle=False, collate_fn=collate)

    print(f"train rows: {len(idx_train)}, val rows: {len(idx_val)} "
          f"({rows_per_batch*2} pairs/batch, {len(train_loader)} batches/epoch) "
          f"| use_edge_features={use_edge_features}")

    model = HDN_DDI(in_dim=55, hidden_dim=64, n_blocks=6, heads=2, n_rel_types=n_rel_types,
                     use_edge_features=use_edge_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        start = time.time()
        loss_sum = 0.0
        for i, (batch_x, batch_y, rels, labels) in enumerate(train_loader):
            if max_batches_per_epoch and i >= max_batches_per_epoch:
                break
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            rels, labels = rels.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            scores = model(batch_x, batch_y, rels)
            loss = torch.nn.functional.binary_cross_entropy(scores, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        train_loss = loss_sum / (i + 1)
        val_loss, val_acc = evaluate(model, val_loader, DEVICE, max_batches=20)
        elapsed = time.time() - start
        print(f"epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_acc {val_acc:.4f} | {elapsed:.1f}s")

        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path)

    return model


if __name__ == "__main__":
    train(
        train_csv="data/raw/DrugBank/warm_start/fold0/train.csv",
        graphs_path="data/processed/drugbank_graphs.pt",
        rows_per_batch=256,
        n_epochs=30,
        checkpoint_path="results/hdn_ddi_warmstart_fold0_v2.pt",
    )
