"""
Phase 3d: training loop for HDN-DDI.

Updated to match the AUTHORS' ACTUAL verified training configuration, extracted
directly from their released training script (drugbank_test/transductive_train.py)
and real run logs (log/warm-start/*.log) in the HDN-DDI GitHub repo -- not just the
paper's prose, which turned out to omit several details that matter a lot:
  - Adam with weight_decay=5e-4 (paper's text doesn't mention weight decay at all)
  - An exponential LR decay schedule: lr *= 0.96 per epoch
  - Up to 200 epochs, with early stopping: stop once 40 epochs pass with no
    improvement in the mean of (val_acc, val_auroc, val_f1). Their own fold-0 run
    went 127 epochs (best at epoch 87) to reach 97.32% test accuracy -- we were
    stopping at a fixed 30 epochs, nowhere near enough.
  - batch_size=1024 POSITIVE triples per step (2048 total pairs with negatives),
    not the 512 total pairs we were using.
"""
import sys
import time

import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "src")
from dataset import DrugPairDataset, make_collate_fn
from model import HDN_DDI

DEVICE = torch.device("cpu")  # see project notes: MPS measured slower for this workload


def evaluate(model, loader, device):
    """Full pass over the given loader (no sampling/capping -- matches the authors'
    own approach of validating on the entire val set every epoch, which matters for
    a reliable early-stopping signal)."""
    model.eval()
    loss_sum, all_scores, all_labels = 0.0, [], []
    with torch.no_grad():
        for i, (batch_x, batch_y, rels, labels) in enumerate(loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            rels, labels = rels.to(device), labels.to(device)
            scores = model(batch_x, batch_y, rels)
            loss_sum += torch.nn.functional.binary_cross_entropy(scores, labels).item()
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (scores > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    auroc = roc_auc_score(labels, scores)
    f1 = f1_score(labels, preds)
    return loss_sum / (i + 1), acc, auroc, f1


def train(
    train_csv, graphs_path, rows_per_batch=1024, lr=1e-3, weight_decay=5e-4,
    n_epochs=200, patience=40, n_rel_types=86, val_fraction=0.2, checkpoint_path=None,
):
    graphs = torch.load(graphs_path, weights_only=False)
    full_dataset = DrugPairDataset(train_csv, graphs)

    idx_train, idx_val = train_test_split(range(len(full_dataset)), test_size=val_fraction, random_state=42)
    collate = make_collate_fn(graphs)
    train_loader = DataLoader(Subset(full_dataset, idx_train), batch_size=rows_per_batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Subset(full_dataset, idx_val), batch_size=rows_per_batch * 3, shuffle=False, collate_fn=collate)

    print(f"train rows: {len(idx_train)}, val rows: {len(idx_val)} "
          f"({rows_per_batch*2} pairs/batch, {len(train_loader)} batches/epoch)")

    model = HDN_DDI(in_dim=55, hidden_dim=64, n_blocks=6, heads=2, n_rel_types=n_rel_types).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)

    best_mean_metric, best_epoch = 0.0, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        start = time.time()
        loss_sum = 0.0
        for i, (batch_x, batch_y, rels, labels) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            rels, labels = rels.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            scores = model(batch_x, batch_y, rels)
            loss = torch.nn.functional.binary_cross_entropy(scores, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        train_loss = loss_sum / (i + 1)
        val_loss, val_acc, val_auroc, val_f1 = evaluate(model, val_loader, DEVICE)
        mean_metric = (val_acc + val_auroc + val_f1) / 3
        scheduler.step()
        elapsed = time.time() - start

        is_best = mean_metric > best_mean_metric
        if is_best:
            best_mean_metric, best_epoch = mean_metric, epoch
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)

        flag = "*" if is_best else " "
        print(f"epoch {epoch:3d}{flag} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_acc {val_acc:.4f} | val_auroc {val_auroc:.4f} | val_f1 {val_f1:.4f} | {elapsed:.1f}s")

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}, best epoch: {best_epoch} (mean_metric={best_mean_metric:.4f})")
            break

    return model


if __name__ == "__main__":
    train(
        train_csv="data/raw/DrugBank/warm_start/fold0/train.csv",
        graphs_path="data/processed/drugbank_graphs.pt",
        checkpoint_path="results/hdn_ddi_warmstart_fold0_v5.pt",
    )
