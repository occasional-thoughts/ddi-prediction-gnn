"""
Phase 4: cold-start validation, folds 0-2.

Trains the verified baseline config (no edge features -- same hyperparameters as the
locked-in 89.40% warm-start result: Adam, lr=1e-3, 256 rows/batch, 6 blocks, 2 heads,
30 epochs) on each cold-start fold's train.csv, validating against that fold's own
val.csv (cold-start folds ship a dedicated one, unlike warm-start's single train.csv
that we split ourselves). After training, evaluates on both standard cold-start test
partitions:
  S1 (s1.csv) - pairs where exactly one drug is unseen during training
  S2 (s2.csv) - pairs where both drugs are unseen during training
matching the protocol SSI-DDI/DSN-DDI/HDN-DDI all report against.

No cold-start fold (including fold0) had been run before this script.
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


def evaluate_full(model, csv_path, graphs, batch_size=512):
    dataset = DrugPairDataset(csv_path, graphs)
    loader = DataLoader(dataset, batch_size=batch_size // 2, shuffle=False, collate_fn=make_collate_fn(graphs))
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y, rels, labels in loader:
            scores = model(batch_x, batch_y, rels)
            all_scores.append(scores)
            all_labels.append(labels)
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (scores > 0.5).astype(int)
    return {
        "acc": accuracy_score(labels, preds),
        "auroc": roc_auc_score(labels, scores),
        "aupr": average_precision_score(labels, scores),
        "f1": f1_score(labels, preds),
        "n": len(labels),
    }


def train_fold(fold, graphs, n_epochs=30, rows_per_batch=256, lr=1e-3, n_rel_types=86):
    base = f"data/raw/DrugBank/cold_start/fold{fold}"

    train_dataset = DrugPairDataset(f"{base}/train.csv", graphs)
    val_dataset = DrugPairDataset(f"{base}/val.csv", graphs)
    collate = make_collate_fn(graphs)
    train_loader = DataLoader(train_dataset, batch_size=rows_per_batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=rows_per_batch, shuffle=False, collate_fn=collate)

    print(f"[fold{fold}] train rows: {len(train_dataset)}, val rows: {len(val_dataset)} "
          f"({len(train_loader)} batches/epoch)", flush=True)

    model = HDN_DDI(in_dim=55, hidden_dim=64, n_blocks=6, heads=2, n_rel_types=n_rel_types,
                     use_edge_features=False).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = f"results/hdn_ddi_coldstart_fold{fold}.pt"
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

        model.eval()
        correct, total, vloss_sum = 0, 0, 0.0
        with torch.no_grad():
            for j, (batch_x, batch_y, rels, labels) in enumerate(val_loader):
                if j >= 20:
                    break
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                rels, labels = rels.to(DEVICE), labels.to(DEVICE)
                scores = model(batch_x, batch_y, rels)
                vloss_sum += torch.nn.functional.binary_cross_entropy(scores, labels).item()
                correct += ((scores > 0.5).float() == labels).sum().item()
                total += len(labels)
        val_loss, val_acc = vloss_sum / (j + 1), correct / total
        elapsed = time.time() - start
        print(f"[fold{fold}] epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_acc {val_acc:.4f} | {elapsed:.1f}s", flush=True)
        torch.save(model.state_dict(), ckpt_path)

    print(f"[fold{fold}] training done, evaluating on S1/S2...", flush=True)
    s1 = evaluate_full(model, f"{base}/s1.csv", graphs)
    s2 = evaluate_full(model, f"{base}/s2.csv", graphs)
    print(f"[fold{fold}] S1 (n={s1['n']}): ACC {s1['acc']*100:.2f}% AUROC {s1['auroc']*100:.2f}% "
          f"AUPR {s1['aupr']*100:.2f}% F1 {s1['f1']*100:.2f}%", flush=True)
    print(f"[fold{fold}] S2 (n={s2['n']}): ACC {s2['acc']*100:.2f}% AUROC {s2['auroc']*100:.2f}% "
          f"AUPR {s2['aupr']*100:.2f}% F1 {s2['f1']*100:.2f}%", flush=True)
    return s1, s2


if __name__ == "__main__":
    folds = [int(f) for f in sys.argv[1:]] if len(sys.argv) > 1 else [0, 1, 2]
    graphs = torch.load("data/processed/drugbank_graphs.pt", weights_only=False)
    results = {}
    for fold in folds:
        results[fold] = train_fold(fold, graphs)
    print("\n=== Cold-start summary ===", flush=True)
    for fold, (s1, s2) in results.items():
        print(f"fold{fold}: S1 ACC {s1['acc']*100:.2f}% | S2 ACC {s2['acc']*100:.2f}%", flush=True)
