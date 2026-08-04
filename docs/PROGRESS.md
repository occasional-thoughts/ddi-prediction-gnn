# Project Progress

## Phase 1 — Get and understand the data ✅ DONE
- DrugBank (1,706 drugs, 191,808 pairs), Twosides (645 drugs, 4.58M pairs), BIOSNAP (1,514 drugs, 48,514 pairs) all downloaded and verified against published paper statistics exactly.
- 100% SMILES validity confirmed on DrugBank and Twosides.
- BIOSNAP: 1,273/1,514 drugs already covered by DrugBank's SMILES table.

## Phase 2 — Turn drugs into graphs ✅ DONE
- `src/molecular_graph.py`: SMILES → RDKit molecule → BRICS fragment decomposition → 3-level hierarchical graph (atom / substructure / molecule nodes), following HDN-DDI's paper description.
- Ran on all 1,706 DrugBank + 645 Twosides drugs, 100% success, cached to `data/processed/*.pt`.
- **Known finding**: HDN-DDI's own public GitHub code does *not* implement this hierarchical BRICS module — we built it independently from the paper's written Methods section.
- Extended for Gap 1 (see Phase 5): every atom-atom edge now also carries a 6-dim real bond-chemistry feature (bond type, conjugation, ring membership).

## Phase 3 — Reimplement the baseline model (HDN-DDI) ✅ DONE
- Model architecture (`src/model.py`) — hierarchical-view GAT, interactive-view bipartite GAT, update layer, 6-block encoder, co-attention decoder.
- Batched training — **6.8x speedup** (67 min/epoch → 9.8 min/epoch on CPU). GPU (MPS) tested and found slower for this workload due to per-pair bookkeeping overhead — training stays on CPU.
- Data loader (`src/dataset.py`) — reads DrugBank's train/test CSVs, uses the dataset's own pre-generated negative samples.
- Training loop (`src/train.py`) + evaluation (`src/full_evaluate.py`).

## Phase 4 — Validate against warm-start and cold-start 🔶 IN PROGRESS
- **Verified working baseline: 89.40% ACC / 95.23% AUROC / 94.00% AUPR / 89.71% F1** on DrugBank warm-start test set (vs. paper's published 97.93% / 99.73% / 99.72% / 98.03%).
- Multiple fix attempts tried and compared head-to-head: found and fixed a real architecture bug (update layer was 64-dim instead of the paper's specified 128-dim, +0.6pt); tried matching the paper's equations more literally in the interactive-view layer (self-loops, weight-sharing) — measured worse; tried matching the authors' exact training hyperparameters extracted from their real run logs (weight decay, LR decay, 1024-batch, 200-epoch early stopping) — measured worse still, plateaued ~71%.
- **Key finding**: dug into the authors' GitHub repo's `log/` folder and `transductive_train.py` (not just the paper text) and confirmed their released code implements a different, simpler architecture (DSN-DDI's design) than the hierarchical BRICS module the paper describes — their own repo doesn't contain what the paper claims to introduce. This explains why chasing their exact number on our from-scratch, paper-faithful implementation has diminishing returns.
- **Decision**: stopped chasing exact number-matching; reverted to and locked in the 89.40% configuration as the verified baseline (confirmed via `strict=True` state_dict loading — zero mismatches). This is the baseline all Phase 5 contributions are compared against.

## Phase 5 — Research contribution 🔶 IN PROGRESS
- ✅ **Gap 1: Edge-aware attention** — incorporated real bond chemistry (bond type, conjugation, ring membership) into the hierarchical-view GAT's attention score via PyG's principled `edge_dim` mechanism, not the naive constant-gate approach that SSI-DDI/DSN-DDI/HDN-DDI all report hurts performance.
  **Result: clean improvement across all four metrics** — 89.80% ACC (+0.40), 95.56% AUROC (+0.33), 94.54% AUPR (+0.54), 89.97% F1 (+0.26) vs. the 89.40% baseline. No tradeoffs. This is the project's core positive finding so far.
- ⬜ Gap 2: Cross-dataset generalization to BIOSNAP (never tested by any paper in this lineage)
- ⬜ Gap 4: Quantitative interpretability evaluation (replacing hand-picked case studies with real precision/recall)

## Phase 6 — Ablation study ⬜ NOT STARTED
Phase 5's Gap 1 result above (edge-aware vs. baseline) already functions as one clean ablation; formal write-up of this comparison plus further ablations (e.g. bond-feature components individually) still to do.

## Phase 7 — Write up ⬜ NOT STARTED
Literature review + research gap analysis already drafted (see conversation history / can be regenerated on request).

## Phase 8 — Demo/UI (optional, stretch goal) ⬜ NOT STARTED
