# Project Progress

## Phase 1 — Get and understand the data ✅ DONE
- DrugBank (1,706 drugs, 191,808 pairs), Twosides (645 drugs, 4.58M pairs), BIOSNAP (1,514 drugs, 48,514 pairs) all downloaded and verified against published paper statistics exactly.
- 100% SMILES validity confirmed on DrugBank and Twosides.
- BIOSNAP: 1,273/1,514 drugs already covered by DrugBank's SMILES table.

## Phase 2 — Turn drugs into graphs ✅ DONE
- `src/molecular_graph.py`: SMILES → RDKit molecule → BRICS fragment decomposition → 3-level hierarchical graph (atom / substructure / molecule nodes), following HDN-DDI's paper description.
- Ran on all 1,706 DrugBank + 645 Twosides drugs, 100% success, cached to `data/processed/*.pt`.
- **Known finding**: HDN-DDI's own public GitHub code does *not* implement this hierarchical BRICS module — we built it independently from the paper's written Methods section.

## Phase 3 — Reimplement the baseline model (HDN-DDI) 🔶 IN PROGRESS
- ✅ 3a: Model architecture (`src/model.py`) — hierarchical-view GAT, interactive-view bipartite GAT, update layer, 6-block encoder, co-attention decoder. Verified correct via real forward+backward pass.
- ✅ 3b: Batched training — reworked to process many drug pairs per step instead of one at a time. **6.8x speedup** (67 min/epoch → 9.8 min/epoch on CPU). GPU (MPS) tested and found currently slower due to per-pair bookkeeping overhead — staying on CPU.
- ⬜ 3c: Data loader (reading DrugBank's train/test CSVs + negative sampling into batches) — **next up**
- ⬜ 3d: Full training loop + first baseline training run

## Phase 4 — Validate against warm-start and cold-start ⬜ NOT STARTED
Compare our trained model's ACC/AUROC/AUPR/F1 against HDN-DDI's published numbers to confirm the reimplementation is correct.

## Phase 5 — Research contribution ⬜ NOT STARTED
- Gap 1: Edge-aware attention (bond chemistry currently ignored, as in all 4 base papers)
- Gap 2: Cross-dataset generalization to BIOSNAP (never tested by any paper in this lineage)
- Gap 4: Quantitative interpretability evaluation (replacing hand-picked case studies with real precision/recall)

## Phase 6 — Ablation study ⬜ NOT STARTED

## Phase 7 — Write up ⬜ NOT STARTED
Literature review + research gap analysis already drafted (see conversation history / can be regenerated on request).

## Phase 8 — Demo/UI (optional, stretch goal) ⬜ NOT STARTED
