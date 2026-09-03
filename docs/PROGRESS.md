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

## Phase 4 — Validate against warm-start and cold-start ✅ DONE
- **Verified working baseline: 89.40% ACC / 95.23% AUROC / 94.00% AUPR / 89.71% F1** on DrugBank warm-start test set (vs. paper's published 97.93% / 99.73% / 99.72% / 98.03%).
- Multiple fix attempts tried and compared head-to-head: found and fixed a real architecture bug (update layer was 64-dim instead of the paper's specified 128-dim, +0.6pt); tried matching the paper's equations more literally in the interactive-view layer (self-loops, weight-sharing) — measured worse; tried matching the authors' exact training hyperparameters extracted from their real run logs (weight decay, LR decay, 1024-batch, 200-epoch early stopping) — measured worse still, plateaued ~71%.
- **Key finding**: dug into the authors' GitHub repo's `log/` folder and `transductive_train.py` (not just the paper text) and confirmed their released code implements a different, simpler architecture (DSN-DDI's design) than the hierarchical BRICS module the paper describes — their own repo doesn't contain what the paper claims to introduce. This explains why chasing their exact number on our from-scratch, paper-faithful implementation has diminishing returns.
- **Decision**: stopped chasing exact number-matching; reverted to and locked in the 89.40% configuration as the verified baseline (confirmed via `strict=True` state_dict loading — zero mismatches). This is the baseline all Phase 5 contributions are compared against.
- **Cold-start validation (baseline, no edge features), 3 folds** — `src/train_coldstart.py`, run to completion:

  | Fold | S1 ACC | S1 AUROC | S1 AUPR | S1 F1 | S2 ACC | S2 AUROC | S2 AUPR | S2 F1 |
  |---|---|---|---|---|---|---|---|---|
  | fold0 | 58.30% | 67.75% | 68.15% | 36.32% | 70.92% | 80.64% | 81.33% | 64.10% |
  | fold1 | 62.34% | 71.97% | 73.32% | 46.34% | 74.00% | 83.12% | 83.80% | 69.40% |
  | fold2 | 58.21% | 68.59% | 69.54% | 34.27% | 70.53% | 81.34% | 82.00% | 62.88% |
  | **avg** | **59.62%** | **69.44%** | **70.34%** | **38.98%** | **71.82%** | **81.70%** | **82.38%** | **65.46%** |

  S1 (one drug unseen) is consistently harder than S2 (both drugs seen, novel pairing), which is harder than warm-start — the expected ordering, holding cleanly across all three folds with no outliers. Cold-start CSVs from the dataset use a `split` column (not `Neg samples`) for the negative-sample id — `DrugPairDataset` now auto-detects which column a given file uses.

## Phase 5 — Research contribution 🔶 IN PROGRESS
- ✅ **Gap 1: Edge-aware attention** — incorporated real bond chemistry (bond type, conjugation, ring membership) into the hierarchical-view GAT's attention score via PyG's principled `edge_dim` mechanism, not the naive constant-gate approach that SSI-DDI/DSN-DDI/HDN-DDI all report hurts performance.
  **Result: clean improvement across all four metrics** — 89.80% ACC (+0.40), 95.56% AUROC (+0.33), 94.54% AUPR (+0.54), 89.97% F1 (+0.26) vs. the 89.40% baseline. No tradeoffs. This is the project's core positive finding so far.
- ⬜ Gap 2: Cross-dataset generalization to BIOSNAP (never tested by any paper in this lineage)
- ⬜ Gap 4: Quantitative interpretability evaluation (replacing hand-picked case studies with real precision/recall)

## Phase 6 — Ablation study ⬜ NOT STARTED
Phase 5's Gap 1 result above (edge-aware vs. baseline) already functions as one clean ablation; formal write-up of this comparison plus further ablations (e.g. bond-feature components individually) still to do.

## Phase 7 — Write up ⬜ NOT STARTED
Literature review + research gap analysis already drafted (see conversation history / can be regenerated on request).

## Phase 8 — Demo/UI (optional, stretch goal) ✅ DONE
- `src/demo_predict.py` — CLI: give it two drugs (DrugBank ID or raw SMILES), it builds the graphs, loads the trained model, and prints the predicted interaction, computed live (not a test-set metric).
- `app.py` — Streamlit web UI wrapping the same logic: example pairs, free-text drug input, a selector between the warm-start bond-aware model (default), the warm-start baseline, and each of the three cold-start fold checkpoints, a verdict + top-5 chart, and a "show all 86" expander with the full sorted table.
- `data/raw/DrugBank/interaction_types.csv` — maps this project's 0-85 interaction-type codes to real text descriptions. Source: the original DeepDDI dataset release (Ryu, Kim & Lee, 2018, *PNAS*, `Interaction_information.csv` at `bitbucket.org/kaistsystemsbiology/deepddi`), which is the origin of the 86-type DDI benchmark this project's DrugBank split also uses. That file's own type numbering (1-86) doesn't match this dataset's (0-85) directly, so the offset was verified rather than assumed: this project's single most frequent type (31.7% of all 191,808 pairs, by a wide margin over the next-most-frequent) maps under a −1 offset to DeepDDI's generic "risk or severity of adverse effects can be increased" category — the expected dominant class in this kind of benchmark — and 8 further high-frequency types were spot-checked and all map to coherent, plausible DDI mechanisms (metabolism, serum concentration, hypotensive/CNS effects, etc.). Not a certainty, but corroborated rather than guessed.
- `data/raw/DrugBank/drug_names.csv` — DrugBank ID → common drug name, sourced from Daniel Himmelstein's `dhimmel/drugbank` parse of DrugBank's public XML (a well-cited open-source project used in the Hetionet network-medicine work). Covers 1,412 of this project's 1,706 drugs (82.8%) — the rest are newer DrugBank entries (DB09xxx/DB11xxx range) not in that snapshot. Uncovered IDs are shown plainly rather than guessed. Both demo tools now show real names (e.g. "Verteporfin (DB00460)") instead of bare IDs, and substitute them into the interaction-type sentences.
