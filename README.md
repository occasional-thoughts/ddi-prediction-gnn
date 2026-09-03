# Bond-Aware Hierarchical Graph Neural Network for Generalizable Drug–Drug Interaction Prediction

A graph neural network for predicting drug–drug interactions (DDIs) directly from molecular structure, following the hierarchical dual-view design introduced by **HDN-DDI** (Sun & Zheng, 2025, *BMC Bioinformatics*), and extended with an original bond-aware attention contribution.

## Problem

Drug–drug interactions are a leading cause of adverse clinical outcomes. Existing GNN-based DDI predictors (SSI-DDI, GMPNN-CS, DSN-DDI, HDN-DDI) predict interactions from molecular graphs, but:
- **Ignore real bond-level chemistry** — naively adding bond features has been shown to *degrade* accuracy in this model lineage
- **Are validated on only one dataset** — none test whether learned chemistry generalizes to a differently-curated interaction resource
- **Never quantify their own interpretability claims** — every paper's "the model found the right functional group" case study is a handful of hand-picked examples, not a measured metric
- **Are never checked for true-negative calibration** — accuracy is measured against a single fixed candidate interaction type per pair, never against "is there any interaction at all"

## This project

1. **Implements** a hierarchical dual-view GNN for DDI prediction (`src/model.py`), built directly from HDN-DDI's published equations (Eq. 1–12).
2. **Bond-aware attention (Gap 1) — done.** Real bond chemistry (bond type, conjugation, ring membership) is incorporated into the hierarchical-view GAT's attention score via PyTorch Geometric's `edge_dim` mechanism — the principled formula, not the naive constant-gate approach prior work reports degrades performance. Result: 89.80% ACC (+0.40 over the 89.40% baseline), with gains on AUROC, AUPR, and F1 too, no tradeoffs.
3. **Cross-dataset generalization (Gap 2) — not started.** Training on DrugBank, evaluating transfer to BIOSNAP.
4. **Quantitative interpretability evaluation (Gap 4) — not started.** Replacing hand-picked case studies with a measurable attribution precision/recall benchmark.

## Architecture

Each drug is decomposed into a 3-level hierarchical graph (atom → BRICS chemical substructure → whole molecule) via RDKit. A 6-block dual-view encoder alternates between:
- a **hierarchical-view** GAT (within one drug's own graph, bond-aware via edge features)
- an **interactive-view** GAT (a bipartite attention layer between the two drugs' substructures)

followed by a co-attention decoder that scores the interaction across all block depths (Eq. 10–11) against a specific candidate interaction type.

Two revisions were tried and reverted after measuring worse than the simpler configuration above: matching the paper's interactive-view equations more literally (explicit self-loops, weight-sharing with the plain transform), and matching the authors' exact training hyperparameters extracted from their own released logs (weight decay, LR decay, larger batch size, longer training). Both are documented in `docs/PROGRESS.md` rather than silently dropped.

**Note on reproducing HDN-DDI's published numbers:** their [publicly released code](https://github.com/jcsun-00/HDN-DDI) does not implement the hierarchical BRICS-based module their paper describes — `transductive_train.py`/`models.py` in that repository implement a simpler architecture closer to DSN-DDI's design. This project's encoder/decoder were built directly from the paper's written equations instead of the authors' code, which is why this baseline is compared against its own verified, from-scratch result (89.40% ACC) rather than against their reported number.

## Known limitation: true-negative calibration

The model has only ever been evaluated on picking the correct interaction type *given that a documented interaction exists* — that's what the reported accuracy numbers measure. Testing it against 45 pairs verified (via the dataset's own negative-sampling construction) to have **no** documented interaction, only 1 was correctly identified as non-interacting when scored across all 86 types and thresholded — the model is not calibrated to reject unlikely pairs, only to rank likely interaction types. This is an open gap, not a solved problem; see `docs/PROGRESS.md` for the full test.

## Try it

```bash
pip install -r requirements.txt
python src/demo_predict.py DB00460 DB04571          # CLI: predicts a real interaction, live
streamlit run app.py                                 # same thing, as a web UI
```

## Repository structure

```
src/
  molecular_graph.py    # SMILES -> hierarchical (atom/substructure/molecule) graph via RDKit + BRICS
  model.py               # HDN-DDI encoder + decoder, reimplemented from the paper's equations
  dataset.py              # DrugBank pair loading + negative sampling (handles both the
                           #   warm-start "Neg samples" and cold-start "split" column layouts)
  train.py                 # warm-start training loop
  train_coldstart.py        # cold-start training + S1/S2 evaluation, all 3 folds
  full_evaluate.py         # ACC / AUROC / AUPR / F1 on a held-out warm-start test set
  demo_predict.py           # CLI: predict a live interaction for any two drugs
notebooks/
  01_data_exploration.py  # dataset statistics, SMILES validity checks
  02_build_graphs.py      # batch-converts a drug database into cached hierarchical graphs
app.py                    # Streamlit demo UI (model selector, example pairs, full 86-type breakdown)
data/raw/DrugBank/
  interaction_types.csv   # this project's 0-85 type codes -> real text descriptions
                           #   (source: the original DeepDDI dataset release; see PROGRESS.md
                           #   for how the numbering offset was verified)
  drug_names.csv           # DrugBank ID -> common name (source: dhimmel/drugbank)
docs/
  PROGRESS.md              # detailed phase-by-phase project log
```

## Datasets

Not included in this repository — DrugBank's terms of use restrict redistribution, and Twosides/BIOSNAP are large. To reproduce:
- **DrugBank**: [go.drugbank.com](https://go.drugbank.com/) (free academic license) — 1,706 drugs, 191,808 labeled interactions, 86 interaction types
- **Twosides**: derived by [Zitnik et al.](https://doi.org/10.1093/bioinformatics/bty294) — 645 drugs, 4.58M interactions, 963 side-effect types
- **BIOSNAP**: [Stanford SNAP ChCh-Miner](http://snap.stanford.edu/biodata/datasets/10001/10001-ChCh-Miner.html) — 1,514 drugs, 48,514 untyped interaction pairs

## Current Status

- **Phase 1–4 (data, graph pipeline, baseline model, warm/cold-start validation): done.** Baseline verified at 89.40% ACC (warm-start) / 59.62–71.82% ACC (cold-start S1/S2 average across 3 folds).
- **Phase 5, Gap 1 (bond-aware attention): done**, 89.80% ACC, clean improvement on all four metrics.
- **Phase 5, Gap 2 (BIOSNAP generalization) and Gap 4 (interpretability): not started.**
- **Phase 8 (demo): done** — CLI and Streamlit UI, see "Try it" above.

See `docs/PROGRESS.md` for the detailed phase-by-phase log.

## References

- Sun, J., & Zheng, H. (2025). HDN-DDI: a novel framework for predicting drug-drug interactions using hierarchical molecular graphs and enhanced dual-view representation learning. *BMC Bioinformatics*, 26, 28.
- Li, Z. et al. (2023). DSN-DDI: an accurate and generalized framework for drug–drug interaction prediction by dual-view representation learning. *Briefings in Bioinformatics*, 24(1), bbac597.
- Nyamabo, A.K. et al. (2022). Drug–drug interaction prediction with learnable size-adaptive molecular substructures. *Briefings in Bioinformatics*, 23(1), bbab441.
- Nyamabo, A.K. et al. (2021). SSI–DDI: substructure–substructure interactions for drug–drug interaction prediction. *Briefings in Bioinformatics*, 22(6), bbab133.
- Ryu, J.Y., Kim, H.U., & Lee, S.Y. (2018). Deep learning improves prediction of drug–drug and drug–food interactions. *PNAS*, 115(18), E4304-E4311.
