# Bond-Aware Hierarchical Graph Neural Network for Generalizable Drug–Drug Interaction Prediction

A graph neural network framework for predicting drug–drug interactions (DDIs) directly from molecular structure, built on top of **HDN-DDI** (Sun & Zheng, 2025, *BMC Bioinformatics*) — the current state-of-the-art hierarchical dual-view GNN for this task — and extended with three original contributions.

## Problem

Drug–drug interactions are a leading cause of adverse clinical outcomes. Existing GNN-based DDI predictors (SSI-DDI, GMPNN-CS, DSN-DDI, HDN-DDI) predict interactions from molecular graphs, but:
- **Ignore real bond-level chemistry** — naively adding bond features has been shown to *degrade* accuracy in this entire model lineage
- **Are validated on only one dataset** — none test whether learned chemistry generalizes to a differently-curated interaction resource
- **Never quantify their own interpretability claims** — every paper's "the model found the right functional group" case study is a handful of hand-picked examples, not a measured metric

## This project

1. **Reimplements HDN-DDI** from its published equations — its own public repository does not actually contain the hierarchical BRICS-based substructure module described in the paper, so the encoder/decoder here (`src/model.py`) was built directly from the paper's Eq. 1–12, not copied from the authors' code.
2. **Edge-aware attention** *(in progress)* — incorporating real bond chemistry into the GAT attention mechanism without the degradation prior work reports.
3. **Cross-dataset generalization** *(planned)* — training on DrugBank/Twosides, evaluating transfer to BIOSNAP, a differently-curated DDI network no prior paper in this lineage has tested against.
4. **Quantitative interpretability evaluation** *(planned)* — replacing hand-picked case studies with a measurable attribution precision/recall benchmark.

## Architecture

Each drug is decomposed into a 3-level hierarchical graph (atom → BRICS chemical substructure → whole molecule) via RDKit. A 6-block dual-view encoder alternates between:
- a **hierarchical-view** GAT (within one drug's own graph)
- an **interactive-view** GAT (a bipartite attention layer between the two drugs' substructures, with weight-sharing and self-loop terms matching the paper's Eq. 4–7)

followed by a co-attention decoder that scores the interaction across all block depths (Eq. 10–11).

## Repository structure

```
src/
  molecular_graph.py   # SMILES -> hierarchical (atom/substructure/molecule) graph via RDKit + BRICS
  model.py              # HDN-DDI encoder + decoder, reimplemented from the paper's equations
  dataset.py             # DrugBank/Twosides pair loading + negative sampling
  train.py                # training loop (matches the authors' verified hyperparameters:
                           #   Adam + weight decay, exponential LR decay, early stopping)
  full_evaluate.py       # ACC / AUROC / AUPR / F1 on a held-out test set
notebooks/
  01_data_exploration.py # dataset statistics, SMILES validity checks
  02_build_graphs.py     # batch-converts a drug database into cached hierarchical graphs
docs/
  PROGRESS.md             # project status log
```

## Datasets

Not included in this repository — DrugBank's terms of use restrict redistribution, and Twosides/BIOSNAP are large. To reproduce:
- **DrugBank**: [go.drugbank.com](https://go.drugbank.com/) (free academic license) — 1,706 drugs, 191,808 labeled interactions, 86 interaction types
- **Twosides**: derived by [Zitnik et al.](https://doi.org/10.1093/bioinformatics/bty294) — 645 drugs, 4.58M interactions, 963 side-effect types
- **BIOSNAP**: [Stanford SNAP ChCh-Miner](http://snap.stanford.edu/biodata/datasets/10001/10001-ChCh-Miner.html) — 1,514 drugs, 48,514 untyped interaction pairs

## Status

Base model architecture, data pipeline, and training loop are implemented and running; currently validating against HDN-DDI's published benchmarks before building the three extensions above. See `docs/PROGRESS.md` for the detailed phase-by-phase log.

## References

- Sun, J., & Zheng, H. (2025). HDN-DDI: a novel framework for predicting drug-drug interactions using hierarchical molecular graphs and enhanced dual-view representation learning. *BMC Bioinformatics*, 26, 28.
- Li, Z. et al. (2023). DSN-DDI: an accurate and generalized framework for drug–drug interaction prediction by dual-view representation learning. *Briefings in Bioinformatics*, 24(1), bbac597.
- Nyamabo, A.K. et al. (2022). Drug–drug interaction prediction with learnable size-adaptive molecular substructures. *Briefings in Bioinformatics*, 23(1), bbab441.
- Nyamabo, A.K. et al. (2021). SSI–DDI: substructure–substructure interactions for drug–drug interaction prediction. *Briefings in Bioinformatics*, 22(6), bbab133.
