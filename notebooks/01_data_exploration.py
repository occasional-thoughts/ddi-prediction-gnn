"""
Phase 1 data exploration: DrugBank, Twosides, BIOSNAP.
Checks SMILES validity, interaction-type distributions, and cross-dataset drug overlap.
"""
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit's per-molecule warnings

RAW = "/Users/abish/Downloads/DDI-Prediction-Project/data/raw"


def check_smiles_validity(df, smiles_col="smiles", label=""):
    n = len(df)
    valid = df[smiles_col].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    n_valid = valid.sum()
    print(f"[{label}] {n_valid}/{n} SMILES parse successfully with RDKit ({n - n_valid} invalid)")
    if n_valid < n:
        print(f"[{label}] invalid rows:\n{df[~valid]}")
    return valid


print("=" * 60)
print("DRUGBANK")
print("=" * 60)
db_smiles = pd.read_csv(f"{RAW}/DrugBank/drug_smiles.csv")
db_ddis = pd.read_csv(f"{RAW}/DrugBank/ddis.csv")
print(f"drugs: {len(db_smiles)}, ddi pairs: {len(db_ddis)}")
check_smiles_validity(db_smiles, label="DrugBank")

print("\ninteraction type distribution (86 types expected):")
type_counts = db_ddis["type"].value_counts().sort_index()
print(f"  number of distinct types: {db_ddis['type'].nunique()}")
print(f"  most common type: {type_counts.idxmax()} ({type_counts.max()} pairs)")
print(f"  least common type: {type_counts.idxmin()} ({type_counts.min()} pairs)")
print(f"  mean pairs/type: {type_counts.mean():.0f}, median: {type_counts.median():.0f}")

print("\ndrug degree (how many interactions each drug has):")
deg = pd.concat([db_ddis["d1"], db_ddis["d2"]]).value_counts()
print(f"  min degree: {deg.min()}, max degree: {deg.max()}, mean: {deg.mean():.1f}")

print("\n" + "=" * 60)
print("TWOSIDES")
print("=" * 60)
ts_smiles = pd.read_csv(f"{RAW}/Twosides/drug_smiles.csv")
print(f"drugs: {len(ts_smiles)}")
check_smiles_validity(ts_smiles, label="Twosides")

ts_ddis = pd.read_csv(f"{RAW}/Twosides/ddis.csv", nrows=200_000)  # full file is 195MB, sample first
print(f"ddi pairs (sampled 200k rows): {len(ts_ddis)}, columns: {list(ts_ddis.columns)}")
print(f"distinct interaction types in this sample: {ts_ddis['type'].nunique()} (963 expected across full file)")

print("\n" + "=" * 60)
print("BIOSNAP")
print("=" * 60)
biosnap = pd.read_csv(f"{RAW}/BIOSNAP/ChCh-Miner_durgbank-chem-chem.tsv", sep="\t", header=None, names=["d1", "d2"])
print(f"ddi pairs: {len(biosnap)}")
biosnap_drugs = set(biosnap["d1"]) | set(biosnap["d2"])
print(f"distinct drugs: {len(biosnap_drugs)}")

print("\ncross-dataset overlap: how many BIOSNAP drugs have SMILES in DrugBank's table?")
db_drug_ids = set(db_smiles["drug_id"])
overlap = biosnap_drugs & db_drug_ids
print(f"  {len(overlap)}/{len(biosnap_drugs)} BIOSNAP drugs found in DrugBank's drug_smiles.csv")
missing = biosnap_drugs - db_drug_ids
if missing:
    print(f"  {len(missing)} missing drug IDs (sample): {list(missing)[:10]}")
