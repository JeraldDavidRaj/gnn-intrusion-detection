
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "cic_ids2017_cleaned.csv"

print("=" * 60)
print("FULL CLEANED DATASET VALIDATION")
print("=" * 60)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATASET_PATH}"
    )

print(f"Dataset: {DATASET_PATH}")
print(f"File size: {DATASET_PATH.stat().st_size / (1024**2):.2f} MB")

# Read the complete dataset in chunks to avoid loading it all into memory.
total_rows = 0
blank_rows = 0
missing_labels = 0
label_counts = {}
source_ips = set()
destination_ips = set()
expected_columns = None

for chunk in pd.read_csv(
    DATASET_PATH,
    chunksize=100_000,
    low_memory=False
):
    if expected_columns is None:
        expected_columns = list(chunk.columns)

    blank_mask = chunk.isna().all(axis=1)
    blank_rows += int(blank_mask.sum())
    chunk = chunk.loc[~blank_mask].copy()

    total_rows += len(chunk)
    missing_labels += int(chunk["Label"].isna().sum())

    counts = chunk["Label"].fillna("MISSING").value_counts()
    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + int(count)

    source_ips.update(chunk["Source IP"].dropna().astype(str).unique())
    destination_ips.update(
        chunk["Destination IP"].dropna().astype(str).unique()
    )

print("\nVALIDATION RESULTS")
print("-" * 60)
print(f"Total valid rows: {total_rows:,}")
print(f"Completely blank rows: {blank_rows:,}")
print(f"Missing labels: {missing_labels:,}")
print(f"Number of columns: {len(expected_columns)}")
print(f"Unique source IPs: {len(source_ips):,}")
print(f"Unique destination IPs: {len(destination_ips):,}")

print("\nFULL LABEL DISTRIBUTION")
for label, count in sorted(
    label_counts.items(), key=lambda item: item[1], reverse=True
):
    print(f"{label}: {count:,}")

print("\nDataset validation completed.")
