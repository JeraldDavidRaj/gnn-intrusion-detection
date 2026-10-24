
from collections import Counter

import pandas as pd

from src.data_preprocessing import (
    discover_csv_files,
    METADATA_COLUMNS,
    FEATURE_COLUMNS,
)

CHUNK_SIZE = 100_000

files = discover_csv_files()

total_rows = 0
total_blank_rows = 0
total_missing_features = 0
label_counts = Counter()

print("=" * 60)
print("CIC-IDS2017 DATASET VALIDATION")
print("=" * 60)
print(f"Dataset folder: {files[0].parent.resolve()}")
print(f"CSV files found: {len(files)}")
print(f"Chunk size: {CHUNK_SIZE:,}\n")

for file_number, file_path in enumerate(files, start=1):
    print(f"[{file_number}/{len(files)}] {file_path.name}")

    file_rows = 0
    file_labels = Counter()

    for chunk in pd.read_csv(
        file_path,
        encoding="cp1252",
        low_memory=False,
        chunksize=CHUNK_SIZE,
    ):
        chunk.columns = chunk.columns.str.strip()

        blank_mask = chunk.isna().all(axis=1)
        total_blank_rows += int(blank_mask.sum())

        chunk = chunk.loc[~blank_mask].copy()

        required_columns = METADATA_COLUMNS + FEATURE_COLUMNS
        missing_columns = [
            column
            for column in required_columns
            if column not in chunk.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing: {missing_columns}"
            )

        # Remove rows without usable endpoints or labels.
        for column in ["Source IP", "Destination IP", "Label"]:
            chunk[column] = (
                chunk[column].astype("string").str.strip()
            )

        valid_mask = (
            chunk["Source IP"].notna()
            & chunk["Destination IP"].notna()
            & chunk["Label"].notna()
            & chunk["Source IP"].ne("")
            & chunk["Destination IP"].ne("")
            & chunk["Label"].ne("")
        )

        chunk = chunk.loc[valid_mask]

        numeric_features = chunk[FEATURE_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )

        numeric_features = numeric_features.replace(
            [float("inf"), float("-inf")],
            float("nan"),
        )

        total_missing_features += int(
            numeric_features.isna().sum().sum()
        )

        file_rows += len(chunk)
        file_labels.update(
            chunk["Label"].value_counts().to_dict()
        )

    total_rows += file_rows
    label_counts.update(file_labels)

    print(f"  Valid labelled rows: {file_rows:,}")
    print(f"  Distinct labels: {len(file_labels)}\n")

print("=" * 60)
print("FULL DATASET SCAN RESULTS")
print("=" * 60)
print(f"Files scanned: {len(files)}")
print(f"Valid labelled rows: {total_rows:,}")
print(f"Completely blank rows: {total_blank_rows:,}")
print(f"Missing/invalid numeric feature values: {total_missing_features:,}")

print("\nLabel distribution:")
for label, count in label_counts.most_common():
    print(f"{label}: {count:,}")

print("\nDataset validation completed successfully.")
