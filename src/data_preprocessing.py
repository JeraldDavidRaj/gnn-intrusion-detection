
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "Dataset" / "TrafficLabelling"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "cic_ids2017_cleaned.csv"


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

METADATA_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Label",
]

FEATURE_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Packet Length Mean",
    "Packet Length Std",
    "SYN Flag Count",
    "ACK Flag Count",
    "RST Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Active Mean",
    "Idle Mean",
]

REQUIRED_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS

CHUNK_SIZE = 100_000


# ============================================================
# 3. DISCOVER DATASET FILES
# ============================================================

def discover_csv_files(dataset_dir=DATASET_DIR):
    """Find CSV files inside the CIC-IDS2017 dataset folder."""

    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}"
        )

    csv_files = sorted(dataset_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found inside: {dataset_dir}"
        )

    return csv_files


# ============================================================
# 4. CLEAN A DATAFRAME
# ============================================================

def clean_flows(df):
    """Clean metadata, labels and selected numeric features."""

    df.columns = df.columns.str.strip()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    # Remove completely blank rows.
    df = df.dropna(how="all")

    # Clean text columns.
    text_columns = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Label",
        "Timestamp",
    ]

    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    # Remove rows without valid endpoint or label information.
    valid_rows = (
        df["Source IP"].notna()
        & df["Destination IP"].notna()
        & df["Label"].notna()
        & df["Source IP"].ne("")
        & df["Destination IP"].ne("")
        & df["Label"].ne("")
    )

    df = df.loc[valid_rows].copy()

    # Convert selected traffic features to numeric values.
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Replace infinite values with missing values.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    return df


# ============================================================
# 5. LOAD A SINGLE CSV
# ============================================================

def load_flows(file_path, nrows=None):
    """Load and clean one CSV file."""

    df = pd.read_csv(
        file_path,
        encoding="cp1252",
        low_memory=False,
        nrows=nrows,
    )

    return clean_flows(df).reset_index(drop=True)


# ============================================================
# 6. DATASET SUMMARY
# ============================================================

def summarize_flows(df):
    """Generate a summary of the loaded traffic flows."""

    return {
        "flow_count": len(df),
        "unique_source_ips": df["Source IP"].nunique(),
        "unique_destination_ips": df["Destination IP"].nunique(),
        "label_counts": df["Label"].value_counts().to_dict(),
        "missing_feature_values": int(
            df[FEATURE_COLUMNS].isna().sum().sum()
        ),
    }


# ============================================================
# 7. PROCESS THE FULL DATASET IN CHUNKS
# ============================================================

def preprocess_dataset(
    dataset_dir=DATASET_DIR,
    output_path=OUTPUT_FILE,
    chunksize=CHUNK_SIZE,
):
    """Process every CSV and save the cleaned flows to one CSV."""

    csv_files = discover_csv_files(dataset_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start a fresh output file.
    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    total_blank_rows = 0
    total_missing_features = 0
    label_counts = {}

    print("=" * 65)
    print("CIC-IDS2017 FULL DATASET PREPROCESSING")
    print("=" * 65)
    print(f"Dataset folder: {Path(dataset_dir).resolve()}")
    print(f"CSV files found: {len(csv_files)}")
    print(f"Chunk size: {chunksize:,}\n")

    for file_number, file_path in enumerate(csv_files, start=1):
        print(
            f"[{file_number}/{len(csv_files)}] "
            f"Processing: {file_path.name}"
        )

        file_rows = 0

        for chunk in pd.read_csv(
            file_path,
            encoding="cp1252",
            low_memory=False,
            chunksize=chunksize,
        ):
            # Count fully blank rows before cleaning.
            blank_mask = chunk.isna().all(axis=1)
            total_blank_rows += int(blank_mask.sum())

            cleaned = clean_flows(chunk)

            if cleaned.empty:
                continue

            total_missing_features += int(
                cleaned[FEATURE_COLUMNS].isna().sum().sum()
            )

            counts = cleaned["Label"].value_counts()

            for label, count in counts.items():
                label_counts[label] = (
                    label_counts.get(label, 0) + int(count)
                )

            # Write the first chunk with headers, later chunks without.
            cleaned.to_csv(
                output_path,
                mode="a",
                header=(total_rows == 0),
                index=False,
            )

            rows_written = len(cleaned)
            total_rows += rows_written
            file_rows += rows_written

        print(f"  Valid flows written: {file_rows:,}\n")

    if total_rows == 0:
        raise ValueError(
            "No valid flows were processed. Check the dataset files."
        )

    print("=" * 65)
    print("PREPROCESSING COMPLETED")
    print("=" * 65)
    print(f"CSV files processed: {len(csv_files)}")
    print(f"Total valid flows: {total_rows:,}")
    print(f"Completely blank rows: {total_blank_rows:,}")
    print(f"Missing numeric feature values: {total_missing_features:,}")
    print(f"Selected numeric features: {len(FEATURE_COLUMNS)}")
    print(f"Output file: {output_path.resolve()}")
    print(f"Output size: {output_path.stat().st_size / (1024 ** 2):.2f} MB")

    print("\nOverall label distribution:")
    for label, count in sorted(
        label_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {label}: {count:,}")

    return output_path


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    preprocess_dataset()

