
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cic_ids2017_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "zero_day"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ATTACK = "PortScan"

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
    "Idle Mean"
]

METADATA_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label"
]

USE_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS

TRAIN_DATES = [
    "2017-07-03",
    "2017-07-04",
    "2017-07-05"
]

VALIDATION_DATE = "2017-07-06"
BENIGN_TEST_DATE = "2017-07-07"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("ZERO-DAY DATA PREPARATION")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=USE_COLUMNS,
    low_memory=False
)

print("Dataset shape:", df.shape)

# Clean metadata
for column in [
    "Source IP",
    "Destination IP",
    "Label"
]:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )

df["Timestamp"] = (
    df["Timestamp"]
    .astype("string")
    .str.strip()
)


# ============================================================
# 3. TIMESTAMP CONVERSION
# ============================================================

print("\n" + "=" * 70)
print("TIMESTAMP CONVERSION")
print("=" * 70)

raw_timestamps = df["Timestamp"]

# Format 1: 6/7/2017 1:00
parsed_minutes = pd.to_datetime(
    raw_timestamps,
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

# Format 2: 03/07/2017 11:04:03
parsed_seconds = pd.to_datetime(
    raw_timestamps,
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

# Format 3: fractional seconds
parsed_fractional = pd.to_datetime(
    raw_timestamps,
    format="%d/%m/%Y %H:%M:%S.%f",
    errors="coerce"
)

df["Verified_Timestamp"] = (
    parsed_minutes
    .fillna(parsed_seconds)
    .fillna(parsed_fractional)
)

valid_count = int(
    df["Verified_Timestamp"].notna().sum()
)

invalid_count = int(
    df["Verified_Timestamp"].isna().sum()
)

print("Total timestamps:", len(df))
print("Valid timestamps:", valid_count)
print("Invalid timestamps:", invalid_count)

print(
    "Earliest timestamp:",
    df["Verified_Timestamp"].min()
)

print(
    "Latest timestamp:",
    df["Verified_Timestamp"].max()
)

if invalid_count > 0:

    print("\nInvalid timestamp examples:")

    print(
        raw_timestamps[
            df["Verified_Timestamp"].isna()
        ].value_counts().head(30)
    )

    raise ValueError(
        "Invalid timestamps found. Preparation stopped."
    )

print("\nAll timestamps successfully parsed.")


# ============================================================
# 4. NUMERIC FEATURE CLEANING
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC FEATURE CLEANING")
print("=" * 70)

for column in FEATURE_COLUMNS:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df[FEATURE_COLUMNS] = df[
    FEATURE_COLUMNS
].replace(
    [np.inf, -np.inf],
    np.nan
)

missing_features = int(
    df[FEATURE_COLUMNS].isna().sum().sum()
)

print("Missing feature values:", missing_features)

print(
    "\nMissing feature values will be imputed later "
    "using training data only."
)


# ============================================================
# 5. SEPARATE BENIGN AND PORTSCAN TRAFFIC
# ============================================================

print("\n" + "=" * 70)
print("SEPARATING BENIGN AND PORTSCAN TRAFFIC")
print("=" * 70)

benign_df = df[
    df["Label"] == "BENIGN"
].copy()

portscan_df = df[
    df["Label"] == TARGET_ATTACK
].copy()

other_attack_count = (
    len(df)
    - len(benign_df)
    - len(portscan_df)
)

print("Total benign flows:", len(benign_df))
print("Total PortScan flows:", len(portscan_df))
print("Other attack flows excluded:", other_attack_count)

if benign_df.empty:
    raise ValueError("No benign flows found.")

if portscan_df.empty:
    raise ValueError("No PortScan flows found.")


# ============================================================
# 6. DATE-BASED BENIGN DATA SPLITTING
# ============================================================

print("\n" + "=" * 70)
print("DATE-BASED BENIGN DATA SPLITTING")
print("=" * 70)

# Create a date column for splitting
benign_df["Date"] = (
    benign_df["Verified_Timestamp"].dt.strftime("%Y-%m-%d")
)

portscan_df["Date"] = (
    portscan_df["Verified_Timestamp"].dt.strftime("%Y-%m-%d")
)

# Training: July 3, 4 and 5
benign_train = benign_df[
    benign_df["Date"].isin(TRAIN_DATES)
].copy()

# Validation: July 6
benign_validation = benign_df[
    benign_df["Date"] == VALIDATION_DATE
].copy()

# Benign testing: July 7
benign_test = benign_df[
    benign_df["Date"] == BENIGN_TEST_DATE
].copy()

# Zero-day testing: all PortScan flows
portscan_test = portscan_df.copy()

print("Training dates:", TRAIN_DATES)
print("Validation date:", VALIDATION_DATE)
print("Benign testing date:", BENIGN_TEST_DATE)
print("Zero-day testing attack:", TARGET_ATTACK)


# ============================================================
# 7. VERIFY DATASET SPLITS
# ============================================================

print("\n" + "=" * 70)
print("DATASET SPLIT VERIFICATION")
print("=" * 70)

datasets = {
    "Benign Training": benign_train,
    "Benign Validation": benign_validation,
    "Benign Testing": benign_test,
    "PortScan Testing": portscan_test
}

for name, dataset in datasets.items():

    print("\n" + name)

    print("Rows:", len(dataset))

    print(
        "Start:",
        dataset["Verified_Timestamp"].min()
    )

    print(
        "End:",
        dataset["Verified_Timestamp"].max()
    )

    print(
        "Labels:",
        dataset["Label"].value_counts().to_dict()
    )

    if dataset.empty:
        raise ValueError(
            f"{name} dataset is empty."
        )

# Verify labels
assert benign_train["Label"].eq("BENIGN").all()

assert benign_validation["Label"].eq("BENIGN").all()

assert benign_test["Label"].eq("BENIGN").all()

assert portscan_test["Label"].eq(TARGET_ATTACK).all()

# Verify date boundaries
assert set(benign_train["Date"].unique()).issubset(
    set(TRAIN_DATES)
)

assert benign_validation["Date"].eq(
    VALIDATION_DATE
).all()

assert benign_test["Date"].eq(
    BENIGN_TEST_DATE
).all()

assert portscan_test["Date"].eq(
    BENIGN_TEST_DATE
).all()

# Verify chronological separation
assert (
    benign_train["Verified_Timestamp"].max()
    < benign_validation["Verified_Timestamp"].min()
)

assert (
    benign_validation["Verified_Timestamp"].max()
    < benign_test["Verified_Timestamp"].min()
)

# Verify no PortScan flows enter benign splits
assert not any(
    dataset["Label"].eq(TARGET_ATTACK).any()
    for dataset in [
        benign_train,
        benign_validation,
        benign_test
    ]
)

# Verify timestamp separation
train_timestamps = set(
    benign_train["Verified_Timestamp"].unique()
)

validation_timestamps = set(
    benign_validation["Verified_Timestamp"].unique()
)

test_timestamps = set(
    benign_test["Verified_Timestamp"].unique()
)

assert train_timestamps.isdisjoint(validation_timestamps)

assert train_timestamps.isdisjoint(test_timestamps)

assert validation_timestamps.isdisjoint(test_timestamps)

print("\nAll date-based split verification checks passed.")


# ============================================================
# 8. SAVE PREPARED DATASETS
# ============================================================

print("\n" + "=" * 70)
print("SAVING PREPARED DATASETS")
print("=" * 70)

benign_train.to_csv(
    OUTPUT_DIR / "benign_train.csv",
    index=False
)

benign_validation.to_csv(
    OUTPUT_DIR / "benign_validation.csv",
    index=False
)

benign_test.to_csv(
    OUTPUT_DIR / "benign_test.csv",
    index=False
)

portscan_test.to_csv(
    OUTPUT_DIR / "portscan_test.csv",
    index=False
)

print("Benign training dataset saved.")
print("Benign validation dataset saved.")
print("Benign testing dataset saved.")
print("PortScan testing dataset saved.")


# ============================================================
# 9. SAVE SPLIT METADATA
# ============================================================

metadata = {
    "dataset": "CIC-IDS2017",

    "target_zero_day_attack": TARGET_ATTACK,

    "split_method": "Date-based chronological splitting",

    "training_dates": TRAIN_DATES,

    "validation_date": VALIDATION_DATE,

    "benign_testing_date": BENIGN_TEST_DATE,

    "features": FEATURE_COLUMNS,

    "metadata_columns": METADATA_COLUMNS,

    "timestamp_interpretation": "DD/MM/YYYY",

    "benign_training_rows": len(benign_train),

    "benign_validation_rows": len(benign_validation),

    "benign_testing_rows": len(benign_test),

    "portscan_testing_rows": len(portscan_test),

    "training_start": str(
        benign_train["Verified_Timestamp"].min()
    ),

    "training_end": str(
        benign_train["Verified_Timestamp"].max()
    ),

    "validation_start": str(
        benign_validation["Verified_Timestamp"].min()
    ),

    "validation_end": str(
        benign_validation["Verified_Timestamp"].max()
    ),

    "benign_testing_start": str(
        benign_test["Verified_Timestamp"].min()
    ),

    "benign_testing_end": str(
        benign_test["Verified_Timestamp"].max()
    ),

    "portscan_start": str(
        portscan_test["Verified_Timestamp"].min()
    ),

    "portscan_end": str(
        portscan_test["Verified_Timestamp"].max()
    ),

    "portscan_used_for_training": False,

    "portscan_used_for_validation": False,

    "original_dataset_modified": False
}

metadata_path = OUTPUT_DIR / "split_metadata.json"

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )

print("\nSplit metadata saved.")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL ZERO-DAY DATA PREPARATION SUMMARY")
print("=" * 70)

summary = {
    "Benign Training": len(benign_train),
    "Benign Validation": len(benign_validation),
    "Benign Testing": len(benign_test),
    "PortScan Testing": len(portscan_test)
}

for name, count in summary.items():

    print(f"{name}: {count:,}")

print("\nOutput directory:", OUTPUT_DIR)

print("\nGenerated files:")

filenames = [
    "benign_train.csv",
    "benign_validation.csv",
    "benign_test.csv",
    "portscan_test.csv",
    "split_metadata.json"
]

for filename in filenames:

    file_path = OUTPUT_DIR / filename

    print(
        f"{filename}: "
        f"{file_path.stat().st_size / (1024 ** 2):.2f} MB"
    )

print("\nTraining: BENIGN only (July 3–5)")
print("Validation: BENIGN only (July 6)")
print("Benign testing: BENIGN only (July 7)")
print("Zero-day testing: PortScan only (July 7)")

print("\nOriginal dataset remains unchanged.")

print("\n" + "=" * 70)
print("ZERO-DAY DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 70)
