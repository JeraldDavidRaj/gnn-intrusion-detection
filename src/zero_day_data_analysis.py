import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cic_ids2017_cleaned.csv"

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label"
]


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 65)
print("ZERO-DAY DATASET ANALYSIS")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(
    DATA_PATH,
    usecols=REQUIRED_COLUMNS,
    low_memory=False
)

print("\nDataset shape:", df.shape)
print("Dataset columns:", df.columns.tolist())


# ============================================================
# 2. ATTACK CATEGORY DISTRIBUTION
# ============================================================

print("\nATTACK CATEGORY DISTRIBUTION")
print("-" * 50)

label_counts = df["Label"].value_counts()

print(label_counts.to_string())

label_counts.to_csv(
    REPORT_DIR / "zero_day_label_distribution.csv",
    header=["Count"]
)


# ============================================================
# 3. PORTSCAN ANALYSIS
# ============================================================

print("\nPORTSCAN ANALYSIS")
print("-" * 50)

portscan_df = df[df["Label"] == "PortScan"].copy()

print("PortScan flows:", len(portscan_df))

print(
    "Unique source IPs:",
    portscan_df["Source IP"].nunique()
)

print(
    "Unique destination IPs:",
    portscan_df["Destination IP"].nunique()
)

print("\nPORTSCAN SOURCE IP DISTRIBUTION")

print(
    portscan_df["Source IP"]
    .value_counts()
    .head(20)
    .to_string()
)

print("\nPORTSCAN DESTINATION IP DISTRIBUTION")

print(
    portscan_df["Destination IP"]
    .value_counts()
    .head(20)
    .to_string()
)

print("\nPORTSCAN SOURCE-DESTINATION PAIRS")

portscan_pairs = (
    portscan_df
    .groupby(
        ["Source IP", "Destination IP"],
        dropna=False
    )
    .size()
    .sort_values(ascending=False)
)

print(portscan_pairs.head(20).to_string())

portscan_pairs.to_csv(
    REPORT_DIR / "zero_day_portscan_host_pairs.csv",
    header=["Flow Count"]
)


# ============================================================
# 4. UNIQUE HOSTS PER ATTACK CATEGORY
# ============================================================

print("\nUNIQUE HOSTS PER ATTACK CATEGORY")
print("-" * 50)

host_statistics = []

for label, group in df.groupby("Label"):

    source_count = group["Source IP"].nunique()

    destination_count = group["Destination IP"].nunique()

    all_hosts = (
        set(group["Source IP"].dropna())
        |
        set(group["Destination IP"].dropna())
    )

    host_statistics.append({
        "Label": label,
        "Flow Count": len(group),
        "Unique Source IPs": source_count,
        "Unique Destination IPs": destination_count,
        "Unique Total Hosts": len(all_hosts)
    })

host_stats_df = pd.DataFrame(host_statistics)

print(host_stats_df.to_string(index=False))

host_stats_df.to_csv(
    REPORT_DIR / "zero_day_host_statistics.csv",
    index=False
)


# ============================================================
# 5. HOST OVERLAP ANALYSIS
# ============================================================

print("\nHOST OVERLAP ANALYSIS")
print("-" * 50)

benign_df = df[df["Label"] == "BENIGN"]

known_attack_df = df[
    ~df["Label"].isin(["BENIGN", "PortScan"])
]

benign_hosts = (
    set(benign_df["Source IP"].dropna())
    |
    set(benign_df["Destination IP"].dropna())
)

known_attack_hosts = (
    set(known_attack_df["Source IP"].dropna())
    |
    set(known_attack_df["Destination IP"].dropna())
)

portscan_hosts = (
    set(portscan_df["Source IP"].dropna())
    |
    set(portscan_df["Destination IP"].dropna())
)

benign_overlap = portscan_hosts & benign_hosts

known_overlap = portscan_hosts & known_attack_hosts

print("Benign hosts:", len(benign_hosts))

print("Known attack hosts:", len(known_attack_hosts))

print("PortScan hosts:", len(portscan_hosts))

print(
    "PortScan hosts also appearing in benign traffic:",
    len(benign_overlap)
)

print(
    "PortScan hosts also appearing in known attacks:",
    len(known_overlap)
)

print("\nPortScan hosts:", sorted(portscan_hosts))

print("\nPortScan hosts overlapping benign:", sorted(benign_overlap))

print("\nPortScan hosts overlapping known attacks:", sorted(known_overlap))


# ============================================================
# 6. TIMESTAMP FORMAT ANALYSIS
# ============================================================

print("\nTIMESTAMP FORMAT ANALYSIS")
print("-" * 50)

raw_timestamps = (
    df["Timestamp"]
    .astype("string")
    .str.strip()
)

print("\nRaw timestamp examples:")

print(
    raw_timestamps
    .dropna()
    .head(20)
    .to_string(index=False)
)

# CIC-IDS2017 dates are interpreted as day/month/year.
# Parse formats explicitly to prevent ambiguous dates.

timestamp_formats = [
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %I:%M %p",
    "%d/%m/%Y %I:%M:%S %p"
]

parsed_timestamps = pd.Series(
    pd.NaT,
    index=df.index,
    dtype="datetime64[ns]"
)

for timestamp_format in timestamp_formats:

    remaining = parsed_timestamps.isna()

    if not remaining.any():
        break

    newly_parsed = pd.to_datetime(
        raw_timestamps.loc[remaining],
        format=timestamp_format,
        errors="coerce"
    )

    parsed_timestamps.loc[remaining] = newly_parsed

df["Verified_Timestamp"] = parsed_timestamps

invalid_timestamp_count = (
    df["Verified_Timestamp"].isna().sum()
)

print("\nVERIFIED TIMESTAMP RESULTS")

print("Total timestamps:", len(df))

print("Valid timestamps:", len(df) - invalid_timestamp_count)

print("Invalid timestamps:", invalid_timestamp_count)

print(
    "Earliest timestamp:",
    df["Verified_Timestamp"].min()
)

print(
    "Latest timestamp:",
    df["Verified_Timestamp"].max()
)


# ============================================================
# 7. INVALID TIMESTAMP DIAGNOSTICS
# ============================================================

print("\nINVALID TIMESTAMP DIAGNOSTICS")
print("-" * 50)

invalid_mask = df["Verified_Timestamp"].isna()

if invalid_mask.any():

    invalid_examples = raw_timestamps.loc[invalid_mask]

    print("\nFirst 30 invalid timestamp values:")

    print(
        invalid_examples
        .head(30)
        .to_string(index=False)
    )

    print("\nMost frequent invalid timestamp values:")

    print(
        invalid_examples
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )

    invalid_report = df.loc[
        invalid_mask,
        [
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Timestamp",
            "Label"
        ]
    ]

    invalid_report.head(1000).to_csv(
        REPORT_DIR / "zero_day_invalid_timestamp_examples.csv",
        index=False
    )

else:

    print("All timestamps parsed successfully.")


# ============================================================
# 8. TIMESTAMP RANGE BY ATTACK CATEGORY
# ============================================================

print("\nTIMESTAMP RANGE BY ATTACK CATEGORY")
print("-" * 50)

timestamp_summary = (
    df.groupby("Label")["Verified_Timestamp"]
    .agg(["min", "max", "count"])
    .sort_values("count", ascending=False)
)

print(timestamp_summary.to_string())

timestamp_summary.to_csv(
    REPORT_DIR / "zero_day_timestamp_summary.csv"
)


# ============================================================
# 9. PORTSCAN TIMESTAMP ANALYSIS
# ============================================================

print("\nPORTSCAN TIMESTAMP ANALYSIS")
print("-" * 50)

portscan_timestamps = df.loc[
    df["Label"] == "PortScan",
    "Verified_Timestamp"
]

print("PortScan flows:", len(portscan_timestamps))

print(
    "Valid PortScan timestamps:",
    portscan_timestamps.notna().sum()
)

print(
    "Invalid PortScan timestamps:",
    portscan_timestamps.isna().sum()
)

print(
    "PortScan start:",
    portscan_timestamps.min()
)

print(
    "PortScan end:",
    portscan_timestamps.max()
)

print("\nMost frequent PortScan timestamp values:")

print(
    portscan_timestamps
    .value_counts()
    .head(20)
    .to_string()
)


# ============================================================
# 10. FLOW ID DUPLICATE ANALYSIS
# ============================================================

print("\nFLOW ID DUPLICATE ANALYSIS")
print("-" * 50)

total_rows = len(df)

unique_flow_ids = df["Flow ID"].nunique()

repeated_flow_id_rows = df["Flow ID"].duplicated().sum()

print("Total rows:", total_rows)

print("Unique Flow IDs:", unique_flow_ids)

print("Repeated Flow ID rows:", repeated_flow_id_rows)

print("\nMost frequent Flow IDs:")

print(
    df["Flow ID"]
    .value_counts()
    .head(10)
    .to_string()
)

print(
    "\nNote: Repeated Flow IDs do not necessarily indicate "
    "duplicate network-flow records."
)


# ============================================================
# 11. DUPLICATE ANALYSIS
# ============================================================

print("\nDUPLICATE ANALYSIS")
print("-" * 50)

# Duplicate rows based only on the five analysis columns.

selected_column_duplicates = (
    df[REQUIRED_COLUMNS]
    .duplicated()
    .sum()
)

print(
    "Duplicates across selected five columns:",
    selected_column_duplicates
)

print(
    "These are not necessarily duplicates across "
    "all original dataset columns."
)


# ============================================================
# 12. FULL-ROW DUPLICATE VERIFICATION
# ============================================================

print("\nFULL-ROW DUPLICATE VERIFICATION")
print("-" * 50)

print("Checking full-row duplicates in chunks...")

chunk_size = 100000

seen_hashes = set()

full_row_duplicate_count = 0

processed_rows = 0

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=chunk_size,
    low_memory=False
):

    # Hash every column of each original cleaned record.
    # This is an efficient screening method; hash collisions
    # are theoretically possible.

    row_hashes = pd.util.hash_pandas_object(
        chunk,
        index=False
    ).to_numpy()

    for row_hash in row_hashes:

        hash_value = int(row_hash)

        if hash_value in seen_hashes:

            full_row_duplicate_count += 1

        else:

            seen_hashes.add(hash_value)

    processed_rows += len(chunk)

    print(
        f"Processed {processed_rows:,} / {total_rows:,} rows",
        end="\r"
    )

print()

print(
    "Repeated full-row hashes:",
    full_row_duplicate_count
)

print(
    "Note: Hash-based duplicate screening does not "
    "guarantee collision-free exact-row verification."
)


# ============================================================
# 13. MISSING VALUE ANALYSIS
# ============================================================

print("\nMISSING VALUE ANALYSIS")
print("-" * 50)

missing_values = df[REQUIRED_COLUMNS].isna().sum()

print(missing_values.to_string())

missing_values.to_csv(
    REPORT_DIR / "zero_day_missing_values.csv",
    header=["Missing Count"]
)


# ============================================================
# 14. ZERO-DAY EXPERIMENT FEASIBILITY
# ============================================================

print("\nZERO-DAY EXPERIMENT FEASIBILITY")
print("-" * 50)

benign_count = (df["Label"] == "BENIGN").sum()

known_attack_count = (
    ~df["Label"].isin(["BENIGN", "PortScan"])
).sum()

portscan_count = (df["Label"] == "PortScan").sum()

print("Benign flows:", benign_count)

print("Known attack flows:", known_attack_count)

print("Held-out PortScan flows:", portscan_count)

print("\nPROPOSED ZERO-DAY EXPERIMENT")

print(
    "Training: BENIGN flows only"
)

print(
    "Validation: Separate BENIGN flows for threshold calibration"
)

print(
    "Testing: Untouched BENIGN test flows + held-out PortScan flows"
)

print(
    "PortScan must not be included in training, "
    "threshold calibration, or model selection."
)

print(
    "\nImportant: This experiment evaluates detection of "
    "an unseen attack category, not unseen hosts."
)


# ============================================================
# 15. SAVE SUMMARY REPORT
# ============================================================

summary = {
    "Total Flows": total_rows,
    "Benign Flows": int(benign_count),
    "Known Attack Flows": int(known_attack_count),
    "PortScan Flows": int(portscan_count),
    "Unique Flow IDs": int(unique_flow_ids),
    "Repeated Flow ID Rows": int(repeated_flow_id_rows),
    "Selected Column Duplicate Rows": int(
        selected_column_duplicates
    ),
    "Repeated Full Row Hashes": int(
        full_row_duplicate_count
    ),
    "Valid Timestamps": int(
        len(df) - invalid_timestamp_count
    ),
    "Invalid Timestamps": int(
        invalid_timestamp_count
    ),
    "Benign Hosts": len(benign_hosts),
    "Known Attack Hosts": len(known_attack_hosts),
    "PortScan Hosts": len(portscan_hosts),
    "PortScan Benign Host Overlap": len(benign_overlap),
    "PortScan Known Attack Host Overlap": len(known_overlap)
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    REPORT_DIR / "zero_day_dataset_summary.csv",
    index=False
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL ZERO-DAY DATASET SUMMARY")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nReports saved to:", REPORT_DIR)

print("\nZERO-DAY DATA ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 65)