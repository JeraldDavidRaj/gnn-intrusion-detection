
import pandas as pd
from pathlib import Path

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

OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("ZERO-DAY TEMPORAL ANALYSIS")
print("=" * 70)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=["Timestamp", "Label"],
    low_memory=False
)

print("Total records:", len(df))

# ============================================================
# 3. PARSE TIMESTAMPS
# ============================================================

print("\nParsing timestamps...")

raw = df["Timestamp"].astype("string").str.strip()

parsed_minutes = pd.to_datetime(
    raw,
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

parsed_seconds = pd.to_datetime(
    raw,
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

parsed_fractional = pd.to_datetime(
    raw,
    format="%d/%m/%Y %H:%M:%S.%f",
    errors="coerce"
)

df["Timestamp"] = (
    parsed_minutes
    .fillna(parsed_seconds)
    .fillna(parsed_fractional)
)

invalid = int(df["Timestamp"].isna().sum())

print("Invalid timestamps:", invalid)

if invalid > 0:
    raise ValueError("Invalid timestamps found.")

df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")
df["Hour"] = df["Timestamp"].dt.hour

# ============================================================
# 4. DAILY FLOW DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DAILY FLOW DISTRIBUTION")
print("=" * 70)

daily = (
    df.groupby(["Date", "Label"])
    .size()
    .unstack(fill_value=0)
)

print(daily.to_string())

daily.to_csv(
    OUTPUT_DIR / "zero_day_daily_distribution.csv"
)

# ============================================================
# 5. DAILY BENIGN AND ATTACK DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DAILY BENIGN AND ATTACK DISTRIBUTION")
print("=" * 70)

df["Traffic_Type"] = df["Label"].apply(
    lambda x: "BENIGN" if x == "BENIGN" else "ATTACK"
)

daily_traffic = (
    df.groupby(["Date", "Traffic_Type"])
    .size()
    .unstack(fill_value=0)
)

print(daily_traffic.to_string())

daily_traffic.to_csv(
    OUTPUT_DIR / "zero_day_daily_traffic.csv"
)

# ============================================================
# 6. DAILY PORTSCAN DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DAILY PORTSCAN DISTRIBUTION")
print("=" * 70)

portscan = df[df["Label"] == "PortScan"]

portscan_daily = (
    portscan.groupby("Date")
    .size()
    .rename("PortScan Flows")
)

print(portscan_daily.to_string())

portscan_daily.to_csv(
    OUTPUT_DIR / "zero_day_portscan_daily.csv"
)

# ============================================================
# 7. HOURLY BENIGN TRAFFIC DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("HOURLY BENIGN TRAFFIC DISTRIBUTION")
print("=" * 70)

benign = df[df["Label"] == "BENIGN"]

benign_hourly = (
    benign.groupby(["Date", "Hour"])
    .size()
    .rename("Benign Flows")
    .reset_index()
)

print(benign_hourly.to_string(index=False))

benign_hourly.to_csv(
    OUTPUT_DIR / "zero_day_benign_hourly.csv",
    index=False
)

# ============================================================
# 8. PORTSCAN HOURLY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PORTSCAN HOURLY DISTRIBUTION")
print("=" * 70)

portscan_hourly = (
    portscan.groupby(["Date", "Hour"])
    .size()
    .rename("PortScan Flows")
    .reset_index()
)

print(portscan_hourly.to_string(index=False))

portscan_hourly.to_csv(
    OUTPUT_DIR / "zero_day_portscan_hourly.csv",
    index=False
)

# ============================================================
# 9. BENIGN TRAFFIC AROUND PORTSCAN PERIOD
# ============================================================

print("\n" + "=" * 70)
print("BENIGN TRAFFIC AROUND PORTSCAN PERIOD")
print("=" * 70)

portscan_start = portscan["Timestamp"].min()
portscan_end = portscan["Timestamp"].max()

print("PortScan start:", portscan_start)
print("PortScan end:", portscan_end)

# Examine benign traffic on the same day as PortScan.
portscan_date = portscan_start.date()

same_day_benign = benign[
    benign["Timestamp"].dt.date == portscan_date
].copy()

print(
    "Benign flows on PortScan date:",
    len(same_day_benign)
)

print(
    "Benign flows before PortScan:",
    int((same_day_benign["Timestamp"] < portscan_start).sum())
)

print(
    "Benign flows during PortScan time window:",
    int(
        (
            (same_day_benign["Timestamp"] >= portscan_start)
            &
            (same_day_benign["Timestamp"] <= portscan_end)
        ).sum()
    )
)

print(
    "Benign flows after PortScan:",
    int((same_day_benign["Timestamp"] > portscan_end).sum())
)

# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL ANALYSIS SUMMARY")
print("=" * 70)

print("Dataset start:", df["Timestamp"].min())
print("Dataset end:", df["Timestamp"].max())

print("\nDaily total flows:")
print(df.groupby("Date").size().to_string())

print("\nDaily benign flows:")
print(benign.groupby("Date").size().to_string())

print("\nDaily PortScan flows:")
print(portscan.groupby("Date").size().to_string())

print("\nReports saved in:", OUTPUT_DIR)

print("\nGenerated reports:")
print("- zero_day_daily_distribution.csv")
print("- zero_day_daily_traffic.csv")
print("- zero_day_portscan_daily.csv")
print("- zero_day_benign_hourly.csv")
print("- zero_day_portscan_hourly.csv")

print("\n" + "=" * 70)
print("TEMPORAL ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)
