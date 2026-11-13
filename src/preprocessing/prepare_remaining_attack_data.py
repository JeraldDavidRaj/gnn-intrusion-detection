
from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data" / "processed" / "cic_ids2017_cleaned.csv"
OUTPUT = ROOT / "data" / "remaining_attacks"

OUTPUT.mkdir(parents=True, exist_ok=True)

ATTACKS = {
    "ddos": ["DDoS"],
    "bot": ["Bot"],
    "dos_hulk": ["DoS Hulk"],
    "dos_goldeneye": ["DoS GoldenEye"],
    "web_attack": [
        "Web Attack â€“ Brute Force",
        "Web Attack â€“ XSS",
        "Web Attack â€“ Sql Injection"
    ],
    "infiltration": ["Infiltration"]
}

print("=" * 75)
print("CORRECTED CHRONOLOGICAL DATA PREPARATION")
print("=" * 75)

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

df = pd.read_csv(SOURCE, low_memory=False)

time_column = (
    "Verified_Timestamp"
    if "Verified_Timestamp" in df.columns
    else "Timestamp"
)

df["_time"] = pd.to_datetime(
    df[time_column],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

invalid = int(df["_time"].isna().sum())

print("Total rows:", f"{len(df):,}")
print("Invalid timestamps:", f"{invalid:,}")

if invalid > 0:
    raise ValueError(
        "Invalid timestamps detected. Stop and inspect them."
    )

df["Label"] = df["Label"].astype(str).str.strip()
df["_date"] = df["_time"].dt.normalize()

available_dates = sorted(df["_date"].unique())

print("\nAvailable dates:")
for date in available_dates:
    print(date.date())

benign_df = df[df["Label"] == "BENIGN"].copy()

metadata = {
    "timestamp_column": time_column,
    "total_rows": int(len(df)),
    "invalid_timestamps": invalid,
    "experiments": {}
}

for attack_name, attack_labels in ATTACKS.items():

    print("\n" + "=" * 75)
    print("ATTACK:", attack_name.upper())
    print("=" * 75)

    attack_df = df[
        df["Label"].isin(attack_labels)
    ].copy()

    if attack_df.empty:
        raise ValueError(
            f"No rows found for {attack_name}"
        )

    attack_dates = sorted(
        attack_df["_date"].unique()
    )

    if len(attack_dates) != 1:
        raise ValueError(
            f"{attack_name} occurs on multiple dates: {attack_dates}"
        )

    target_date = attack_dates[0]

    previous_dates = [
        date for date in available_dates
        if date < target_date
    ]

    if len(previous_dates) < 2:
        raise ValueError(
            f"Not enough earlier dates for {attack_name}"
        )

    validation_date = previous_dates[-1]

    training_dates = previous_dates[:-1]

    # Training: benign traffic from earlier dates.
    train_df = benign_df[
        benign_df["_date"].isin(training_dates)
    ].copy()

    # Validation: all benign traffic from the immediately
    # preceding complete date.
    val_df = benign_df[
        benign_df["_date"] == validation_date
    ].copy()

    # Test: ALL benign traffic on the attack date.
    benign_test_df = benign_df[
        benign_df["_date"] == target_date
    ].copy()

    # Test attack: only the selected attack category.
    attack_test_df = attack_df.copy()

    train_df = train_df.sort_values(
        "_time"
    ).reset_index(drop=True)

    val_df = val_df.sort_values(
        "_time"
    ).reset_index(drop=True)

    benign_test_df = benign_test_df.sort_values(
        "_time"
    ).reset_index(drop=True)

    attack_test_df = attack_test_df.sort_values(
        "_time"
    ).reset_index(drop=True)

    # Chronological integrity checks.
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"{attack_name}: training or validation split is empty."
        )

    if train_df["_time"].max() >= val_df["_time"].min():
        raise ValueError(
            f"{attack_name}: training overlaps validation."
        )

    if val_df["_time"].max() >= target_date:
        raise ValueError(
            f"{attack_name}: validation overlaps attack date."
        )

    if benign_test_df.empty or attack_test_df.empty:
        raise ValueError(
            f"{attack_name}: test split is empty."
        )

    # Save the four datasets expected by the training engine.
    train_df.to_csv(
        OUTPUT / f"{attack_name}_benign_train_candidates.csv",
        index=False
    )

    val_df.to_csv(
        OUTPUT / f"{attack_name}_benign_validation.csv",
        index=False
    )

    benign_test_df.to_csv(
        OUTPUT / f"{attack_name}_benign_test.csv",
        index=False
    )

    attack_test_df.to_csv(
        OUTPUT / f"{attack_name}_test.csv",
        index=False
    )

    metadata["experiments"][attack_name] = {
        "attack_date": str(target_date.date()),
        "training_dates": [
            str(d.date()) for d in training_dates
        ],
        "validation_date": str(validation_date.date()),
        "training_benign_rows": int(len(train_df)),
        "validation_benign_rows": int(len(val_df)),
        "test_benign_rows": int(len(benign_test_df)),
        "test_attack_rows": int(len(attack_test_df))
    }

    print("Training dates:", [
        str(d.date()) for d in training_dates
    ])

    print("Validation date:", validation_date.date())
    print("Test date:", target_date.date())

    print("Benign training:", f"{len(train_df):,}")
    print("Benign validation:", f"{len(val_df):,}")
    print("Benign test:", f"{len(benign_test_df):,}")
    print("Attack test:", f"{len(attack_test_df):,}")

with open(
    OUTPUT / "preparation_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

print("\n" + "=" * 75)
print("CORRECTED DATA PREPARATION COMPLETED")
print("Output:", OUTPUT)
print("=" * 75)

