
"""
DoS Hulk component-level and temporal drift analysis.

Reads:
    experiments/dos_hulk/data/prepared/
        dos_hulk_benign_validation.csv
        dos_hulk_benign_test.csv
        dos_hulk_test.csv

    experiments/dos_hulk/reports/
        dos_hulk_flow_predictions.csv
        dos_hulk_calibration.json

Writes diagnostic outputs only to:
    experiments/dos_hulk/reports/diagnostics/

Does not retrain or modify any model.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "dos_hulk"

DATA = EXP / "data" / "prepared"
REPORTS = EXP / "reports"
OUTPUT = REPORTS / "diagnostics"

OUTPUT.mkdir(parents=True, exist_ok=True)

VALIDATION_FILE = DATA / "dos_hulk_benign_validation.csv"
BENIGN_TEST_FILE = DATA / "dos_hulk_benign_test.csv"
ATTACK_TEST_FILE = DATA / "dos_hulk_test.csv"

PREDICTIONS_FILE = REPORTS / "dos_hulk_flow_predictions.csv"
CALIBRATION_FILE = REPORTS / "dos_hulk_calibration.json"

FOCUS_SOURCE = "192.168.10.5"

FEATURES = [
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

COMPONENTS = {
    "GNN": "GNN_Rank",
    "Isolation_Forest": "Temporal_IForest_Rank",
    "Behaviour": "Behaviour_Rank",
}

print("=" * 75)
print("DOS HULK COMPONENT AND DRIFT ANALYSIS")
print("=" * 75)


# =========================================================
# 2. CHECK INPUT FILES
# =========================================================

required_files = [
    VALIDATION_FILE,
    BENIGN_TEST_FILE,
    ATTACK_TEST_FILE,
    PREDICTIONS_FILE,
    CALIBRATION_FILE,
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(f"Missing input file: {file}")

print("All required input files found.")


# =========================================================
# 3. LOAD DATA
# =========================================================

def load_csv(path):
    print("Loading:", path.name)
    return pd.read_csv(path, low_memory=False)


validation = load_csv(VALIDATION_FILE)
benign = load_csv(BENIGN_TEST_FILE)
attack = load_csv(ATTACK_TEST_FILE)
predictions = load_csv(PREDICTIONS_FILE)

with open(CALIBRATION_FILE, encoding="utf-8") as f:
    calibration = json.load(f)

threshold = float(calibration["threshold"])

print("\nLoaded rows:")
print("Validation benign:", len(validation))
print("Test benign:", len(benign))
print("Test DoS Hulk:", len(attack))
print("Prediction rows:", len(predictions))


# =========================================================
# 4. VERIFY PREDICTION ALIGNMENT
# =========================================================

expected_rows = len(benign) + len(attack)

if len(predictions) != expected_rows:
    raise ValueError(
        f"Prediction count mismatch. Expected {expected_rows}, "
        f"received {len(predictions)}."
    )

expected_labels = (
    ["BENIGN"] * len(benign)
    + ["DOS_HULK"] * len(attack)
)

if predictions["Actual_Label"].astype(str).tolist() != expected_labels:
    raise ValueError(
        "Prediction row order or labels do not match the prepared CSV files."
    )

if predictions[
    list(COMPONENTS.values())
    + ["Combined_Anomaly_Score"]
].isna().any().any():
    raise ValueError("Prediction file contains missing component scores.")

# The training script saved benign predictions first,
# followed by attack predictions, preserving original CSV row order.
benign_pred = predictions.iloc[:len(benign)].copy().reset_index(drop=True)
attack_pred = predictions.iloc[len(benign):].copy().reset_index(drop=True)

benign = benign.reset_index(drop=True)
attack = attack.reset_index(drop=True)

print("Prediction-to-CSV row alignment verified.")


# =========================================================
# 5. PREPARE TIMESTAMPS AND DATA
# =========================================================

def prepare_frame(df):
    df = df.copy()

    df["_time"] = pd.to_datetime(
        df["_time"] if "_time" in df.columns else df["Timestamp"],
        errors="coerce",
        dayfirst=True,
    )

    df = df.dropna(
        subset=["_time", "Source IP", "Destination IP"]
    ).copy()

    df["hour"] = df["_time"].dt.hour
    df["date"] = df["_time"].dt.strftime("%Y-%m-%d")

    df["Source IP"] = df["Source IP"].astype(str).str.strip()
    df["Destination IP"] = (
        df["Destination IP"].astype(str).str.strip()
    )

    df["Destination Port"] = pd.to_numeric(
        df["Destination Port"],
        errors="coerce",
    )

    df["Protocol"] = pd.to_numeric(
        df["Protocol"],
        errors="coerce",
    )

    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df.reset_index(drop=True)


validation = prepare_frame(validation)
benign = prepare_frame(benign)
attack = prepare_frame(attack)

if len(benign) != len(benign_pred):
    raise ValueError(
        "Rows were removed while parsing benign test data. "
        "Check timestamps and IP addresses."
    )

if len(attack) != len(attack_pred):
    raise ValueError(
        "Rows were removed while parsing attack test data. "
        "Check timestamps and IP addresses."
    )

benign = pd.concat(
    [benign, benign_pred],
    axis=1,
)

attack = pd.concat(
    [attack, attack_pred],
    axis=1,
)

benign["actual"] = 0
attack["actual"] = 1

benign["predicted"] = (
    benign["Predicted_Label"] == "ANOMALY"
).astype(int)

attack["predicted"] = (
    attack["Predicted_Label"] == "ANOMALY"
).astype(int)

test = pd.concat(
    [benign, attack],
    ignore_index=True,
)

# Separate benign validation is retained for comparison.
validation["actual"] = 0


# =========================================================
# 6. BASIC EVALUATION
# =========================================================

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "FPR": float(fp / max(tn + fp, 1)),
        "FNR": float(fn / max(fn + tp, 1)),
    }


overall_metrics = calculate_metrics(
    test["actual"],
    test["predicted"],
)

print("\nOverall model metrics:")
print(json.dumps(overall_metrics, indent=4))


# =========================================================
# 7. COMPONENT-LEVEL ABLATION
# =========================================================

# The component ranks are percentile ranks against benign validation.
# Therefore, 0.99 is the corresponding 99th-percentile component cutoff.
# This evaluates components independently without fitting on test labels.

component_results = []

for name, column in COMPONENTS.items():

    score = test[column].to_numpy(dtype=float)

    component_pred = (
        score > 0.99
    ).astype(int)

    result = calculate_metrics(
        test["actual"],
        component_pred,
    )

    result["component"] = name
    result["rank_threshold"] = 0.99

    component_results.append(result)

component_results = pd.DataFrame(component_results)

component_results.to_csv(
    OUTPUT / "component_ablation.csv",
    index=False,
)

print("\nComponent ablation:")
print(component_results.to_string(index=False))


# =========================================================
# 8. COMPONENT CONTRIBUTION FOR EACH PREDICTION GROUP
# =========================================================

def component_group_summary(df, group_name):

    rows = []

    groups = {
        "True Negative": df[
            (df["actual"] == 0) & (df["predicted"] == 0)
        ],
        "False Positive": df[
            (df["actual"] == 0) & (df["predicted"] == 1)
        ],
        "True Positive": df[
            (df["actual"] == 1) & (df["predicted"] == 1)
        ],
        "False Negative": df[
            (df["actual"] == 1) & (df["predicted"] == 0)
        ],
    }

    for outcome, subset in groups.items():

        row = {
            "dataset_group": group_name,
            "outcome": outcome,
            "flows": len(subset),
        }

        for name, column in COMPONENTS.items():
            row[name + "_mean_rank"] = (
                float(subset[column].mean())
                if len(subset)
                else None
            )

        row["mean_combined_score"] = (
            float(subset["Combined_Anomaly_Score"].mean())
            if len(subset)
            else None
        )

        rows.append(row)

    return rows


group_rows = component_group_summary(
    test,
    "combined_test",
)

group_summary = pd.DataFrame(group_rows)

group_summary.to_csv(
    OUTPUT / "component_group_summary.csv",
    index=False,
)

print("\nComponent scores by prediction outcome:")
print(group_summary.to_string(index=False))


# =========================================================
# 9. FALSE POSITIVES BY HOUR
# =========================================================

hourly = (
    benign.groupby("hour")
    .agg(
        total_flows=("predicted", "size"),
        false_positives=("predicted", "sum"),
        average_gnn_rank=("GNN_Rank", "mean"),
        average_iforest_rank=("Temporal_IForest_Rank", "mean"),
        average_behaviour_rank=("Behaviour_Rank", "mean"),
    )
    .reset_index()
)

hourly["FPR"] = (
    hourly["false_positives"]
    / hourly["total_flows"].clip(lower=1)
)

hourly["FPR_percent"] = hourly["FPR"] * 100

hourly.to_csv(
    OUTPUT / "false_positives_by_hour.csv",
    index=False,
)

print("\nFalse positives by hour:")
print(
    hourly.sort_values(
        "FPR",
        ascending=False,
    ).to_string(index=False)
)


# =========================================================
# 10. SOURCE IP ANALYSIS
# =========================================================

source_summary = (
    benign.groupby("Source IP")
    .agg(
        total_flows=("predicted", "size"),
        false_positives=("predicted", "sum"),
        mean_combined_score=("Combined_Anomaly_Score", "mean"),
        mean_gnn_rank=("GNN_Rank", "mean"),
        mean_iforest_rank=("Temporal_IForest_Rank", "mean"),
        mean_behaviour_rank=("Behaviour_Rank", "mean"),
    )
    .reset_index()
)

source_summary["FPR"] = (
    source_summary["false_positives"]
    / source_summary["total_flows"].clip(lower=1)
)

source_summary = source_summary.sort_values(
    "false_positives",
    ascending=False,
)

source_summary.to_csv(
    OUTPUT / "source_ip_false_positive_summary.csv",
    index=False,
)

print("\nTop 20 sources by false-positive count:")
print(source_summary.head(20).to_string(index=False))


# =========================================================
# 11. DESTINATION PORT ANALYSIS
# =========================================================

port_summary = (
    benign.groupby("Destination Port", dropna=False)
    .agg(
        total_flows=("predicted", "size"),
        false_positives=("predicted", "sum"),
        mean_combined_score=("Combined_Anomaly_Score", "mean"),
        mean_gnn_rank=("GNN_Rank", "mean"),
        mean_iforest_rank=("Temporal_IForest_Rank", "mean"),
        mean_behaviour_rank=("Behaviour_Rank", "mean"),
    )
    .reset_index()
)

port_summary["FPR"] = (
    port_summary["false_positives"]
    / port_summary["total_flows"].clip(lower=1)
)

port_summary = port_summary.sort_values(
    "false_positives",
    ascending=False,
)

port_summary.to_csv(
    OUTPUT / "destination_port_false_positive_summary.csv",
    index=False,
)

print("\nTop destination ports by false-positive count:")
print(port_summary.head(20).to_string(index=False))


# =========================================================
# 12. HOURLY SOURCE-PORT COMBINATIONS
# =========================================================

source_hour = (
    benign.groupby(
        ["hour", "Source IP"],
        dropna=False,
    )
    .agg(
        total_flows=("predicted", "size"),
        false_positives=("predicted", "sum"),
        mean_combined_score=("Combined_Anomaly_Score", "mean"),
        mean_gnn_rank=("GNN_Rank", "mean"),
        mean_iforest_rank=("Temporal_IForest_Rank", "mean"),
        mean_behaviour_rank=("Behaviour_Rank", "mean"),
    )
    .reset_index()
)

source_hour["FPR"] = (
    source_hour["false_positives"]
    / source_hour["total_flows"].clip(lower=1)
)

source_hour = source_hour.sort_values(
    "false_positives",
    ascending=False,
)

source_hour.to_csv(
    OUTPUT / "source_ip_hourly_false_positives.csv",
    index=False,
)


# =========================================================
# 13. FOCUS SOURCE ANALYSIS
# =========================================================

focus_validation = validation[
    validation["Source IP"] == FOCUS_SOURCE
].copy()

focus_test = benign[
    benign["Source IP"] == FOCUS_SOURCE
].copy()

def focus_summary(df, date_name):

    if len(df) == 0:
        return {
            "date": date_name,
            "flows": 0,
        }

    return {
        "date": date_name,
        "flows": int(len(df)),
        "false_positives": int(df["predicted"].sum())
        if "predicted" in df.columns else None,
        "false_positive_rate": float(df["predicted"].mean())
        if "predicted" in df.columns else None,
        "unique_destinations": int(
            df["Destination IP"].nunique()
        ),
        "unique_destination_ports": int(
            df["Destination Port"].nunique()
        ),
        "mean_flow_bytes_per_second": float(
            df["Flow Bytes/s"].mean()
        ),
        "median_flow_bytes_per_second": float(
            df["Flow Bytes/s"].median()
        ),
        "median_flow_duration": float(
            df["Flow Duration"].median()
        ),
        "median_forward_packets": float(
            df["Total Fwd Packets"].median()
        ),
        "median_backward_packets": float(
            df["Total Backward Packets"].median()
        ),
        "mean_gnn_rank": float(df["GNN_Rank"].mean())
        if "GNN_Rank" in df.columns else None,
        "mean_iforest_rank": float(
            df["Temporal_IForest_Rank"].mean()
        ) if "Temporal_IForest_Rank" in df.columns else None,
        "mean_behaviour_rank": float(
            df["Behaviour_Rank"].mean()
        ) if "Behaviour_Rank" in df.columns else None,
    }


focus_results = [
    focus_summary(focus_validation, "2017-07-04"),
    focus_summary(focus_test, "2017-07-05"),
]

with open(
    OUTPUT / "focus_source_comparison.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(focus_results, f, indent=4)

print("\nFocus source comparison:")
print(json.dumps(focus_results, indent=4))


# Focus source by hour on July 5.
focus_hour = (
    focus_test.groupby("hour")
    .agg(
        total_flows=("predicted", "size"),
        false_positives=("predicted", "sum"),
        mean_gnn_rank=("GNN_Rank", "mean"),
        mean_iforest_rank=("Temporal_IForest_Rank", "mean"),
        mean_behaviour_rank=("Behaviour_Rank", "mean"),
    )
    .reset_index()
)

focus_hour["FPR"] = (
    focus_hour["false_positives"]
    / focus_hour["total_flows"].clip(lower=1)
)

focus_hour.to_csv(
    OUTPUT / "focus_source_by_hour.csv",
    index=False,
)


# Focus source destination-port analysis.
focus_ports = (
    focus_test.groupby("Destination Port", dropna=False)
    .agg(
        total_flows=("predicted", "size"),
        false_positives=("predicted", "sum"),
        mean_combined_score=("Combined_Anomaly_Score", "mean"),
        mean_gnn_rank=("GNN_Rank", "mean"),
        mean_iforest_rank=("Temporal_IForest_Rank", "mean"),
        mean_behaviour_rank=("Behaviour_Rank", "mean"),
    )
    .reset_index()
)

focus_ports["FPR"] = (
    focus_ports["false_positives"]
    / focus_ports["total_flows"].clip(lower=1)
)

focus_ports = focus_ports.sort_values(
    "false_positives",
    ascending=False,
)

focus_ports.to_csv(
    OUTPUT / "focus_source_destination_ports.csv",
    index=False,
)


# =========================================================
# 14. FEATURE DRIFT: PSI AND KS TEST
# =========================================================

def calculate_psi(reference, current, bins=10):

    ref = pd.to_numeric(
        pd.Series(reference),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    cur = pd.to_numeric(
        pd.Series(current),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(ref) == 0 or len(cur) == 0:
        return None

    quantiles = np.linspace(0, 1, bins + 1)

    edges = np.unique(
        np.quantile(ref, quantiles)
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(
        ref,
        bins=edges,
    )[0]

    cur_counts = np.histogram(
        cur,
        bins=edges,
    )[0]

    ref_pct = ref_counts / len(ref)
    cur_pct = cur_counts / len(cur)

    epsilon = 1e-6

    ref_pct = np.clip(
        ref_pct,
        epsilon,
        None,
    )

    cur_pct = np.clip(
        cur_pct,
        epsilon,
        None,
    )

    return float(
        np.sum(
            (cur_pct - ref_pct)
            * np.log(cur_pct / ref_pct)
        )
    )


def drift_category(psi):

    if psi is None:
        return "Unavailable"

    if psi < 0.10:
        return "Low"

    if psi < 0.25:
        return "Moderate"

    return "High"


def feature_drift(reference, current, group_name):

    rows = []

    for col in FEATURES:

        ref = pd.to_numeric(
            reference[col],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        cur = pd.to_numeric(
            current[col],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(ref) == 0 or len(cur) == 0:
            continue

        psi = calculate_psi(ref, cur)

        ks = ks_2samp(
            ref,
            cur,
            alternative="two-sided",
        )

        rows.append({
            "comparison": group_name,
            "feature": col,
            "reference_count": len(ref),
            "current_count": len(cur),
            "reference_mean": float(ref.mean()),
            "current_mean": float(cur.mean()),
            "reference_median": float(ref.median()),
            "current_median": float(cur.median()),
            "reference_std": float(ref.std()),
            "current_std": float(cur.std()),
            "PSI": psi,
            "PSI_interpretation": drift_category(psi),
            "KS_statistic": float(ks.statistic),
            "KS_p_value": float(ks.pvalue),
        })

    return rows


drift_rows = []

# Full-day comparison.
drift_rows.extend(
    feature_drift(
        validation,
        benign,
        "July 4 vs July 5 - full day",
    )
)

# Compare the same hour across days.
for hour in sorted(
    set(validation["hour"].unique())
    & set(benign["hour"].unique())
):

    ref_hour = validation[
        validation["hour"] == hour
    ]

    cur_hour = benign[
        benign["hour"] == hour
    ]

    if len(ref_hour) < 100 or len(cur_hour) < 100:
        continue

    drift_rows.extend(
        feature_drift(
            ref_hour,
            cur_hour,
            f"July 4 vs July 5 - hour {hour:02d}",
        )
    )


drift_df = pd.DataFrame(drift_rows)

drift_df.to_csv(
    OUTPUT / "feature_drift_psi_ks.csv",
    index=False,
)

print("\nTop full-day feature drift:")

full_day_drift = drift_df[
    drift_df["comparison"] == "July 4 vs July 5 - full day"
].sort_values(
    "PSI",
    ascending=False,
)

print(full_day_drift.head(15).to_string(index=False))


# =========================================================
# 15. CATEGORICAL TRAFFIC DRIFT
# =========================================================

def categorical_drift(reference, current, column):

    ref = (
        reference[column]
        .fillna("MISSING")
        .astype(str)
        .value_counts(normalize=True)
    )

    cur = (
        current[column]
        .fillna("MISSING")
        .astype(str)
        .value_counts(normalize=True)
    )

    result = pd.concat(
        [
            ref.rename("July_4_share"),
            cur.rename("July_5_share"),
        ],
        axis=1,
    ).fillna(0)

    result["absolute_share_change"] = (
        result["July_5_share"] - result["July_4_share"]
    ).abs()

    result["category"] = column

    return result.reset_index(names="value")


categorical_results = []

for col in [
    "Destination Port",
    "Source Port",
    "Protocol",
]:

    result = categorical_drift(
        validation,
        benign,
        col,
    )

    categorical_results.append(result)


categorical_df = pd.concat(
    categorical_results,
    ignore_index=True,
)

categorical_df.to_csv(
    OUTPUT / "categorical_traffic_drift.csv",
    index=False,
)


# =========================================================
# 16. FALSE POSITIVE FEATURE PROFILE
# =========================================================

false_positive_flows = benign[
    benign["predicted"] == 1
]

true_negative_flows = benign[
    benign["predicted"] == 0
]

fp_profile = []

for col in FEATURES:

    fp = false_positive_flows[col].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    tn = true_negative_flows[col].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    fp_profile.append({
        "feature": col,
        "false_positive_median": (
            float(fp.median()) if len(fp) else None
        ),
        "true_negative_median": (
            float(tn.median()) if len(tn) else None
        ),
        "false_positive_mean": (
            float(fp.mean()) if len(fp) else None
        ),
        "true_negative_mean": (
            float(tn.mean()) if len(tn) else None
        ),
        "false_positive_count": len(fp),
        "true_negative_count": len(tn),
    })

pd.DataFrame(fp_profile).to_csv(
    OUTPUT / "false_positive_feature_profile.csv",
    index=False,
)


# =========================================================
# 17. SAVE EXECUTIVE SUMMARY
# =========================================================

summary = {
    "experiment": "DoS Hulk component and drift analysis",
    "focus_source": FOCUS_SOURCE,
    "overall_metrics": overall_metrics,
    "total_benign_test_flows": int(len(benign)),
    "total_benign_false_positives": int(
        benign["predicted"].sum()
    ),
    "total_attack_test_flows": int(len(attack)),
    "total_detected_attack_flows": int(
        attack["predicted"].sum()
    ),
    "false_positive_hour_with_highest_rate": (
        int(hourly.loc[hourly["FPR"].idxmax(), "hour"])
        if len(hourly) else None
    ),
    "top_false_positive_source": (
        str(source_summary.iloc[0]["Source IP"])
        if len(source_summary) else None
    ),
    "top_false_positive_destination_port": (
        float(port_summary.iloc[0]["Destination Port"])
        if len(port_summary) else None
    ),
    "focus_source_validation_flows": int(
        len(focus_validation)
    ),
    "focus_source_test_flows": int(
        len(focus_test)
    ),
    "component_ablation_file": "component_ablation.csv",
    "hourly_analysis_file": "false_positives_by_hour.csv",
    "source_analysis_file": "source_ip_false_positive_summary.csv",
    "feature_drift_file": "feature_drift_psi_ks.csv",
    "categorical_drift_file": "categorical_traffic_drift.csv",
}

with open(
    OUTPUT / "diagnostic_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )


# =========================================================
# 18. FINISH
# =========================================================

print("\n" + "=" * 75)
print("DIAGNOSTIC ANALYSIS COMPLETED")
print("=" * 75)

print("Overall benign FPR:", round(
    overall_metrics["FPR"] * 100, 4
), "%")

print("Overall DoS Hulk recall:", round(
    overall_metrics["recall"] * 100, 4
), "%")

print("\nDiagnostic output directory:")
print(OUTPUT)

print("\nGenerated files:")

for path in sorted(OUTPUT.glob("*")):
    print(path.name)

print("\nNo model files were changed.")
print("No model retraining was performed.")
