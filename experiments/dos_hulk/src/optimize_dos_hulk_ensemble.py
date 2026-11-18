
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

EXPERIMENT = ROOT / "experiments" / "dos_hulk"
REPORTS = EXPERIMENT / "reports"
DIAGNOSTICS = REPORTS / "diagnostics"

PREDICTIONS_FILE = REPORTS / "dos_hulk_flow_predictions.csv"
VALIDATION_FILE = EXPERIMENT / "data" / "prepared" / "dos_hulk_benign_validation.csv"
BENIGN_TEST_FILE = EXPERIMENT / "data" / "prepared" / "dos_hulk_benign_test.csv"
ATTACK_TEST_FILE = EXPERIMENT / "data" / "prepared" / "dos_hulk_test.csv"

OUTPUT_DIR = REPORTS / "ensemble_optimization"

for path in [
    PREDICTIONS_FILE,
    VALIDATION_FILE,
    BENIGN_TEST_FILE,
    ATTACK_TEST_FILE
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD EXISTING RESULTS
# ============================================================

print("=" * 75)
print("DOS HULK ENSEMBLE COMPARISON")
print("=" * 75)

pred = pd.read_csv(PREDICTIONS_FILE)

validation = pd.read_csv(VALIDATION_FILE)
benign_test = pd.read_csv(BENIGN_TEST_FILE)
attack_test = pd.read_csv(ATTACK_TEST_FILE)

print("Validation benign:", len(validation))
print("Test benign:", len(benign_test))
print("Test DoS Hulk:", len(attack_test))
print("Prediction rows:", len(pred))


# ============================================================
# 3. VALIDATE ROW ALIGNMENT
# ============================================================

expected_rows = len(benign_test) + len(attack_test)

if len(pred) != expected_rows:
    raise ValueError(
        f"Prediction rows ({len(pred)}) do not match "
        f"test rows ({expected_rows})."
    )

if len(validation) == 0:
    raise ValueError("Validation dataset is empty.")


# ============================================================
# 4. FIND COMPONENT RANK COLUMNS
# ============================================================

def find_column(dataframe, candidates, component_name):
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise KeyError(
        f"Could not find {component_name} rank column.\n"
        f"Available columns: {list(dataframe.columns)}\n"
        f"Edit the candidate names in this script to match your CSV."
    )


gnn_col = find_column(
    pred,
    ["GNN_Rank", "GNN_Percentile_Rank", "GNN_Percentile"],
    "GNN"
)

if_col = find_column(
    pred,
    [
        "Temporal_IForest_Rank",
        "Isolation_Forest_Rank",
        "IForest_Rank",
        "IF_Rank"
    ],
    "Isolation Forest"
)

behaviour_col = find_column(
    pred,
    [
        "Behaviour_Rank",
        "Behavior_Rank",
        "Behaviour_Percentile_Rank",
        "Behaviour_Percentile"
    ],
    "Behaviour"
)

print("\nComponent columns:")
print("GNN:", gnn_col)
print("Isolation Forest:", if_col)
print("Behaviour:", behaviour_col)


# ============================================================
# 5. EXTRACT COMPONENT RANKS
# ============================================================

gnn = pred[gnn_col].to_numpy(dtype=np.float64)
iforest = pred[if_col].to_numpy(dtype=np.float64)
behaviour = pred[behaviour_col].to_numpy(dtype=np.float64)

components = {
    "GNN": gnn,
    "Isolation_Forest": iforest,
    "Behaviour": behaviour
}

for name, values in components.items():
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite scores.")

    if values.min() < 0 or values.max() > 1:
        raise ValueError(
            f"{name} ranks must be between 0 and 1. "
            "Check the prediction CSV column."
        )


# ============================================================
# 6. TEST LABELS
# ============================================================

# The existing prediction file follows the original test order:
# benign test first, followed by DoS Hulk test.

y_true = np.concatenate([
    np.zeros(len(benign_test), dtype=np.int32),
    np.ones(len(attack_test), dtype=np.int32)
])

if len(y_true) != len(pred):
    raise RuntimeError("Test labels and predictions are misaligned.")


# ============================================================
# 7. DEFINE FIXED COMPONENT COMBINATIONS
# ============================================================

# Original weights:
# GNN = 0.35
# Isolation Forest = 0.40
# Behaviour = 0.25
#
# Pair weights are normalized from the original weights.
# No attack test labels are used to select weights.

experiments = {
    "GNN_only": {
        "weights": {
            "GNN": 1.0
        }
    },

    "Isolation_Forest_only": {
        "weights": {
            "Isolation_Forest": 1.0
        }
    },

    "Behaviour_only": {
        "weights": {
            "Behaviour": 1.0
        }
    },

    "GNN_IF": {
        "weights": {
            "GNN": 0.35 / 0.75,
            "Isolation_Forest": 0.40 / 0.75
        }
    },

    "GNN_Behaviour": {
        "weights": {
            "GNN": 0.35 / 0.60,
            "Behaviour": 0.25 / 0.60
        }
    },

    "IF_Behaviour": {
        "weights": {
            "Isolation_Forest": 0.40 / 0.65,
            "Behaviour": 0.25 / 0.65
        }
    },

    "Original_Three_Component": {
        "weights": {
            "GNN": 0.35,
            "Isolation_Forest": 0.40,
            "Behaviour": 0.25
        }
    }
}


# ============================================================
# 8. CONSERVATIVE VALIDATION THRESHOLD
# ============================================================

# Component ranks are empirical percentile ranks against benign
# validation scores. For k components, use a component-rank
# cutoff of 1 - 0.01/k.
#
# If the weighted average exceeds this cutoff, at least one
# included component must exceed its individual cutoff.
# By the union bound, validation FPR is at most 1%.
#
# This is a conservative bound, not an exact joint 99th
# percentile of the combined validation score.

TARGET_VALIDATION_FPR = 0.01


# ============================================================
# 9. EVALUATE EACH COMBINATION
# ============================================================

results = []
prediction_outputs = []

for experiment_name, config in experiments.items():

    weights = config["weights"]
    number_of_components = len(weights)

    component_cutoff = (
        1.0 - TARGET_VALIDATION_FPR / number_of_components
    )

    combined_score = np.zeros(len(pred), dtype=np.float64)

    for component_name, weight in weights.items():
        combined_score += (
            weight * components[component_name]
        )

    # Conservative threshold derived from benign validation
    # percentile ranks, not from held-out test labels.
    threshold = component_cutoff

    y_pred = (combined_score > threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    roc_auc = roc_auc_score(
        y_true,
        combined_score
    )

    pr_auc = average_precision_score(
        y_true,
        combined_score
    )

    result = {
        "experiment": experiment_name,
        "components": "+".join(weights.keys()),
        "weights": json.dumps(weights),
        "threshold": threshold,
        "component_rank_cutoff": component_cutoff,
        "validation_fpr_upper_bound": TARGET_VALIDATION_FPR,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "FPR": fpr,
        "FNR": fnr,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    results.append(result)

    prediction_outputs.append(
        pd.DataFrame({
            "Experiment": experiment_name,
            "Actual_Label": y_true,
            "Combined_Score": combined_score,
            "Threshold": threshold,
            "Predicted_Label": y_pred
        })
    )

    print("\n" + "-" * 75)
    print(experiment_name)
    print("-" * 75)

    for key, value in result.items():
        print(f"{key}: {value}")


# ============================================================
# 10. SAVE COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1",
    ascending=False
)

results_path = OUTPUT_DIR / "ensemble_comparison.csv"
results_df.to_csv(results_path, index=False)

all_predictions = pd.concat(
    prediction_outputs,
    ignore_index=True
)

predictions_path = OUTPUT_DIR / "ensemble_predictions.csv"
all_predictions.to_csv(predictions_path, index=False)


# ============================================================
# 11. SAVE SUMMARY JSON
# ============================================================

summary = {
    "experiment": "DoS Hulk component combination comparison",
    "training_performed": False,
    "existing_model_modified": False,
    "threshold_method": (
        "Conservative validation percentile-rank cutoff "
        "using the union bound"
    ),
    "target_validation_fpr_upper_bound": TARGET_VALIDATION_FPR,
    "test_rows": int(len(pred)),
    "benign_test_rows": int(len(benign_test)),
    "dos_hulk_test_rows": int(len(attack_test)),
    "results": results
}

summary_path = OUTPUT_DIR / "ensemble_comparison_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)


# ============================================================
# 12. FINAL DISPLAY
# ============================================================

print("\n" + "=" * 75)
print("ENSEMBLE COMPARISON COMPLETED")
print("=" * 75)

display_columns = [
    "experiment",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "FPR",
    "FNR",
    "ROC_AUC",
    "PR_AUC"
]

print(
    results_df[display_columns].to_string(index=False)
)

print("\nSaved files:")
print(results_path)
print(predictions_path)
print(summary_path)

print("\nOriginal DoS Hulk model and reports were not modified.")
print("No model retraining was performed.")
