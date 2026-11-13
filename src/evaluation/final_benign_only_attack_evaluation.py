
"""
Final benign-only recalibration and evaluation.

Processes:
    DDoS
    Bot
    DoS Hulk
    DoS GoldenEye
    Web Attack
    Infiltration

Protocol:
    1. Uses existing V1 score files.
    2. Uses benign validation scores as reference distributions.
    3. Uses fixed ensemble weights.
    4. Calibrates threshold using benign validation only.
    5. Uses attack labels only for final evaluation.

Does not:
    - Train on attack samples.
    - Tune weights using attack labels.
    - Change existing V1/V2 checkpoints.
    - Overwrite existing reports.
"""

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


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

# V2 outputs are used for DDoS / DoS Hulk / GoldenEye.
# V3 outputs are used for Bot / Infiltration / Web Attack.
ATTACK_REPORT_DIRS = {
    "ddos": ROOT / "reports" / "zero_day" / "attack_v2" / "ddos",
    "dos_hulk": ROOT / "reports" / "zero_day" / "attack_v2" / "dos_hulk",
    "dos_goldeneye": ROOT / "reports" / "zero_day" / "attack_v2" / "dos_goldeneye",
    "bot": ROOT / "reports" / "zero_day" / "attack_v3" / "bot",
    "infiltration": ROOT / "reports" / "zero_day" / "attack_v3" / "infiltration",
    "web_attack": ROOT / "reports" / "zero_day" / "attack_v3" / "web_attack",
}

OUTPUT_DIR = (
    ROOT / "reports" / "zero_day" / "benign_only_benchmark"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACKS = [
    "ddos",
    "bot",
    "dos_hulk",
    "dos_goldeneye",
    "web_attack",
    "infiltration"
]

# Fixed weights. Never optimise these using attack labels.
WEIGHTS = {
    "gnn_score": 0.35,
    "if_score": 0.40,
    "scan_score": 0.25
}

TARGET_BENIGN_FPR = 0.01

SCORE_COLUMNS = [
    "gnn_score",
    "if_score",
    "scan_score"
]


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(path)


def validate_scores(dataframe, columns, description):

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{description} is missing columns: {missing}"
        )

    scores = dataframe[columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if scores.isna().any().any():
        raise ValueError(
            f"{description} contains missing or non-numeric scores."
        )

    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError(
            f"{description} contains infinite scores."
        )

    return scores


def percentile_scores(reference, values):
    """
    Convert scores into empirical percentiles.

    The reference distribution comes only from benign
    validation traffic.

    Higher percentile means more anomalous.
    """

    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)

    reference = np.sort(reference)

    ranks = np.searchsorted(
        reference,
        values,
        side="right"
    )

    return ranks / len(reference)


def convert_labels(label_series):
    """
    Convert ground-truth labels:
        0 = benign
        1 = attack

    Used only after predictions have been generated.
    """

    values = (
        label_series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    benign_values = {
        "BENIGN",
        "NORMAL",
        "NORMAL_TRAFFIC",
        "BENIGN_TRAFFIC",
        "0",
        "0.0"
    }

    return (~values.isin(benign_values)).astype(int).to_numpy()


def calculate_metrics(y_true, scores, threshold):

    y_pred = (
        np.asarray(scores) > threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "false_positive_rate": float(
            fp / (fp + tn) if (fp + tn) else 0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    if len(np.unique(y_true)) == 2:

        metrics["roc_auc"] = float(
            roc_auc_score(y_true, scores)
        )

        metrics["pr_auc"] = float(
            average_precision_score(y_true, scores)
        )

    else:

        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics, y_pred


# =========================================================
# 3. PROCESS ONE ATTACK
# =========================================================

def process_attack(attack):

    print("\n" + "=" * 70)
    print(f"PROCESSING {attack.upper()}")
    print("=" * 70)

    attack_dir = ATTACK_REPORT_DIRS[attack]

    predictions_path = (
        attack_dir / "predictions.csv"
    )

    validation_path = (
        attack_dir / "validation_scores.csv"
    )

    predictions_df = load_csv(predictions_path)
    validation_df = load_csv(validation_path)

    # -----------------------------------------------------
    # A. Validate the actual score columns
    # -----------------------------------------------------

    test_scores = validate_scores(
        predictions_df,
        SCORE_COLUMNS,
        "Test predictions"
    )

    validation_scores = validate_scores(
        validation_df,
        SCORE_COLUMNS,
        "Validation scores"
    )

    if "Label" not in predictions_df.columns:
        raise ValueError(
            f"Label column not found for {attack}"
        )

    # -----------------------------------------------------
    # B. Transform validation scores
    # -----------------------------------------------------

    validation_percentiles = {}

    for column in SCORE_COLUMNS:

        validation_percentiles[column] = (
            percentile_scores(
                validation_scores[column].to_numpy(),
                validation_scores[column].to_numpy()
            )
        )

    validation_combined = np.zeros(
        len(validation_df),
        dtype=float
    )

    for column, weight in WEIGHTS.items():

        validation_combined += (
            weight * validation_percentiles[column]
        )

    # Threshold is determined only from benign validation.
    threshold = float(
        np.quantile(
            validation_combined,
            1 - TARGET_BENIGN_FPR,
            method="higher"
        )
    )

    validation_predictions = (
        validation_combined > threshold
    ).astype(int)

    validation_fpr = float(
        validation_predictions.mean()
    )

    # -----------------------------------------------------
    # C. Transform test scores using frozen references
    # -----------------------------------------------------

    test_percentiles = {}

    for column in SCORE_COLUMNS:

        test_percentiles[column] = (
            percentile_scores(
                validation_scores[column].to_numpy(),
                test_scores[column].to_numpy()
            )
        )

    # -----------------------------------------------------
    # D. Calculate the final anomaly score
    # -----------------------------------------------------

    combined_score = np.zeros(
        len(predictions_df),
        dtype=float
    )

    for column, weight in WEIGHTS.items():

        combined_score += (
            weight * test_percentiles[column]
        )

    # Freeze predictions before reading labels.
    frozen_predictions = (
        combined_score > threshold
    ).astype(int)

    # -----------------------------------------------------
    # E. Evaluate using labels only now
    # -----------------------------------------------------

    y_true = convert_labels(
        predictions_df["Label"]
    )

    metrics, evaluated_predictions = calculate_metrics(
        y_true,
        combined_score,
        threshold
    )

    assert np.array_equal(
        frozen_predictions,
        evaluated_predictions
    )

    # -----------------------------------------------------
    # F. Save results
    # -----------------------------------------------------

    attack_output = OUTPUT_DIR / attack

    attack_output.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df = predictions_df.copy()

    result_df["gnn_percentile"] = (
        test_percentiles["gnn_score"]
    )

    result_df["iforest_percentile"] = (
        test_percentiles["if_score"]
    )

    result_df["scan_percentile"] = (
        test_percentiles["scan_score"]
    )

    result_df["benign_only_score"] = (
        combined_score
    )

    result_df["benign_only_prediction"] = (
        frozen_predictions
    )

    result_df.to_csv(
        attack_output / "predictions.csv",
        index=False
    )

    result = {
        "attack": attack,
        "method": "Benign-only percentile score calibration",
        "weights": WEIGHTS,
        "threshold": threshold,
        "target_validation_fpr": TARGET_BENIGN_FPR,
        "actual_validation_fpr": validation_fpr,
        "validation_samples": int(len(validation_df)),
        "test_samples": int(len(predictions_df)),
        "attack_labels_used_for_calibration": False,
        "metrics": metrics
    }

    with open(
        attack_output / "metrics.json",
        "w"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print("\nFixed weights:", WEIGHTS)
    print("Threshold:", threshold)
    print("Validation FPR:", validation_fpr)

    print("\nTest metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    return {
        "attack": attack,
        "threshold": threshold,
        "validation_fpr": validation_fpr,
        **metrics
    }


# =========================================================
# 4. RUN ALL SIX ATTACK EXPERIMENTS
# =========================================================

def main():

    # Existing benchmark results are preserved. Move/archive them manually
    # before intentionally regenerating this evaluation.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing benchmark results: {OUTPUT_DIR}"
        )

    results = []
    failures = []

    for attack in ATTACKS:

        try:

            result = process_attack(attack)
            results.append(result)

        except Exception as error:

            print(
                f"\nFAILED: {attack.upper()}"
            )

            print(str(error))

            failures.append({
                "attack": attack,
                "error": str(error)
            })

    # -----------------------------------------------------
    # A. Combined comparison
    # -----------------------------------------------------

    if results:

        comparison = pd.DataFrame(results)

        comparison.to_csv(
            OUTPUT_DIR / "all_attack_metrics.csv",
            index=False
        )

        with open(
            OUTPUT_DIR / "all_attack_metrics.json",
            "w"
        ) as file:

            json.dump(
                results,
                file,
                indent=4
            )

    # -----------------------------------------------------
    # B. Failure report
    # -----------------------------------------------------

    with open(
        OUTPUT_DIR / "failed_experiments.json",
        "w"
    ) as file:

        json.dump(
            failures,
            file,
            indent=4
        )

    # -----------------------------------------------------
    # C. Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL BENIGN-ONLY EVALUATION")
    print("=" * 70)

    if results:

        print(
            pd.DataFrame(results).to_string(
                index=False
            )
        )

    print("\nCompleted:", len(results))
    print("Failed:", len(failures))

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()

