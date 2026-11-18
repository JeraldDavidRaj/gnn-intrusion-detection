from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

# ============================================================
# 1. PATHS AND PAGE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_ZERO_DAY_DIR = DATA_DIR / "zero_day"
DATA_REMAINING_DIR = DATA_DIR / "remaining_attacks"

# PortScan V6
PORTSCAN_MODEL_DIR = BASE_DIR / "models" / "portscan_v6"
PORTSCAN_REPORT_DIR = (
    BASE_DIR / "reports" / "zero_day" / "portscan_v6"
)
PORTSCAN_EVAL_DIR = PORTSCAN_REPORT_DIR / "evaluation"
PORTSCAN_PUBLICATION_DIR = PORTSCAN_REPORT_DIR / "publication"

# DoS Hulk remains in its original experiment directory.
DOS_HULK_DIR = BASE_DIR / "experiments" / "dos_hulk"
DOS_HULK_MODEL_DIR = DOS_HULK_DIR / "models"
DOS_HULK_REPORT_DIR = DOS_HULK_DIR / "reports"
DOS_HULK_DIAGNOSTICS_DIR = DOS_HULK_REPORT_DIR / "diagnostics"
DOS_HULK_OPTIMIZATION_DIR = (
    DOS_HULK_REPORT_DIR / "ensemble_optimization"
)

# Hybrid GNN
HYBRID_MODEL_DIR = BASE_DIR / "models" / "hybrid"
HYBRID_REPORT_DIR = BASE_DIR / "reports" / "hybrid"

st.set_page_config(
    page_title="GNN Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. GENERAL HELPERS
# ============================================================

st.title("🛡️ GNN-Based Intrusion Detection System")
st.caption(
    "CIC-IDS2017 | Graph Neural Networks | "
    "PortScan V6 · DoS Hulk · Hybrid GNN"
)


@st.cache_data(show_spinner=False)
def load_csv(path_string, modified_time):
    """Load a CSV and refresh the cache when its file timestamp changes."""
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as error:
        return pd.DataFrame({"_load_error": [str(error)]})


@st.cache_data(show_spinner=False)
def load_json(path_string, modified_time):
    path = Path(path_string)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def file_mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def csv_at(path):
    return load_csv(str(path), file_mtime(path))


def json_at(path):
    return load_json(str(path), file_mtime(path))


def flatten_dict(data, prefix=""):
    result = {}
    if not isinstance(data, dict):
        return result
    for key, value in data.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_dict(value, name))
        elif isinstance(value, (int, float, str, bool)) or value is None:
            result[name] = value
    return result


def normalize_key(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("%", "percent")
    )


def get_metric(metrics, aliases):
    flat = flatten_dict(metrics)
    lookup = {normalize_key(key): value for key, value in flat.items()}
    for alias in aliases:
        value = lookup.get(normalize_key(alias))
        if value is not None:
            try:
                value = float(value)
                if math.isfinite(value):
                    return value
            except (TypeError, ValueError):
                continue
    return None


def to_percent(value):
    if value is None:
        return None
    return value * 100 if 0 <= value <= 1 else value


def format_percent(value, digits=3):
    value = to_percent(value)
    return "N/A" if value is None else f"{value:.{digits}f}%"


def format_number(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def show_metric_cards(metrics, include_balanced_accuracy=True):
    definitions = [
        ("Accuracy", ["accuracy", "acc"]),
        ("Precision", ["precision"]),
        ("Recall", ["recall", "sensitivity", "tpr"]),
        ("F1 Score", ["f1_score", "f1"]),
        ("False Positive Rate", ["false_positive_rate", "fpr"]),
        ("ROC-AUC", ["roc_auc", "rocauc"]),
        ("PR-AUC", ["pr_auc", "average_precision"]),
    ]
    if include_balanced_accuracy:
        definitions.append(
            ("Balanced Accuracy", ["balanced_accuracy"])
        )

    found = [
        (label, get_metric(metrics, aliases))
        for label, aliases in definitions
    ]
    found = [(label, value) for label, value in found if value is not None]

    if not found:
        st.warning("No recognised evaluation metrics were found.")
        return

    for start in range(0, len(found), 4):
        group = found[start:start + 4]
        columns = st.columns(len(group))
        for column, (label, value) in zip(columns, group):
            column.metric(label, format_percent(value))


def display_metrics(metrics):
    if not metrics:
        st.warning("Evaluation metrics JSON is missing or could not be read.")
        return
    show_metric_cards(metrics)
    with st.expander("View complete saved metrics JSON"):
        st.json(metrics)


def artifact_status(paths):
    rows = []
    for label, path in paths:
        rows.append(
            {
                "Artifact": label,
                "Path": str(path.relative_to(BASE_DIR))
                if path.is_relative_to(BASE_DIR)
                else str(path),
                "Status": "Available" if path.exists() else "Missing",
                "Size (MB)": (
                    round(path.stat().st_size / (1024 * 1024), 2)
                    if path.is_file()
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def display_artifact_status(paths):
    table = artifact_status(paths)
    available = int((table["Status"] == "Available").sum())
    st.caption(f"{available} of {len(table)} listed artifacts available.")
    st.dataframe(table, width="stretch", hide_index=True)


# ============================================================
# 3. PREDICTION / EVALUATION HELPERS
# ============================================================

NORMAL_LABELS = {
    "BENIGN",
    "NORMAL",
    "0",
    "FALSE",
    "NO ATTACK",
    "NON-ATTACK",
    "NONATTACK",
}


def binary_labels(series):
    values = series.astype(str).str.strip().str.upper()
    return values.apply(lambda value: 0 if value in NORMAL_LABELS else 1)


def find_column(dataframe, candidates):
    if dataframe.empty:
        return None
    lookup = {normalize_key(column): column for column in dataframe.columns}
    for candidate in candidates:
        match = lookup.get(normalize_key(candidate))
        if match is not None:
            return match
    return None


def plot_confusion(dataframe, actual_col, predicted_col, title):
    if dataframe.empty:
        st.info("Prediction records are unavailable.")
        return

    missing = [
        name for name in [actual_col, predicted_col]
        if name not in dataframe.columns
    ]
    if missing:
        st.info(f"Confusion matrix unavailable; missing: {', '.join(missing)}")
        return

    data = dataframe[[actual_col, predicted_col]].dropna()
    if data.empty:
        st.info("No valid prediction rows for the confusion matrix.")
        return

    y_true = binary_labels(data[actual_col])
    y_pred = binary_labels(data[predicted_col])
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(4.2, 3.1), dpi=90)
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Benign", "Attack"],
    ).plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    axis.set_title(title, fontsize=12)
    axis.tick_params(axis="both", labelsize=9)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    figure.tight_layout()
    st.pyplot(figure, width="content")
    plt.close(figure)

    tn, fp, fn, tp = matrix.ravel()
    columns = st.columns(4)
    columns[0].metric("True Negatives", format_number(tn))
    columns[1].metric("False Positives", format_number(fp))
    columns[2].metric("False Negatives", format_number(fn))
    columns[3].metric("True Positives", format_number(tp))


def plot_roc_pr(dataframe, actual_col, score_col, title):
    if dataframe.empty:
        st.info("Prediction records are unavailable for ROC/PR curves.")
        return
    if actual_col not in dataframe.columns or score_col not in dataframe.columns:
        st.info(
            f"ROC/PR curves need '{actual_col}' and '{score_col}' columns."
        )
        return

    data = dataframe[[actual_col, score_col]].copy()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data = data.dropna()
    data = data[np.isfinite(data[score_col])]

    if data.empty:
        st.info("No valid score values are available.")
        return

    y_true = binary_labels(data[actual_col])
    scores = data[score_col].astype(float)

    if y_true.nunique() != 2 or scores.nunique() < 2:
        st.info("ROC/PR curves require both classes and continuous scores.")
        return

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_value = roc_auc_score(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_value = average_precision_score(y_true, scores)

    left, right = st.columns(2)
    with left:
        figure, axis = plt.subplots(figsize=(4.2, 3.1), dpi=90)
        axis.plot(fpr, tpr, label=f"ROC-AUC = {roc_value:.4f}")
        axis.plot([0, 1], [0, 1], "--", label="Random baseline")
        axis.set(
            title=f"{title} — ROC Curve",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        axis.grid(alpha=0.25)
        axis.tick_params(axis="both", labelsize=9)
        axis.legend()
        figure.tight_layout()
        st.pyplot(figure, width="content")
        plt.close(figure)

    with right:
        figure, axis = plt.subplots(figsize=(4.2, 3.1), dpi=90)
        axis.plot(recall, precision, label=f"Average Precision = {pr_value:.4f}")
        axis.set(
            title=f"{title} — Precision-Recall Curve",
            xlabel="Recall",
            ylabel="Precision",
        )
        axis.grid(alpha=0.25)
        axis.tick_params(axis="both", labelsize=9)
        axis.legend()
        figure.tight_layout()
        st.pyplot(figure, width="content")
        plt.close(figure)

    st.caption(
        f"Recalculated from {len(data):,} saved prediction rows. "
        f"ROC-AUC = {roc_value:.5f}; Average Precision = {pr_value:.5f}."
    )


def plot_score_distribution(dataframe, actual_col, score_col, title, score_label):
    if dataframe.empty:
        st.info("Prediction records are unavailable.")
        return
    if actual_col not in dataframe.columns or score_col not in dataframe.columns:
        st.info(
            f"Score distribution requires '{actual_col}' and '{score_col}'."
        )
        return

    data = dataframe[[actual_col, score_col]].copy()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data = data.dropna()
    data = data[np.isfinite(data[score_col])]
    if data.empty:
        st.info("No valid anomaly scores are available.")
        return

    data["Class"] = binary_labels(data[actual_col]).map(
        {0: "Benign", 1: "Attack"}
    )

    figure, axis = plt.subplots(figsize=(5.0, 3.2), dpi=90)
    for label in ["Benign", "Attack"]:
        values = data.loc[data["Class"] == label, score_col]
        if not values.empty:
            axis.hist(
                values,
                bins=50,
                density=True,
                alpha=0.55,
                label=label,
            )
    axis.set(
        title=f"{title} — {score_label} Distribution",
        xlabel=score_label,
        ylabel="Density",
    )
    axis.grid(alpha=0.2)
    axis.tick_params(axis="both", labelsize=9)
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure, width="content")
    plt.close(figure)


def plot_training_history(
    dataframe,
    title,
    epoch_col="epoch",
    train_col=None,
    validation_col=None,
):
    if dataframe.empty:
        st.info(f"{title} training history is unavailable.")
        return

    if train_col is None or epoch_col not in dataframe.columns:
        st.info(f"{title} training-history columns are unavailable.")
        return

    required = [epoch_col, train_col]
    if validation_col and validation_col in dataframe.columns:
        required.append(validation_col)

    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        st.info(f"Training-history columns missing: {', '.join(missing)}")
        return

    data = dataframe[required].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[epoch_col, train_col]).sort_values(epoch_col)

    if data.empty:
        st.info("No valid training-history rows.")
        return

    figure, axis = plt.subplots(figsize=(5.0, 3.2), dpi=90)
    axis.plot(data[epoch_col], data[train_col], marker="o", label=train_col)

    if validation_col and validation_col in data.columns:
        valid = data.dropna(subset=[validation_col])
        if not valid.empty:
            axis.plot(
                valid[epoch_col],
                valid[validation_col],
                marker="o",
                label=validation_col,
            )

    axis.set(
        title=f"{title} — Training History",
        xlabel="Epoch",
        ylabel="Recorded value",
    )
    axis.grid(alpha=0.25)
    axis.tick_params(axis="both", labelsize=9)
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure, width="content")
    plt.close(figure)


def show_distribution(dataframe, column, title):
    if dataframe.empty or column not in dataframe.columns:
        st.info(f"{title} is unavailable.")
        return
    counts = dataframe[column].astype(str).value_counts()
    st.bar_chart(counts)


def show_filtered_table(dataframe, download_name, key, height=420):
    if dataframe.empty:
        st.info("No prediction records are available.")
        return

    search = st.text_input("Search prediction records", key=f"{key}_search")
    filtered = dataframe.copy()

    if search:
        mask = filtered.astype(str).apply(
            lambda column: column.str.contains(
                search,
                case=False,
                na=False,
                regex=False,
            )
        ).any(axis=1)
        filtered = filtered[mask]

    st.write(f"Matching records: {len(filtered):,}")
    st.dataframe(filtered, width="stretch", height=height)
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name=download_name,
        mime="text/csv",
        key=f"{key}_download",
    )


def show_threshold_tradeoff(metrics, attack_detection_key, title):
    tradeoff = metrics.get("threshold_tradeoff", {})
    if not isinstance(tradeoff, dict) or not tradeoff:
        st.info("Threshold trade-off data is not present in the saved metrics.")
        return

    rows = []
    for percentile, values in tradeoff.items():
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "Validation percentile": str(percentile),
                "Threshold": values.get("threshold"),
                "Benign test FPR": to_percent(
                    values.get("benign_test_fpr")
                ),
                "Attack detection rate": to_percent(
                    values.get(attack_detection_key)
                ),
            }
        )

    if not rows:
        st.info("No usable threshold trade-off rows were found.")
        return

    table = pd.DataFrame(rows)
    st.dataframe(table, width="stretch", hide_index=True)
    chart_data = table.set_index("Validation percentile")[
        ["Benign test FPR", "Attack detection rate"]
    ]
    st.line_chart(chart_data)
    st.caption(
        f"{title}: these are saved threshold-sensitivity results, "
        "not a newly calibrated operating point."
    )


def show_diagnostic_csvs(folder, candidates):
    available = [
        (label, folder / filename)
        for label, filename in candidates
        if (folder / filename).exists()
    ]
    if not available:
        st.info("No matching diagnostic CSVs were found.")
        return

    labels = [label for label, _ in available]
    selected_label = st.selectbox(
        "Choose diagnostic report",
        labels,
        key=f"diag_{folder.name}_{len(labels)}",
    )
    selected_path = dict(available)[selected_label]
    diagnostic = csv_at(selected_path)

    st.caption(f"Source: {selected_path.relative_to(BASE_DIR)}")
    st.dataframe(diagnostic, width="stretch", height=400)
    st.download_button(
        "Download diagnostic CSV",
        diagnostic.to_csv(index=False).encode("utf-8"),
        file_name=selected_path.name,
        mime="text/csv",
        key=f"download_{selected_path.name}",
    )


# ============================================================
# 4. LOAD SAVED EXPERIMENT REPORTS
# ============================================================

# PortScan V6
portscan_metrics_path = PORTSCAN_EVAL_DIR / "zero_day_v6_evaluation_metrics.json"
portscan_calibration_path = PORTSCAN_EVAL_DIR / "zero_day_v6_calibration.json"
portscan_history_path = PORTSCAN_EVAL_DIR / "zero_day_v6_training_history.csv"
portscan_eval_predictions_path = PORTSCAN_EVAL_DIR / "zero_day_v6_flow_predictions.csv"
portscan_publication_path = PORTSCAN_PUBLICATION_DIR / "all_test_flow_predictions.csv"
portscan_detected_path = PORTSCAN_PUBLICATION_DIR / "combined_v6_detected_portscan.csv"
portscan_missed_path = PORTSCAN_PUBLICATION_DIR / "combined_v6_missed_portscan.csv"
portscan_fp_path = PORTSCAN_PUBLICATION_DIR / "combined_v6_false_positives.csv"
portscan_ablation_path = PORTSCAN_PUBLICATION_DIR / "publication_ablation_results.csv"

portscan_metrics = json_at(portscan_metrics_path)
portscan_calibration = json_at(portscan_calibration_path)
portscan_history = csv_at(portscan_history_path)
portscan_eval_predictions = csv_at(portscan_eval_predictions_path)
portscan_publication = csv_at(portscan_publication_path)
portscan_detected = csv_at(portscan_detected_path)
portscan_missed = csv_at(portscan_missed_path)
portscan_false_positives = csv_at(portscan_fp_path)
portscan_ablation = csv_at(portscan_ablation_path)

portscan_predictions = (
    portscan_publication
    if not portscan_publication.empty
    else portscan_eval_predictions
)

# DoS Hulk
dos_metrics_path = DOS_HULK_REPORT_DIR / "dos_hulk_evaluation_metrics.json"
dos_calibration_path = DOS_HULK_REPORT_DIR / "dos_hulk_calibration.json"
dos_history_path = DOS_HULK_REPORT_DIR / "dos_hulk_training_history.csv"
dos_predictions_path = DOS_HULK_REPORT_DIR / "dos_hulk_flow_predictions.csv"

dos_metrics = json_at(dos_metrics_path)
dos_calibration = json_at(dos_calibration_path)
dos_history = csv_at(dos_history_path)
dos_predictions = csv_at(dos_predictions_path)

# Hybrid GNN
hybrid_metrics_path = HYBRID_REPORT_DIR / "hybrid_gnn_evaluation_metrics.json"
hybrid_predictions_path = HYBRID_REPORT_DIR / "hybrid_gnn_flow_predictions.csv"
hybrid_history_path = HYBRID_REPORT_DIR / "hybrid_gnn_training_history.csv"

hybrid_metrics = json_at(hybrid_metrics_path)
hybrid_predictions = csv_at(hybrid_predictions_path)
hybrid_history = csv_at(hybrid_history_path)


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛡️ Navigation")
page = st.sidebar.radio(
    "Select dashboard page",
    [
        "Project Overview",
        "PortScan V6 — Zero-Day",
        "DoS Hulk — Zero-Day",
        "Hybrid GNN",
        "Model Comparison",
        "Flow Explorer",
        "Dataset Inspection",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Experiments")
st.sidebar.write(
    "• PortScan V6: benign-only zero-day evaluation\n\n"
    "• DoS Hulk: benign-only zero-day evaluation\n\n"
    "• Hybrid GNN: supervised unseen-category evaluation"
)
st.sidebar.caption(
    "This dashboard visualises saved experiment artifacts. "
    "It does not perform new model inference on uploaded CSV files."
)


# ============================================================
# 6. PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":
    st.header("Project Overview")
    st.markdown(
        """
        This project investigates graph-based network intrusion detection
        using CIC-IDS2017 traffic. It contains three complementary experiments.

        - **PortScan V6:** benign-only training, with PortScan held out for testing.
        - **DoS Hulk:** benign-only training, with DoS Hulk held out for testing.
        - **Hybrid GNN:** supervised training on benign traffic and selected known
          attack categories, with PortScan excluded from training and validation.
        """
    )

    st.info(
        "PortScan V6 and DoS Hulk are the two benign-only attack-specific "
        "zero-day experiments. Hybrid GNN is a complementary unseen-category "
        "generalisation experiment; it uses a different training setup."
    )

    st.divider()
    cards = st.columns(3)

    experiments = [
        (
            "PortScan V6",
            portscan_metrics,
            "Benign-only training",
            "Held-out PortScan",
            "models/portscan_v6/",
        ),
        (
            "DoS Hulk",
            dos_metrics,
            "Benign-only training",
            "Held-out DoS Hulk",
            "experiments/dos_hulk/models/",
        ),
        (
            "Hybrid GNN",
            hybrid_metrics,
            "Benign + selected known attacks",
            "Held-out PortScan",
            "models/hybrid/",
        ),
    ]

    for column, (name, metrics, training, testing, model_path) in zip(cards, experiments):
        with column:
            st.subheader(name)
            if metrics:
                st.metric("F1 Score", format_percent(
                    get_metric(metrics, ["f1_score", "f1"])
                ))
                st.metric("Recall", format_percent(
                    get_metric(metrics, ["recall"])
                ))
            else:
                st.warning("Metrics report not found.")
            st.caption(f"Training: {training}")
            st.caption(f"Evaluation: {testing}")
            st.code(model_path)

    st.divider()
    st.subheader("Experiment artifact availability")

    overview_artifacts = [
        ("PortScan V6 checkpoint", PORTSCAN_MODEL_DIR / "zero_day_gnn_v6_best.pth"),
        ("PortScan V6 metrics", portscan_metrics_path),
        ("PortScan V6 predictions", portscan_publication_path),
        ("DoS Hulk checkpoint", DOS_HULK_MODEL_DIR / "dos_hulk_gnn_best.pth"),
        ("DoS Hulk metrics", dos_metrics_path),
        ("DoS Hulk predictions", dos_predictions_path),
        ("Hybrid GNN checkpoint", HYBRID_MODEL_DIR / "hybrid_gnn_best.pth"),
        ("Hybrid GNN metrics", hybrid_metrics_path),
        ("Hybrid GNN predictions", hybrid_predictions_path),
    ]
    display_artifact_status(overview_artifacts)

    st.divider()
    st.subheader("Research scope")
    scope = pd.DataFrame(
        [
            [
                "PortScan V6",
                "Benign-only",
                "PortScan",
                "Attack-specific zero-day evaluation",
            ],
            [
                "DoS Hulk",
                "Benign-only",
                "DoS Hulk",
                "Attack-specific zero-day evaluation",
            ],
            [
                "Hybrid GNN",
                "Benign + selected known attack categories",
                "PortScan",
                "Unseen attack-category generalisation",
            ],
        ],
        columns=[
            "Experiment",
            "Training setup",
            "Held-out attack",
            "Research role",
        ],
    )
    st.dataframe(scope, width="stretch", hide_index=True)


# ============================================================
# 7. PORTSCAN V6 PAGE
# ============================================================

elif page == "PortScan V6 — Zero-Day":
    st.header("PortScan V6 — Strict Zero-Day Detection")
    st.caption(
        "Benign-only training with PortScan excluded from model fitting "
        "and threshold calibration."
    )

    st.subheader("Saved evaluation metrics")
    display_metrics(portscan_metrics)

    st.subheader("Model artifacts")
    display_artifact_status(
        [
            ("GNN checkpoint", PORTSCAN_MODEL_DIR / "zero_day_gnn_v6_best.pth"),
            ("Flow imputer", PORTSCAN_MODEL_DIR / "zero_day_imputer.pkl"),
            ("Flow scaler", PORTSCAN_MODEL_DIR / "zero_day_scaler.pkl"),
            ("Temporal Isolation Forest", PORTSCAN_MODEL_DIR / "zero_day_v6_temporal_iforest.pkl"),
            ("Temporal scaler", PORTSCAN_MODEL_DIR / "zero_day_v6_temporal_scaler.pkl"),
        ]
    )

    with st.expander("Threshold calibration details"):
        if portscan_calibration:
            st.json(portscan_calibration)
        else:
            st.info("PortScan calibration JSON was not found.")

    st.divider()
    st.subheader("Confusion matrix")
    plot_confusion(
        portscan_predictions,
        "Actual_Label",
        "Predicted_Label",
        "PortScan V6 Confusion Matrix",
    )

    score_column = find_column(
        portscan_predictions,
        ["Combined_V6_Score", "Combined_Anomaly_Score"],
    )

    st.subheader("ROC and precision-recall curves")
    if score_column:
        plot_roc_pr(
            portscan_predictions,
            "Actual_Label",
            score_column,
            "PortScan V6",
        )
        st.subheader("Anomaly score distribution")
        plot_score_distribution(
            portscan_predictions,
            "Actual_Label",
            score_column,
            "PortScan V6",
            score_column,
        )
    else:
        st.info("No combined anomaly-score column was found.")

    st.subheader("Training history")
    plot_training_history(
        portscan_history,
        "PortScan V6",
        train_col="training_nll",
        validation_col="validation_nll",
    )

    st.subheader("Threshold sensitivity")
    show_threshold_tradeoff(
        portscan_metrics,
        "portscan_detection_rate",
        "PortScan V6",
    )

    st.divider()
    st.subheader("Prediction distribution")
    show_distribution(
        portscan_predictions,
        "Predicted_Label",
        "PortScan V6 prediction distribution",
    )

    st.subheader("Saved flow predictions")
    show_filtered_table(
        portscan_predictions,
        "portscan_v6_predictions_filtered.csv",
        "portscan_v6",
    )

    st.divider()
    st.subheader("Publication evaluation outputs")
    tab_detected, tab_missed, tab_fp, tab_ablation = st.tabs(
        [
            "Detected PortScan",
            "Missed PortScan",
            "False positives",
            "Ablation",
        ]
    )
    with tab_detected:
        st.write(f"Rows: {len(portscan_detected):,}")
        st.dataframe(portscan_detected, width="stretch", height=350)
    with tab_missed:
        st.write(f"Rows: {len(portscan_missed):,}")
        st.dataframe(portscan_missed, width="stretch", height=350)
    with tab_fp:
        st.write(f"Rows: {len(portscan_false_positives):,}")
        st.dataframe(portscan_false_positives, width="stretch", height=350)
    with tab_ablation:
        st.dataframe(portscan_ablation, width="stretch", height=350)


# ============================================================
# 8. DOS HULK PAGE
# ============================================================

elif page == "DoS Hulk — Zero-Day":
    st.header("DoS Hulk — Zero-Day Detection")
    st.caption(
        "Benign-only temporal GCN reconstruction and benign-only temporal "
        "Isolation Forest, evaluated on held-out DoS Hulk traffic."
    )

    st.subheader("Saved evaluation metrics")
    display_metrics(dos_metrics)

    if dos_metrics:
        best_epoch = dos_metrics.get("best_epoch")
        threshold = dos_metrics.get("threshold")
        percentile = dos_metrics.get("threshold_percentile")
        c1, c2, c3 = st.columns(3)
        c1.metric("Best epoch", format_number(best_epoch))
        c2.metric(
            "Calibration percentile",
            f"{percentile:g}th" if isinstance(percentile, (int, float)) else "N/A",
        )
        c3.metric(
            "Saved threshold",
            f"{float(threshold):.6f}" if isinstance(threshold, (int, float)) else "N/A",
        )

    st.subheader("Model artifacts")
    display_artifact_status(
        [
            ("GNN checkpoint", DOS_HULK_MODEL_DIR / "dos_hulk_gnn_best.pth"),
            ("Flow imputer", DOS_HULK_MODEL_DIR / "dos_hulk_imputer.pkl"),
            ("Flow scaler", DOS_HULK_MODEL_DIR / "dos_hulk_scaler.pkl"),
            ("Temporal Isolation Forest", DOS_HULK_MODEL_DIR / "dos_hulk_temporal_iforest.pkl"),
            ("Temporal scaler", DOS_HULK_MODEL_DIR / "dos_hulk_temporal_scaler.pkl"),
        ]
    )

    with st.expander("Threshold calibration details"):
        if dos_calibration:
            st.json(dos_calibration)
        else:
            st.info("DoS Hulk calibration JSON was not found.")

    st.divider()
    st.subheader("Confusion matrix")
    plot_confusion(
        dos_predictions,
        "Actual_Label",
        "Predicted_Label",
        "DoS Hulk Confusion Matrix",
    )

    st.subheader("ROC and precision-recall curves")
    dos_score_column = find_column(
        dos_predictions,
        ["Combined_Anomaly_Score"],
    )
    if dos_score_column:
        plot_roc_pr(
            dos_predictions,
            "Actual_Label",
            dos_score_column,
            "DoS Hulk",
        )
        st.subheader("Combined anomaly score distribution")
        plot_score_distribution(
            dos_predictions,
            "Actual_Label",
            dos_score_column,
            "DoS Hulk",
            "Combined anomaly score",
        )
    else:
        st.info("DoS Hulk combined anomaly scores are unavailable.")

    st.subheader("Training history")
    plot_training_history(
        dos_history,
        "DoS Hulk",
        train_col="training_nll",
        validation_col="validation_nll",
    )

    st.subheader("Threshold sensitivity")
    show_threshold_tradeoff(
        dos_metrics,
        "dos_hulk_detection_rate",
        "DoS Hulk",
    )

    st.divider()
    st.subheader("Component weights")
    weights = dos_metrics.get("weights", {})
    if isinstance(weights, dict) and weights:
        weight_frame = pd.DataFrame(
            [
                {"Component": "GNN", "Weight": weights.get("gnn")},
                {"Component": "Temporal Isolation Forest", "Weight": weights.get("temporal_iforest")},
                {"Component": "Behaviour rule", "Weight": weights.get("behaviour_rule")},
            ]
        )
        st.dataframe(weight_frame, width="stretch", hide_index=True)
        st.caption(
            "The saved DoS Hulk configuration assigns zero weight to the "
            "scan-oriented behaviour rule."
        )
    else:
        st.info("Component weights are not recorded in the metrics file.")

    st.subheader("Prediction distribution")
    show_distribution(
        dos_predictions,
        "Predicted_Label",
        "DoS Hulk prediction distribution",
    )

    st.subheader("Saved flow predictions")
    show_filtered_table(
        dos_predictions,
        "dos_hulk_predictions_filtered.csv",
        "dos_hulk",
    )

    st.divider()
    st.subheader("Diagnostic reports")
    show_diagnostic_csvs(
        DOS_HULK_DIAGNOSTICS_DIR,
        [
            ("Component ablation", "component_ablation.csv"),
            ("Component group summary", "component_group_summary.csv"),
            ("False positives by hour", "false_positives_by_hour.csv"),
            ("False-positive feature profile", "false_positive_feature_profile.csv"),
            ("Destination-port false-positive summary", "destination_port_false_positive_summary.csv"),
            ("Source-IP false-positive summary", "source_ip_false_positive_summary.csv"),
            ("Feature drift PSI / KS", "feature_drift_psi_ks.csv"),
            ("Categorical traffic drift", "categorical_traffic_drift.csv"),
        ],
    )

    diagnostic_summary_path = DOS_HULK_DIAGNOSTICS_DIR / "diagnostic_summary.json"
    diagnostic_summary = json_at(diagnostic_summary_path)
    if diagnostic_summary:
        with st.expander("View diagnostic summary JSON"):
            st.json(diagnostic_summary)

    st.subheader("Ensemble optimisation reports")
    show_diagnostic_csvs(
        DOS_HULK_OPTIMIZATION_DIR,
        [
            ("Ensemble comparison", "ensemble_comparison.csv"),
            ("Ensemble predictions", "ensemble_predictions.csv"),
        ],
    )


# ============================================================
# 9. HYBRID GNN PAGE
# ============================================================

elif page == "Hybrid GNN":
    st.header("Hybrid GNN — Unseen Attack-Category Detection")
    st.caption(
        "Supervised GNN trained on benign traffic and selected known attack "
        "categories, with PortScan excluded from training and validation."
    )

    st.info(
        "This is a complementary unseen-category generalisation experiment. "
        "It has a different training procedure from the benign-only PortScan "
        "V6 and DoS Hulk experiments."
    )

    st.subheader("Saved evaluation metrics")
    display_metrics(hybrid_metrics)

    st.subheader("Model artifacts")
    display_artifact_status(
        [
            ("Hybrid GNN checkpoint", HYBRID_MODEL_DIR / "hybrid_gnn_best.pth"),
            ("Shared PortScan V6 flow imputer", PORTSCAN_MODEL_DIR / "zero_day_imputer.pkl"),
            ("Shared PortScan V6 flow scaler", PORTSCAN_MODEL_DIR / "zero_day_scaler.pkl"),
        ]
    )

    st.divider()
    st.subheader("Confusion matrix")
    plot_confusion(
        hybrid_predictions,
        "Actual_Label",
        "Predicted_Label",
        "Hybrid GNN Confusion Matrix",
    )

    st.subheader("ROC and precision-recall curves")
    hybrid_score_column = find_column(
        hybrid_predictions,
        ["Attack_Probability"],
    )
    if hybrid_score_column:
        plot_roc_pr(
            hybrid_predictions,
            "Actual_Label",
            hybrid_score_column,
            "Hybrid GNN",
        )
        st.subheader("Attack probability distribution")
        plot_score_distribution(
            hybrid_predictions,
            "Actual_Label",
            hybrid_score_column,
            "Hybrid GNN",
            "Attack probability",
        )
    else:
        st.info("Hybrid GNN attack probabilities are unavailable.")

    st.subheader("Training loss")
    plot_training_history(
        hybrid_history,
        "Hybrid GNN",
        train_col="training_loss",
        validation_col=None,
    )

    if "validation_balanced_accuracy" in hybrid_history.columns:
        validation = hybrid_history[
            ["epoch", "validation_balanced_accuracy"]
        ].copy()
        validation["epoch"] = pd.to_numeric(validation["epoch"], errors="coerce")
        validation["validation_balanced_accuracy"] = pd.to_numeric(
            validation["validation_balanced_accuracy"],
            errors="coerce",
        )
        validation = validation.dropna()
        st.subheader("Validation balanced accuracy")
        if not validation.empty:
            st.line_chart(
                validation.set_index("epoch")[
                    ["validation_balanced_accuracy"]
                ]
            )

    st.subheader("Prediction distribution")
    show_distribution(
        hybrid_predictions,
        "Predicted_Label",
        "Hybrid GNN prediction distribution",
    )

    st.subheader("Saved flow predictions")
    show_filtered_table(
        hybrid_predictions,
        "hybrid_gnn_predictions_filtered.csv",
        "hybrid_gnn",
    )

    with st.expander("Training and evaluation protocol"):
        st.write(
            "Training attack categories are selected known categories other "
            "than PortScan. PortScan is explicitly excluded from the training "
            "and validation attack subsets. The final test combines benign "
            "test traffic with held-out PortScan traffic."
        )
        if hybrid_metrics:
            st.json(
                {
                    "training_attack_categories": hybrid_metrics.get(
                        "training_attack_categories"
                    ),
                    "excluded_attack_category": hybrid_metrics.get(
                        "excluded_attack_category"
                    ),
                    "decision_threshold": hybrid_metrics.get(
                        "decision_threshold"
                    ),
                    "best_epoch": hybrid_metrics.get("best_epoch"),
                }
            )


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":
    st.header("Model Comparison")
    st.caption(
        "Comparison of saved metrics. Interpret results alongside each "
        "experiment's training setup and evaluation population."
    )

    metric_definitions = [
        ("Accuracy", ["accuracy"]),
        ("Precision", ["precision"]),
        ("Recall", ["recall"]),
        ("F1 Score", ["f1_score", "f1"]),
        ("False Positive Rate", ["false_positive_rate", "fpr"]),
        ("ROC-AUC", ["roc_auc"]),
        ("PR-AUC", ["pr_auc", "average_precision"]),
        ("Balanced Accuracy", ["balanced_accuracy"]),
    ]

    model_metrics = {
        "PortScan V6": portscan_metrics,
        "DoS Hulk": dos_metrics,
        "Hybrid GNN": hybrid_metrics,
    }

    display_rows = []
    numeric_rows = []

    for metric_name, aliases in metric_definitions:
        display_row = {"Metric": metric_name}
        numeric_row = {"Metric": metric_name}
        for model_name, metrics in model_metrics.items():
            value = get_metric(metrics, aliases)
            display_row[model_name] = format_percent(value)
            numeric_row[model_name] = to_percent(value)
        display_rows.append(display_row)
        numeric_rows.append(numeric_row)

    st.dataframe(
        pd.DataFrame(display_rows),
        width="stretch",
        hide_index=True,
    )

    chart = pd.DataFrame(numeric_rows).set_index("Metric")
    chart = chart.dropna(how="all")
    if not chart.empty:
        st.subheader("Metric comparison chart")
        st.bar_chart(chart)

    st.divider()
    st.subheader("Experimental differences")
    differences = pd.DataFrame(
        [
            [
                "PortScan V6",
                "Benign-only",
                "PortScan",
                "Attack-specific zero-day evaluation",
            ],
            [
                "DoS Hulk",
                "Benign-only",
                "DoS Hulk",
                "Attack-specific zero-day evaluation",
            ],
            [
                "Hybrid GNN",
                "Benign + selected known attacks",
                "PortScan",
                "Unseen attack-category generalisation",
            ],
        ],
        columns=[
            "Experiment",
            "Training data",
            "Held-out test attack",
            "Purpose",
        ],
    )
    st.dataframe(differences, width="stretch", hide_index=True)

    st.warning(
        "These experiments do not all use the same training procedure. "
        "Treat the table as a comparison of complementary experiments, "
        "not as a controlled ranking of interchangeable models."
    )


# ============================================================
# 11. FLOW EXPLORER
# ============================================================

elif page == "Flow Explorer":
    st.header("Flow Prediction Explorer")
    st.caption(
        "Explore saved flow-level predictions from the completed experiments."
    )

    experiment = st.selectbox(
        "Choose experiment",
        ["PortScan V6", "DoS Hulk", "Hybrid GNN"],
    )

    selected_data = {
        "PortScan V6": portscan_predictions,
        "DoS Hulk": dos_predictions,
        "Hybrid GNN": hybrid_predictions,
    }[experiment]

    if selected_data.empty:
        st.warning(
            f"No saved prediction CSV was found for {experiment}."
        )
    else:
        st.write(f"Saved prediction rows: {len(selected_data):,}")

        actual_col = find_column(
            selected_data,
            ["Actual_Label", "actual_label", "true_label"],
        )
        predicted_col = find_column(
            selected_data,
            ["Predicted_Label", "prediction", "predicted_class"],
        )

        score_col = find_column(
            selected_data,
            [
                "Combined_V6_Score",
                "Combined_Anomaly_Score",
                "Attack_Probability",
            ],
        )

        first, second, third = st.columns(3)
        if actual_col:
            first.metric(
                "Actual benign rows",
                format_number((binary_labels(selected_data[actual_col]) == 0).sum()),
            )
            second.metric(
                "Actual attack rows",
                format_number((binary_labels(selected_data[actual_col]) == 1).sum()),
            )
        if predicted_col:
            third.metric(
                "Predicted attack rows",
                format_number(
                    (binary_labels(selected_data[predicted_col]) == 1).sum()
                ),
            )

        filtered = selected_data.copy()
        search = st.text_input(
            "Search IP, flow ID, label, timestamp or score",
            key="explorer_search",
        )
        if search:
            mask = filtered.astype(str).apply(
                lambda column: column.str.contains(
                    search,
                    case=False,
                    na=False,
                    regex=False,
                )
            ).any(axis=1)
            filtered = filtered[mask]

        if actual_col:
            actual_values = sorted(
                filtered[actual_col].dropna().astype(str).unique().tolist()
            )
            chosen_actual = st.multiselect(
                "Filter actual labels",
                actual_values,
                default=actual_values,
                key="explorer_actual",
            )
            filtered = filtered[
                filtered[actual_col].astype(str).isin(chosen_actual)
            ]

        if predicted_col:
            predicted_values = sorted(
                filtered[predicted_col].dropna().astype(str).unique().tolist()
            )
            chosen_predicted = st.multiselect(
                "Filter predicted labels",
                predicted_values,
                default=predicted_values,
                key="explorer_predicted",
            )
            filtered = filtered[
                filtered[predicted_col].astype(str).isin(chosen_predicted)
            ]

        st.write(f"Matching rows: {len(filtered):,}")
        st.dataframe(filtered, width="stretch", height=480)
        st.download_button(
            "Download filtered prediction records",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"{experiment.lower().replace(' ', '_')}_filtered_predictions.csv",
            mime="text/csv",
            key="explorer_download",
        )

        if score_col:
            st.subheader("Score summary")
            numeric_scores = pd.to_numeric(
                filtered[score_col],
                errors="coerce",
            ).dropna()
            if not numeric_scores.empty:
                st.dataframe(
                    numeric_scores.describe().to_frame("Score").T,
                    width="stretch",
                )


# ============================================================
# 12. DATASET INSPECTION
# ============================================================

elif page == "Dataset Inspection":
    st.header("Dataset Inspection")
    folders = [
        DATA_PROCESSED_DIR,
        DATA_ZERO_DAY_DIR,
        DATA_REMAINING_DIR,
        DOS_HULK_DIR / "data" / "prepared",
        BASE_DIR / "Dataset" / "TrafficLabelling",
    ]

    csv_files = []
    seen = set()

    for folder in folders:
        if folder.exists():
            for path in folder.rglob("*.csv"):
                resolved = str(path.resolve())
                if resolved not in seen:
                    csv_files.append(path)
                    seen.add(resolved)

    csv_files = sorted(csv_files, key=lambda path: str(path).lower())

    if not csv_files:
        st.warning("No CSV datasets were found in the configured folders.")
    else:
        st.success(f"CSV files found: {len(csv_files)}")

        info = pd.DataFrame(
            [
                {
                    "File": path.name,
                    "Location": str(path.relative_to(BASE_DIR)),
                    "Size (MB)": round(
                        path.stat().st_size / (1024 * 1024),
                        2,
                    ),
                }
                for path in csv_files
            ]
        )
        st.dataframe(
            info,
            width="stretch",
            hide_index=True,
            height=350,
        )

        selected_file = st.selectbox(
            "Select a dataset",
            csv_files,
            format_func=lambda path: str(path.relative_to(BASE_DIR)),
        )
        preview_rows = st.number_input(
            "Rows to preview",
            min_value=10,
            max_value=10000,
            value=100,
            step=100,
        )

        if st.button("Load dataset preview"):
            try:
                with st.spinner("Loading dataset preview..."):
                    preview = pd.read_csv(
                        selected_file,
                        nrows=int(preview_rows),
                        low_memory=False,
                    )

                first, second = st.columns(2)
                first.metric("Preview rows", f"{len(preview):,}")
                second.metric("Columns", f"{len(preview.columns):,}")

                st.subheader("Dataset preview")
                st.dataframe(
                    preview,
                    width="stretch",
                    height=450,
                )

                st.subheader("Column names")
                st.write(preview.columns.tolist())

                label_col = find_column(
                    preview,
                    ["Label", "Actual_Label", "attack_type"],
                )
                if label_col:
                    st.subheader("Label distribution")
                    st.bar_chart(
                        preview[label_col].astype(str).value_counts()
                    )

                st.subheader("Missing values in preview")
                missing = preview.isnull().sum()
                missing = missing[missing > 0]
                if missing.empty:
                    st.success("No missing values in this preview.")
                else:
                    st.dataframe(
                        missing.rename("Missing Values")
                        .reset_index()
                        .rename(columns={"index": "Column"}),
                        width="stretch",
                        hide_index=True,
                    )
            except Exception as error:
                st.error(f"Could not load dataset: {error}")


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()
st.caption(
    "GNN-Based Intrusion Detection | CIC-IDS2017 | "
    "PortScan V6 and DoS Hulk benign-only zero-day experiments | "
    "Hybrid GNN unseen-category experiment"
)
