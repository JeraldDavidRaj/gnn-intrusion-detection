from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    roc_auc_score, precision_recall_curve, average_precision_score
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PORTSCAN_DIR = BASE_DIR / "reports" / "zero_day" / "portscan_v6"
PORTSCAN_EVAL_DIR = PORTSCAN_DIR / "evaluation"
PORTSCAN_PUBLICATION_DIR = PORTSCAN_DIR / "publication"
HYBRID_DIR = BASE_DIR / "reports" / "hybrid"

st.set_page_config(
    page_title="GNN Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("GNN-Based Intrusion Detection System")
st.caption("CIC-IDS2017 | Graph Neural Networks | Strict Zero-Day and Hybrid Intrusion Detection")


# ------------------------- DATA LOADING -------------------------

@st.cache_data(show_spinner=False)
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(path_string):
    path = Path(path_string)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def flatten_dict(data, prefix=""):
    result = {}
    if not isinstance(data, dict):
        return result
    for key, value in data.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_dict(value, name))
        elif isinstance(value, (int, float, str, bool)):
            result[name] = value
    return result


def norm(value):
    return str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "").replace("%", "percent")


def find_column(df, candidates):
    if df.empty:
        return None
    lookup = {norm(c): c for c in df.columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    return None


def get_metric(metrics, aliases):
    lookup = {norm(k): v for k, v in metrics.items()}
    for alias in aliases:
        value = lookup.get(norm(alias))
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def as_percent(value):
    if value is None or not np.isfinite(value):
        return None
    return value * 100 if 0 <= value <= 1 else value


def fmt_percent(value):
    value = as_percent(value)
    return "N/A" if value is None else f"{value:.3f}%"


def show_metric_cards(metrics):
    definitions = [
        ("Accuracy", ["accuracy", "acc"]),
        ("Precision", ["precision"]),
        ("Recall", ["recall", "sensitivity", "tpr"]),
        ("F1 Score", ["f1", "f1_score"]),
        ("FPR", ["fpr", "false_positive_rate"]),
        ("ROC-AUC", ["roc_auc", "rocauc"]),
        ("PR-AUC", ["pr_auc", "prauc", "average_precision"]),
        ("Balanced Accuracy", ["balanced_accuracy"]),
    ]
    found = [(label, get_metric(metrics, aliases)) for label, aliases in definitions]
    found = [(label, value) for label, value in found if value is not None]
    if not found:
        st.warning("No recognized metrics found in the evaluation JSON.")
        return
    for start in range(0, len(found), 4):
        group = found[start:start + 4]
        cols = st.columns(len(group))
        for col, (label, value) in zip(cols, group):
            col.metric(label, fmt_percent(value))


def display_metrics(metrics):
    if not metrics:
        st.warning("Evaluation metrics JSON was not found or could not be read.")
        return
    show_metric_cards(metrics)
    with st.expander("View raw evaluation metrics"):
        st.json(metrics)


# ------------------------- EVALUATION HELPERS -------------------------

NORMAL_LABELS = {
    "BENIGN", "NORMAL", "0", "FALSE", "NO ATTACK", "NON-ATTACK", "NONATTACK"
}


def binary_labels(series):
    values = series.astype(str).str.strip().str.upper()
    return values.apply(lambda value: 0 if value in NORMAL_LABELS else 1)


def valid_frame(df, actual_col, predicted_col=None, score_col=None):
    required = [actual_col]
    if predicted_col:
        required.append(predicted_col)
    if score_col:
        required.append(score_col)
    missing = [col for col in required if col not in df.columns]
    if missing:
        return pd.DataFrame(), missing
    data = df[required].copy()
    if score_col:
        data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data = data.dropna(subset=required)
    return data, []


def plot_confusion(df, actual_col, predicted_col, title):
    data, missing = valid_frame(df, actual_col, predicted_col=predicted_col)
    if missing:
        st.warning(f"Missing columns for confusion matrix: {', '.join(missing)}")
        return
    if data.empty:
        st.info("No valid rows for confusion matrix.")
        return
    y_true = binary_labels(data[actual_col])
    y_pred = binary_labels(data[predicted_col])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Attack"]
    ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    tn, fp, fn, tp = cm.ravel()
    cols = st.columns(4)
    cols[0].metric("True Negatives", f"{tn:,}")
    cols[1].metric("False Positives", f"{fp:,}")
    cols[2].metric("False Negatives", f"{fn:,}")
    cols[3].metric("True Positives", f"{tp:,}")


def plot_roc_pr(df, actual_col, score_col, title):
    data, missing = valid_frame(df, actual_col, score_col=score_col)
    if missing:
        st.warning(f"Missing columns for curves: {', '.join(missing)}")
        return
    if data.empty:
        st.info("No valid rows for ROC/PR curves.")
        return
    y_true = binary_labels(data[actual_col])
    scores = data[score_col].astype(float)
    if y_true.nunique() != 2 or scores.nunique() < 2 or not np.isfinite(scores).all():
        st.warning("ROC/PR curves need both classes and continuous score variation.")
        return

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_value = roc_auc_score(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap_value = average_precision_score(y_true, scores)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
        ax.plot([0, 1], [0, 1], "--", label="Random baseline")
        ax.set(title=f"{title} - ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(recall, precision, label=f"Average Precision = {ap_value:.4f}")
        ax.set(title=f"{title} - Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        f"Recalculated from {len(data):,} rows. ROC-AUC={auc_value:.5f}; "
        f"Average Precision={ap_value:.5f}. Values can differ from saved "
        "metrics if the source split or evaluation population differs."
    )


def plot_score_distribution(df, score_col, actual_col, title, score_label):
    data, missing = valid_frame(df, actual_col, score_col=score_col)
    if missing:
        st.warning(f"Missing score columns: {', '.join(missing)}")
        return
    if data.empty:
        st.info("No score values available.")
        return
    data["Class"] = binary_labels(data[actual_col]).map({0: "Benign", 1: "Attack"})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label in ["Benign", "Attack"]:
        values = data.loc[data["Class"] == label, score_col]
        if not values.empty:
            ax.hist(values, bins=50, density=True, alpha=0.55, label=label)
    ax.set(title=f"{title} - {score_label} Distribution", xlabel=score_label, ylabel="Density")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_training(df, title, epoch_col, train_col, validation_col=None):
    if df.empty:
        st.info(f"{title} training history is unavailable.")
        return
    required = [epoch_col, train_col] + ([validation_col] if validation_col else [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.warning(f"Missing training-history columns: {', '.join(missing)}")
        return
    data = df[required].copy()
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[epoch_col, train_col]).sort_values(epoch_col)
    if data.empty:
        st.info("No valid training-history rows.")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data[epoch_col], data[train_col], marker="o", label=train_col)
    if validation_col:
        valid = data.dropna(subset=[validation_col])
        if not valid.empty:
            ax.plot(valid[epoch_col], valid[validation_col], marker="o", label=validation_col)
    ax.set(title=f"{title} - Training History", xlabel="Epoch", ylabel="Recorded value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_distribution(df, column, title):
    if df.empty or column not in df.columns:
        st.info(f"{title} is unavailable.")
        return
    st.bar_chart(df[column].astype(str).value_counts())


def show_filtered_table(df, filename, key):
    if df.empty:
        st.info("No prediction records available.")
        return
    search = st.text_input("Search records", key=f"{key}_search")
    filtered = df.copy()
    if search:
        mask = filtered.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False, regex=False)
        ).any(axis=1)
        filtered = filtered[mask]
    st.write(f"Matching records: {len(filtered):,}")
    st.dataframe(filtered, use_container_width=True, height=420)
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"{key}_download"
    )


# ------------------------- LOAD REPORTS -------------------------

portscan_metrics = load_json(PORTSCAN_EVAL_DIR / "zero_day_v6_evaluation_metrics.json")
portscan_calibration = load_json(PORTSCAN_EVAL_DIR / "zero_day_v6_calibration.json")
portscan_history = load_csv(PORTSCAN_EVAL_DIR / "zero_day_v6_training_history.csv")
portscan_eval_predictions = load_csv(PORTSCAN_EVAL_DIR / "zero_day_v6_flow_predictions.csv")
portscan_publication = load_csv(PORTSCAN_PUBLICATION_DIR / "all_test_flow_predictions.csv")
portscan_detected = load_csv(PORTSCAN_PUBLICATION_DIR / "combined_v6_detected_portscan.csv")
portscan_missed = load_csv(PORTSCAN_PUBLICATION_DIR / "combined_v6_missed_portscan.csv")
portscan_fp = load_csv(PORTSCAN_PUBLICATION_DIR / "combined_v6_false_positives.csv")
portscan_ablation = load_csv(PORTSCAN_PUBLICATION_DIR / "publication_ablation_results.csv")

hybrid_metrics = load_json(HYBRID_DIR / "hybrid_gnn_evaluation_metrics.json")
hybrid_predictions = load_csv(HYBRID_DIR / "hybrid_gnn_flow_predictions.csv")
hybrid_history = load_csv(HYBRID_DIR / "hybrid_gnn_training_history.csv")


# ------------------------- NAVIGATION -------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Project Overview",
        "PortScan V6 - Zero-Day",
        "Hybrid GNN",
        "Model Comparison",
        "Flow Explorer",
        "Dataset Inspection",
    ],
)


# =========================================================
# PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":
    st.header("Project Overview")
    st.markdown(
        """
        This project applies Graph Neural Networks to CIC-IDS2017 network
        traffic for intrusion detection.

        **PortScan V6** is the primary strict zero-day experiment: the model
        is trained using benign traffic and evaluated on held-out PortScan
        attack traffic.

        **Hybrid GNN** is a separate hybrid experiment trained using selected
        known attack categories and evaluated on held-out PortScan traffic.
        These are different training setups and must not be presented as
        equivalent zero-day experiments.
        """
    )
    st.divider()
    a, b = st.columns(2)
    with a:
        st.subheader("PortScan V6")
        st.success("Primary strict zero-day experiment")
        st.write("Training: benign-only")
        st.write("Evaluation: held-out PortScan")
        st.metric("F1 Score", "92.742%")
        st.code("models/portscan_v6/zero_day_gnn_v6_best.pth")
    with b:
        st.subheader("Hybrid GNN")
        st.info("Separate hybrid detection experiment")
        st.write("Training: selected known attack categories")
        st.write("Evaluation: held-out PortScan")
        st.metric("F1 Score", "90.356%")
        st.code("models/hybrid/hybrid_gnn_best.pth")

    st.divider()
    st.subheader("Key Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PortScan V6 F1", "92.742%")
    c2.metric("Hybrid GNN F1", "90.356%")
    c3.metric("PortScan V6 FPR", "0.056%")
    c4.metric("Hybrid GNN Recall", "99.976%")

    status = pd.DataFrame([
        ["PortScan V6", "Metrics JSON", "Available" if portscan_metrics else "Missing"],
        ["PortScan V6", "Publication predictions", "Available" if not portscan_publication.empty else "Missing"],
        ["PortScan V6", "Training history", "Available" if not portscan_history.empty else "Missing"],
        ["Hybrid GNN", "Metrics JSON", "Available" if hybrid_metrics else "Missing"],
        ["Hybrid GNN", "Predictions", "Available" if not hybrid_predictions.empty else "Missing"],
        ["Hybrid GNN", "Training history", "Available" if not hybrid_history.empty else "Missing"],
    ], columns=["Experiment", "Artifact", "Status"])
    st.subheader("Report Availability")
    st.dataframe(status, use_container_width=True, hide_index=True)


# =========================================================
# PORTSCAN V6
# =========================================================

elif page == "PortScan V6 - Zero-Day":
    st.header("PortScan V6 - Strict Zero-Day Detection")
    st.caption("Benign-only training and held-out PortScan attack evaluation.")

    st.subheader("Saved Evaluation Metrics")
    display_metrics(portscan_metrics)

    model_path = BASE_DIR / "models" / "portscan_v6" / "zero_day_gnn_v6_best.pth"
    if model_path.exists():
        st.success("PortScan V6 model artifact found.")
    else:
        st.warning("PortScan V6 model artifact was not found.")
    st.code("models/portscan_v6/zero_day_gnn_v6_best.pth")

    st.subheader("Threshold Calibration")
    if portscan_calibration:
        with st.expander("View calibration details"):
            st.json(portscan_calibration)
    else:
        st.info("Calibration report unavailable.")

    pvisual = portscan_publication if not portscan_publication.empty else portscan_eval_predictions
    score_col = "Combined_V6_Score" if "Combined_V6_Score" in pvisual.columns else "Combined_Anomaly_Score"

    st.divider()
    st.header("Advanced Evaluation")
    st.subheader("Confusion Matrix")
    plot_confusion(pvisual, "Actual_Label", "Predicted_Label", "PortScan V6 Confusion Matrix")

    st.subheader("ROC and Precision-Recall Curves")
    plot_roc_pr(pvisual, "Actual_Label", score_col, "PortScan V6")

    st.subheader("Anomaly Score Distribution")
    plot_score_distribution(pvisual, score_col, "Actual_Label", "PortScan V6", "Combined V6 Anomaly Score")

    st.subheader("Training and Validation NLL")
    plot_training(portscan_history, "PortScan V6", "epoch", "training_nll", "validation_nll")

    st.subheader("Prediction Distribution")
    show_distribution(pvisual, "Predicted_Label", "PortScan prediction distribution")

    st.divider()
    st.subheader("Prediction Records")
    show_filtered_table(pvisual, "portscan_v6_predictions_filtered.csv", "portscan")

    st.divider()
    st.subheader("Detected, Missed and False-Positive Flows")
    tab1, tab2, tab3 = st.tabs(["Detected PortScan", "Missed PortScan", "False Positives"])
    with tab1:
        st.write(f"Records: {len(portscan_detected):,}")
        st.dataframe(portscan_detected, use_container_width=True)
    with tab2:
        st.write(f"Records: {len(portscan_missed):,}")
        st.dataframe(portscan_missed, use_container_width=True)
    with tab3:
        st.write(f"Records: {len(portscan_fp):,}")
        st.dataframe(portscan_fp, use_container_width=True)

    if not portscan_ablation.empty:
        st.subheader("Publication Ablation Results")
        st.dataframe(portscan_ablation, use_container_width=True)


# =========================================================
# HYBRID GNN
# =========================================================

elif page == "Hybrid GNN":
    st.header("Hybrid GNN - Hybrid Detection")
    st.caption(
        "Separate experiment trained using selected known attack categories "
        "and evaluated on held-out PortScan traffic."
    )

    st.subheader("Saved Evaluation Metrics")
    display_metrics(hybrid_metrics)

    model_path = BASE_DIR / "models" / "hybrid" / "hybrid_gnn_best.pth"
    if model_path.exists():
        st.success("Hybrid GNN model artifact found.")
    else:
        st.warning("Hybrid GNN model artifact was not found.")
    st.code("models/hybrid/hybrid_gnn_best.pth")

    st.divider()
    st.header("Advanced Evaluation")
    st.subheader("Confusion Matrix")
    plot_confusion(hybrid_predictions, "Actual_Label", "Predicted_Label", "Hybrid GNN Confusion Matrix")

    st.subheader("ROC and Precision-Recall Curves")
    plot_roc_pr(hybrid_predictions, "Actual_Label", "Attack_Probability", "Hybrid GNN")

    st.subheader("Attack Probability Distribution")
    plot_score_distribution(hybrid_predictions, "Attack_Probability", "Actual_Label", "Hybrid GNN", "Attack Probability")

    st.subheader("Training Loss")
    plot_training(hybrid_history, "Hybrid GNN", "epoch", "training_loss")

    st.subheader("Validation Balanced Accuracy")
    if {"epoch", "validation_balanced_accuracy"}.issubset(hybrid_history.columns):
        validation = hybrid_history[["epoch", "validation_balanced_accuracy"]].copy()
        validation["epoch"] = pd.to_numeric(validation["epoch"], errors="coerce")
        validation["validation_balanced_accuracy"] = pd.to_numeric(validation["validation_balanced_accuracy"], errors="coerce")
        validation = validation.dropna()
        if not validation.empty:
            st.line_chart(validation.set_index("epoch"))
        else:
            st.info("No valid validation values.")
    else:
        st.info("Validation balanced accuracy history unavailable.")

    st.subheader("Prediction Distribution")
    show_distribution(hybrid_predictions, "Predicted_Label", "Hybrid prediction distribution")

    st.divider()
    st.subheader("Prediction Records")
    show_filtered_table(hybrid_predictions, "hybrid_gnn_predictions_filtered.csv", "hybrid")


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":
    st.header("Model Comparison")
    st.caption("Saved metrics from two experiments with different training setups.")

    pmetrics = flatten_dict(portscan_metrics)
    hmetrics = flatten_dict(hybrid_metrics)
    definitions = [
        ("Accuracy", ["accuracy", "acc"]),
        ("Precision", ["precision"]),
        ("Recall", ["recall", "sensitivity"]),
        ("F1 Score", ["f1", "f1_score"]),
        ("False Positive Rate", ["fpr", "false_positive_rate"]),
        ("ROC-AUC", ["roc_auc", "rocauc"]),
        ("PR-AUC", ["pr_auc", "prauc", "average_precision"]),
        ("Balanced Accuracy", ["balanced_accuracy"]),
    ]

    rows = []
    numeric_rows = []
    for label, aliases in definitions:
        pv = get_metric(pmetrics, aliases)
        hv = get_metric(hmetrics, aliases)
        rows.append({
            "Metric": label,
            "PortScan V6": fmt_percent(pv),
            "Hybrid GNN": fmt_percent(hv),
        })
        numeric_rows.append({
            "Metric": label,
            "PortScan V6": as_percent(pv),
            "Hybrid GNN": as_percent(hv),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Metric Comparison Chart")
    chart = pd.DataFrame(numeric_rows).set_index("Metric").dropna(how="all")
    if not chart.empty:
        st.bar_chart(chart)

    st.divider()
    st.subheader("Experiment Differences")
    differences = pd.DataFrame({
        "Aspect": ["Primary purpose", "Training data", "Evaluation attack", "Experiment type"],
        "PortScan V6": ["Strict zero-day detection", "Benign-only", "PortScan", "Zero-day experiment"],
        "Hybrid GNN": ["Hybrid detection", "Selected known attack categories", "PortScan", "Separate hybrid experiment"],
    })
    st.dataframe(differences, use_container_width=True, hide_index=True)
    st.info(
        "Do not interpret the comparison as a direct model ranking without "
        "considering the different training procedures and evaluation populations."
    )


# =========================================================
# FLOW EXPLORER
# =========================================================

elif page == "Flow Explorer":
    st.header("Network Flow Explorer")
    experiment = st.selectbox("Choose experiment", ["PortScan V6", "Hybrid GNN"])

    if experiment == "PortScan V6":
        flows = portscan_publication if not portscan_publication.empty else portscan_eval_predictions
    else:
        flows = hybrid_predictions

    if flows.empty:
        st.warning("No prediction records found for this experiment.")
    else:
        st.write(f"Total flow records: {len(flows):,}")
        pred_col = find_column(flows, ["Predicted_Label", "prediction", "predicted_class"])
        actual_col = find_column(flows, ["Actual_Label", "actual_label", "true_label"])

        left, right = st.columns(2)
        if pred_col:
            with left:
                st.subheader("Predicted Class Distribution")
                show_distribution(flows, pred_col, "Predicted class distribution")
        if actual_col:
            with right:
                st.subheader("Actual Class Distribution")
                st.bar_chart(flows[actual_col].astype(str).value_counts())

        st.divider()
        st.subheader("Filter Flow Records")
        filtered = flows.copy()
        search = st.text_input("Search IP address, flow ID, label, timestamp or score", key="flow_search")
        if search:
            mask = filtered.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False, regex=False)
            ).any(axis=1)
            filtered = filtered[mask]

        if actual_col:
            labels = filtered[actual_col].dropna().astype(str).unique().tolist()
            selected = st.multiselect("Filter by actual label", labels, default=labels)
            filtered = filtered[filtered[actual_col].astype(str).isin(selected)]

        if pred_col:
            labels = filtered[pred_col].dropna().astype(str).unique().tolist()
            selected = st.multiselect("Filter by predicted label", labels, default=labels)
            filtered = filtered[filtered[pred_col].astype(str).isin(selected)]

        st.write(f"Matching records: {len(filtered):,}")
        st.dataframe(filtered, use_container_width=True, height=450)
        st.download_button(
            "Download filtered flow records",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="flow_records_filtered.csv",
            mime="text/csv",
        )


# =========================================================
# DATASET INSPECTION
# =========================================================

elif page == "Dataset Inspection":
    st.header("Dataset Inspection")
    folders = [
        DATA_DIR / "processed",
        DATA_DIR / "zero_day",
        DATA_DIR / "remaining_attacks",
        BASE_DIR / "Dataset" / "TrafficLabelling",
    ]

    files = []
    seen = set()
    for folder in folders:
        if folder.exists():
            for path in folder.glob("*.csv"):
                resolved = str(path.resolve())
                if resolved not in seen:
                    files.append(path)
                    seen.add(resolved)

    files = sorted(files, key=lambda path: str(path).lower())

    if not files:
        st.warning("No CSV datasets found in configured folders.")
    else:
        st.success(f"CSV files found: {len(files)}")
        info = pd.DataFrame([
            {
                "File": path.name,
                "Location": str(path.relative_to(BASE_DIR)),
                "Size (MB)": round(path.stat().st_size / (1024 * 1024), 2),
            }
            for path in files
        ])
        st.dataframe(info, use_container_width=True, hide_index=True, height=350)

        selected_file = st.selectbox("Select a dataset", files, format_func=lambda path: path.name)
        rows = st.number_input("Number of rows to preview", min_value=10, max_value=10000, value=100, step=100)

        if st.button("Load Dataset Preview"):
            try:
                with st.spinner("Loading dataset preview..."):
                    preview = pd.read_csv(selected_file, nrows=int(rows), low_memory=False)

                c1, c2 = st.columns(2)
                c1.metric("Preview Rows", f"{len(preview):,}")
                c2.metric("Number of Columns", f"{len(preview.columns):,}")

                st.subheader("Dataset Preview")
                st.dataframe(preview, use_container_width=True, height=450)
                st.subheader("Column Names")
                st.write(preview.columns.tolist())

                label_col = find_column(preview, ["Label", "Actual_Label", "attack_type"])
                if label_col:
                    st.subheader("Label Distribution")
                    st.bar_chart(preview[label_col].astype(str).value_counts())

                st.subheader("Missing Values in Preview")
                missing = preview.isnull().sum()
                missing = missing[missing > 0]
                if missing.empty:
                    st.success("No missing values in this preview.")
                else:
                    st.dataframe(
                        missing.rename("Missing Values").reset_index().rename(columns={"index": "Column"}),
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as error:
                st.error(f"Could not load dataset: {error}")


# ------------------------- FOOTER -------------------------

st.divider()
st.caption(
    "GNN-Based Intrusion Detection | CIC-IDS2017 | "
    "PortScan V6 Strict Zero-Day Experiment | Hybrid GNN Experiment"
)
