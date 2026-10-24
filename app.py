
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

st.set_page_config(
    page_title="GNN Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("GNN-Based Intrusion Detection System")
st.caption(
    "CIC-IDS2017 | Graph Neural Network | "
    "Network threat investigation"
)

def load_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception as error:
            st.error(f"Could not read {path.name}: {error}")
    return pd.DataFrame()

metrics_df = load_csv(REPORTS_DIR / "gnn_model_metrics.csv")
predictions_df = load_csv(REPORTS_DIR / "gnn_test_predictions.csv")
flows_df = load_csv(REPORTS_DIR / "gnn_suspicious_flows.csv")
hosts_df = load_csv(REPORTS_DIR / "gnn_suspicious_hosts.csv")
incidents_df = load_csv(REPORTS_DIR / "gnn_incident_summary.csv")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Dataset Inspection",
        "Suspicious Hosts",
        "Candidate Incidents",
        "Flow Explorer",
        "Test Predictions"
    ]
)

if page == "Overview":
    st.header("Model Overview")

    if not metrics_df.empty:
        row = metrics_df.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        for col, label, key in [
            (col1, "Accuracy", "Accuracy"),
            (col2, "Precision", "Precision"),
            (col3, "Recall", "Recall"),
            (col4, "F1 Score", "F1_Score")
        ]:
            value = row.get(key)
            if pd.notna(value):
                col.metric(label, f"{float(value):.4f}")
            else:
                col.metric(label, "N/A")

        st.subheader("Training Details")
        st.dataframe(metrics_df, use_container_width=True)

        if "Test_Flows" in metrics_df.columns:
            st.metric(
                "Held-out test flows",
                f"{int(row['Test_Flows']):,}"
            )

        if "Predicted_Attacks" in metrics_df.columns:
            st.metric(
                "Flows predicted as attacks",
                f"{int(row['Predicted_Attacks']):,}"
            )
    else:
        st.warning(
            "Metrics are missing. Run `python -m src.train_gnn` "
            "from the project root first."
        )

    st.subheader("Available Reports")

    report_status = pd.DataFrame({
        "Report": [
            "Model metrics",
            "Test predictions",
            "Suspicious flows",
            "Suspicious hosts",
            "Candidate incidents"
        ],
        "Status": [
            "Available" if not df.empty else "Missing"
            for df in [
                metrics_df,
                predictions_df,
                flows_df,
                hosts_df,
                incidents_df
            ]
        ]
    })
    st.dataframe(report_status, use_container_width=True)

elif page == "Dataset Inspection":
    st.header("Dataset Inspection")

    dataset_dirs = [
        DATA_DIR / "processed",
        DATA_DIR,
        BASE_DIR / "Dataset"
    ]

    dataset_files = []
    seen_paths = set()

    for folder in dataset_dirs:
        if folder.exists():
            for file_path in folder.glob("*.csv"):
                resolved = str(file_path.resolve())
                if resolved not in seen_paths:
                    dataset_files.append(file_path)
                    seen_paths.add(resolved)

    if not dataset_files:
        st.warning(
            "No CSV files found in the expected data folders. "
            "Check the dataset location."
        )
    else:
        st.success(f"Found {len(dataset_files)} CSV file(s).")

        dataset_info = pd.DataFrame([
            {
                "File": file.name,
                "Size (MB)": round(
                    file.stat().st_size / (1024 ** 2), 2
                ),
                "Location": str(file.relative_to(BASE_DIR))
            }
            for file in dataset_files
        ])
        st.dataframe(dataset_info, use_container_width=True)

        selected_file = st.selectbox(
            "Select a dataset CSV",
            dataset_files,
            format_func=lambda file: file.name
        )

        if st.button("Load CSV Preview"):
            with st.spinner("Loading preview..."):
                preview = pd.read_csv(
                    selected_file,
                    nrows=1000,
                    low_memory=False
                )

            st.write(f"Preview rows: {len(preview):,}")
            st.write(f"Columns: {len(preview.columns):,}")
            st.dataframe(preview, use_container_width=True)

            st.subheader("Column Names")
            st.write(list(preview.columns))

            if "Label" in preview.columns:
                st.subheader("Label Distribution in Preview")
                st.bar_chart(preview["Label"].value_counts())

elif page == "Suspicious Hosts":
    st.header("Suspicious Host Analysis")

    if hosts_df.empty:
        st.info("No suspicious-host report found.")
    else:
        st.caption(
            "These counts come from model-generated reports; "
            "they are investigation leads, not confirmed incidents."
        )
        st.dataframe(hosts_df, use_container_width=True)

        numeric_cols = hosts_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_cols and "Host IP" in hosts_df.columns:
            count_col = st.selectbox(
                "Chart metric",
                numeric_cols,
                index=len(numeric_cols) - 1
            )
            chart_df = hosts_df.set_index("Host IP")[[count_col]]
            st.bar_chart(chart_df.head(20))

elif page == "Candidate Incidents":
    st.header("Candidate Incident Investigation")

    if incidents_df.empty:
        st.info("No incident summary found.")
    else:
        filtered = incidents_df.copy()

        if "Investigation_Priority" in filtered.columns:
            priorities = filtered["Investigation_Priority"].dropna().unique()
            selected_priorities = st.multiselect(
                "Filter by investigation priority",
                priorities,
                default=list(priorities)
            )
            filtered = filtered[
                filtered["Investigation_Priority"].isin(
                    selected_priorities
                )
            ]

        st.write(f"Candidate groups: {len(filtered):,}")
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "Download incident report",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="gnn_incident_summary_filtered.csv",
            mime="text/csv"
        )

elif page == "Flow Explorer":
    st.header("Suspicious Flow Explorer")

    if flows_df.empty:
        st.info("No suspicious-flow report found.")
    else:
        filtered = flows_df.copy()

        search = st.text_input(
            "Search by IP address, timestamp, or attack label"
        )

        if search:
            mask = filtered.astype(str).apply(
                lambda column: column.str.contains(
                    search,
                    case=False,
                    na=False,
                    regex=False
                )
            ).any(axis=1)
            filtered = filtered[mask]

        if "Label" in filtered.columns:
            labels = filtered["Label"].dropna().unique().tolist()
            selected_labels = st.multiselect(
                "Filter by dataset label",
                labels,
                default=labels
            )
            filtered = filtered[
                filtered["Label"].isin(selected_labels)
            ]

        st.write(f"Matching flows: {len(filtered):,}")
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "Download filtered flows",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="gnn_suspicious_flows_filtered.csv",
            mime="text/csv"
        )

elif page == "Test Predictions":
    st.header("Held-out Test Predictions")

    if predictions_df.empty:
        st.info("No test-prediction report found.")
    else:
        st.write(f"Prediction records: {len(predictions_df):,}")

        search = st.text_input("Search test predictions")

        filtered = predictions_df.copy()
        if search:
            mask = filtered.astype(str).apply(
                lambda column: column.str.contains(
                    search,
                    case=False,
                    na=False,
                    regex=False
                )
            ).any(axis=1)
            filtered = filtered[mask]

        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "Download test predictions",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="gnn_test_predictions.csv",
            mime="text/csv"
        )

st.divider()
st.caption(
    "Model training: src/train_gnn.py | "
    "Dataset preprocessing: src/data_preprocessing.py"
)
