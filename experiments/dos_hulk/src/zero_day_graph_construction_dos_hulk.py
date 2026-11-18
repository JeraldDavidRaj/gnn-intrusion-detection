
"""
DoS Hulk Zero-Day GNN Graph Construction

Methodology:
- Benign-only preprocessing (median imputation and standardization).
- 80% of benign training flows form the context graph.
- Remaining 20% form the benign training query graph.
- Node features are calculated only from benign context traffic.
- Host mapping includes endpoints from all four prepared datasets.
- Validation, benign test and DoS Hulk test are query graphs.
- No DoS Hulk attack features are used to fit preprocessing or node features.

This experiment is isolated from PortScan V6.
"""

from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "dos_hulk"

DATA_DIR = EXPERIMENT_DIR / "data" / "prepared"
OUTPUT_DIR = EXPERIMENT_DIR / "data" / "graphs"
MODEL_DIR = EXPERIMENT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
CONTEXT_RATIO = 0.80

SOURCE_COLUMN = "Source IP"
DESTINATION_COLUMN = "Destination IP"

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

REQUIRED_COLUMNS = [
    SOURCE_COLUMN,
    DESTINATION_COLUMN
] + FEATURE_COLUMNS


# =========================================================
# 2. LOAD DATASET
# =========================================================

def load_dataset(filename):

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    print(f"\nLoading {filename}...")

    dtype_map = {
        SOURCE_COLUMN: "string",
        DESTINATION_COLUMN: "string"
    }

    for feature in FEATURE_COLUMNS:
        dtype_map[feature] = "float32"

    df = pd.read_csv(
        file_path,
        usecols=REQUIRED_COLUMNS,
        dtype=dtype_map,
        low_memory=False
    )

    for column in [SOURCE_COLUMN, DESTINATION_COLUMN]:
        df[column] = df[column].str.strip()

    df.dropna(
        subset=[SOURCE_COLUMN, DESTINATION_COLUMN],
        inplace=True
    )

    for feature in FEATURE_COLUMNS:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df.reset_index(drop=True, inplace=True)

    print("Rows:", len(df))

    return df


# =========================================================
# 3. FIT PREPROCESSING ON BENIGN TRAINING DATA ONLY
# =========================================================

def fit_preprocessing(train_df):

    print("\nFitting imputer and scaler...")

    train_features = train_df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    imputer = SimpleImputer(
        strategy="median"
    )

    imputed_features = imputer.fit_transform(
        train_features
    )

    scaler = StandardScaler()

    scaler.fit(imputed_features)

    joblib.dump(
        imputer,
        MODEL_DIR / "dos_hulk_imputer.pkl"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "dos_hulk_scaler.pkl"
    )

    print("DoS Hulk imputer saved.")
    print("DoS Hulk scaler saved.")

    return imputer, scaler


# =========================================================
# 4. TRANSFORM FEATURES
# =========================================================

def transform_features(df, imputer, scaler):

    raw_features = df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    imputed_features = imputer.transform(
        raw_features
    )

    scaled_features = scaler.transform(
        imputed_features
    )

    return scaled_features.astype(np.float32)


# =========================================================
# 5. CREATE GLOBAL HOST MAPPING
# =========================================================

def create_node_mapping(all_datasets):

    print("\nCreating global host mapping...")

    unique_ips = set()

    for dataset_name, df in all_datasets.items():

        unique_ips.update(
            df[SOURCE_COLUMN].dropna().astype(str).unique()
        )

        unique_ips.update(
            df[DESTINATION_COLUMN].dropna().astype(str).unique()
        )

        print(
            f"{dataset_name}: "
            f"{len(unique_ips)} cumulative unique hosts"
        )

    sorted_ips = sorted(unique_ips)

    node_mapping = {
        ip: index
        for index, ip in enumerate(sorted_ips)
    }

    print("Total unique hosts:", len(node_mapping))

    return node_mapping


# =========================================================
# 6. CONVERT IP ADDRESSES INTO EDGE INDICES
# =========================================================

def convert_ips_to_indices(df, node_mapping):

    source_indices = (
        df[SOURCE_COLUMN]
        .map(node_mapping)
        .to_numpy(dtype=np.int64)
    )

    destination_indices = (
        df[DESTINATION_COLUMN]
        .map(node_mapping)
        .to_numpy(dtype=np.int64)
    )

    if np.any(pd.isna(source_indices)):
        raise ValueError("Unmapped source IP addresses found.")

    if np.any(pd.isna(destination_indices)):
        raise ValueError("Unmapped destination IP addresses found.")

    edge_index = np.vstack([
        source_indices,
        destination_indices
    ])

    return edge_index


# =========================================================
# 7. GENERATE NODE FEATURES FROM TRAINING CONTEXT ONLY
# =========================================================

def create_node_features(
    context_edge_index,
    context_edge_attr,
    number_of_nodes
):

    print("\nGenerating training-only node features...")

    number_of_features = context_edge_attr.shape[1]

    outgoing_sum = np.zeros(
        (number_of_nodes, number_of_features),
        dtype=np.float64
    )

    incoming_sum = np.zeros(
        (number_of_nodes, number_of_features),
        dtype=np.float64
    )

    outgoing_count = np.zeros(
        number_of_nodes,
        dtype=np.float64
    )

    incoming_count = np.zeros(
        number_of_nodes,
        dtype=np.float64
    )

    source_nodes = context_edge_index[0]
    destination_nodes = context_edge_index[1]

    np.add.at(
        outgoing_sum,
        source_nodes,
        context_edge_attr
    )

    np.add.at(
        outgoing_count,
        source_nodes,
        1
    )

    np.add.at(
        incoming_sum,
        destination_nodes,
        context_edge_attr
    )

    np.add.at(
        incoming_count,
        destination_nodes,
        1
    )

    outgoing_mean = np.divide(
        outgoing_sum,
        outgoing_count[:, None],
        out=np.zeros_like(outgoing_sum),
        where=outgoing_count[:, None] != 0
    )

    incoming_mean = np.divide(
        incoming_sum,
        incoming_count[:, None],
        out=np.zeros_like(incoming_sum),
        where=incoming_count[:, None] != 0
    )

    degree_features = np.column_stack([
        np.log1p(outgoing_count),
        np.log1p(incoming_count)
    ])

    node_features = np.concatenate([
        outgoing_mean,
        incoming_mean,
        degree_features
    ], axis=1)

    print("Node feature shape:", node_features.shape)

    return node_features.astype(np.float32)


# =========================================================
# 8. SAVE CONTEXT GRAPH
# =========================================================

def save_context_graph(
    node_features,
    context_edge_index,
    context_edge_attr
):

    context_graph = {

        "node_features": torch.from_numpy(
            node_features
        ),

        "edge_index": torch.from_numpy(
            context_edge_index
        ),

        "edge_attr": torch.from_numpy(
            context_edge_attr
        )

    }

    output_path = OUTPUT_DIR / "dos_hulk_benign_context_graph.pt"

    torch.save(
        context_graph,
        output_path
    )

    print("\nContext graph saved:", output_path)


# =========================================================
# 9. SAVE QUERY GRAPH
# =========================================================

def save_query_graph(
    graph_name,
    df,
    scaled_features,
    node_mapping,
    node_features
):

    print(f"\nPreparing {graph_name}...")

    query_edge_index = convert_ips_to_indices(
        df,
        node_mapping
    )

    query_graph = {

        "node_features": torch.from_numpy(
            node_features
        ),

        "query_edge_index": torch.from_numpy(
            query_edge_index
        ),

        "query_edge_attr": torch.from_numpy(
            scaled_features
        ),

        "metadata": {
            "dataset": graph_name,
            "number_of_nodes": int(
                node_features.shape[0]
            ),
            "number_of_query_edges": int(
                query_edge_index.shape[1]
            ),
            "number_of_node_features": int(
                node_features.shape[1]
            ),
            "number_of_edge_features": int(
                scaled_features.shape[1]
            ),
            "unknown_endpoints": 0
        }

    }

    output_path = OUTPUT_DIR / f"dos_hulk_{graph_name}_graph.pt"

    torch.save(
        query_graph,
        output_path
    )

    print("Saved:", output_path)
    print("Query flows:", len(df))

    return query_graph["metadata"]


# =========================================================
# 10. SAVE HOST MAPPING
# =========================================================

def save_host_mapping(node_mapping):

    output_path = OUTPUT_DIR / "dos_hulk_host_mapping.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            node_mapping,
            file,
            indent=4
        )

    print("Host mapping saved:", output_path)


# =========================================================
# 11. MAIN EXECUTION
# =========================================================

def main():

    print("=" * 70)
    print("DOS HULK ZERO-DAY GNN GRAPH CONSTRUCTION")
    print("=" * 70)

    # A. Load all prepared datasets

    dataset_files = {
        "benign_train": "dos_hulk_benign_train_candidates.csv",
        "benign_validation": "dos_hulk_benign_validation.csv",
        "benign_test": "dos_hulk_benign_test.csv",
        "dos_hulk_test": "dos_hulk_test.csv"
    }

    all_datasets = {}

    for dataset_name, filename in dataset_files.items():

        all_datasets[dataset_name] = load_dataset(
            filename
        )

    # B. Fit preprocessing using benign training only

    train_df = all_datasets["benign_train"]

    imputer, scaler = fit_preprocessing(
        train_df
    )

    train_scaled = transform_features(
        train_df,
        imputer,
        scaler
    )

    # C. Separate context and training query flows

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    shuffled_indices = rng.permutation(
        len(train_df)
    )

    context_size = int(
        len(train_df) * CONTEXT_RATIO
    )

    context_indices = shuffled_indices[
        :context_size
    ]

    query_indices = shuffled_indices[
        context_size:
    ]

    context_df = train_df.iloc[
        context_indices
    ].copy()

    context_features = train_scaled[
        context_indices
    ]

    training_query_df = train_df.iloc[
        query_indices
    ].copy()

    training_query_features = train_scaled[
        query_indices
    ]

    print("\nTraining separation:")
    print("Total:", len(train_df))
    print("Context:", len(context_df))
    print("Training query:", len(training_query_df))

    # D. Create global mapping for distinct IP identities

    node_mapping = create_node_mapping(
        all_datasets
    )

    number_of_nodes = len(node_mapping)

    # E. Build context graph using benign context only

    context_edge_index = convert_ips_to_indices(
        context_df,
        node_mapping
    )

    print("\nContext graph:")
    print("Nodes:", number_of_nodes)
    print("Edges:", context_edge_index.shape[1])

    # F. Generate node features from benign context only

    node_features = create_node_features(
        context_edge_index,
        context_features,
        number_of_nodes
    )

    # G. Save context graph and host mapping

    save_context_graph(
        node_features,
        context_edge_index,
        context_features
    )

    save_host_mapping(
        node_mapping
    )

    # H. Save benign training query graph

    all_metadata = {}

    all_metadata["benign_train"] = save_query_graph(
        "benign_train",
        training_query_df,
        training_query_features,
        node_mapping,
        node_features
    )

    # I. Save validation and test query graphs

    for dataset_name in [
        "benign_validation",
        "benign_test",
        "dos_hulk_test"
    ]:

        df = all_datasets[dataset_name]

        scaled_features = transform_features(
            df,
            imputer,
            scaler
        )

        all_metadata[dataset_name] = save_query_graph(
            dataset_name,
            df,
            scaled_features,
            node_mapping,
            node_features
        )

        del scaled_features
        gc.collect()

    # J. Verify graph endpoints

    print("\n" + "=" * 70)
    print("GRAPH VERIFICATION")
    print("=" * 70)

    for dataset_name, metadata in all_metadata.items():

        print(
            dataset_name,
            "Unknown endpoints:",
            metadata["unknown_endpoints"]
        )

    # K. Save metadata

    construction_metadata = {

        "experiment": "DoS Hulk Zero-Day Detection",

        "random_seed": RANDOM_SEED,

        "context_ratio": CONTEXT_RATIO,

        "number_of_nodes": number_of_nodes,

        "number_of_node_features": int(
            node_features.shape[1]
        ),

        "number_of_edge_features": len(
            FEATURE_COLUMNS
        ),

        "unknown_node_index": None,

        "feature_columns": FEATURE_COLUMNS,

        "training_type": "Benign-only",

        "datasets": all_metadata

    }

    metadata_path = (
        OUTPUT_DIR / "dos_hulk_graph_construction_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            construction_metadata,
            file,
            indent=4
        )

    # L. Final output

    print("\n" + "=" * 70)
    print("DOS HULK GRAPH CONSTRUCTION COMPLETED")
    print("=" * 70)

    print("Total unique hosts:", number_of_nodes)
    print("Node features:", node_features.shape)
    print("Edge features:", len(FEATURE_COLUMNS))
    print("Context edges:", context_edge_index.shape[1])

    print("\nGenerated graph files:")

    for file_path in sorted(OUTPUT_DIR.iterdir()):
        print(file_path.name)

    print("\nGenerated model files:")
    print("dos_hulk_imputer.pkl")
    print("dos_hulk_scaler.pkl")

    print("\nNext stage: DoS Hulk V6 benign-only GNN training.")

    print("=" * 70)


if __name__ == "__main__":
    main()
