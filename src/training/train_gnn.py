from pathlib import Path
import json
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import joblib


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
EPOCHS = 10
HIDDEN_DIM = 64
DROPOUT = 0.3
LEARNING_RATE = 0.001
EDGE_BATCH_SIZE = 65536
PATIENCE = 3

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "cic_ids2017_cleaned.csv"
MODELS_DIR = BASE_DIR / "models" / "baseline"
REPORTS_DIR = BASE_DIR / "reports" / "baseline"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("GNN INTRUSION DETECTION â€” FULL DATASET TRAINING")
print("=" * 65)
print("Dataset:", DATA_PATH)
print("Device:", DEVICE)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check that the cleaned CSV is in data/processed/."
    )


# ============================================================
# 2. LOAD AND VALIDATE THE DATASET
# ============================================================

df = pd.read_csv(DATA_PATH, low_memory=False)

required_columns = [
    "Source IP",
    "Destination IP",
    "Label"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

feature_columns = [
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

missing_features = [
    column for column in feature_columns
    if column not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing selected features: {missing_features}"
    )

df = df.dropna(
    subset=["Source IP", "Destination IP", "Label"]
).copy()

df["Label"] = df["Label"].astype(str).str.strip()

df = df[
    (df["Label"] != "") &
    (df["Source IP"].astype(str).str.strip() != "") &
    (df["Destination IP"].astype(str).str.strip() != "")
].reset_index(drop=True)

df["binary_label"] = (
    df["Label"].str.upper() != "BENIGN"
).astype(np.int64)

print("\nDATASET SUMMARY")
print("-" * 45)
print(f"Valid labelled flows: {len(df):,}")
print(f"Number of features: {len(feature_columns)}")
print("\nAttack categories:")
print(df["Label"].value_counts().to_string())


# ============================================================
# 3. SPLIT FLOWS BEFORE PREPROCESSING OR GRAPH CONSTRUCTION
# ============================================================

all_indices = np.arange(len(df))
all_labels = df["binary_label"].to_numpy()

train_indices, temp_indices = train_test_split(
    all_indices,
    test_size=0.30,
    random_state=SEED,
    stratify=all_labels
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=all_labels[temp_indices]
)

train_df = df.iloc[train_indices].copy().reset_index(drop=True)
val_df = df.iloc[val_indices].copy().reset_index(drop=True)
test_df = df.iloc[test_indices].copy().reset_index(drop=True)

y_train = train_df["binary_label"].to_numpy()
y_val = val_df["binary_label"].to_numpy()
y_test = test_df["binary_label"].to_numpy()

print("\nDATA SPLIT")
print("-" * 45)
print(f"Training flows:   {len(train_df):,}")
print(f"Validation flows: {len(val_df):,}")
print(f"Testing flows:    {len(test_df):,}")


# ============================================================
# 4. FIT PREPROCESSING USING TRAINING DATA ONLY
# ============================================================

for column in feature_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

X_train_raw = train_df[feature_columns].copy()
X_val_raw = val_df[feature_columns].copy()
X_test_raw = test_df[feature_columns].copy()

# Fit imputer exclusively on training data.
imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)

X_train_imputed = imputer.fit_transform(X_train_raw)
X_val_imputed = imputer.transform(X_val_raw)
X_test_imputed = imputer.transform(X_test_raw)

# Fit scaler exclusively on training data.
scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train_imputed
).astype(np.float32)

X_val = scaler.transform(
    X_val_imputed
).astype(np.float32)

X_test = scaler.transform(
    X_test_imputed
).astype(np.float32)

print("\nPREPROCESSING COMPLETE")
print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Test features:", X_test.shape)


# ============================================================
# 5. BUILD A TRAINING-ONLY HOST COMMUNICATION GRAPH
# ============================================================

# IP addresses identify nodes. Only training flows contribute
# edges and numerical node features.
all_hosts = pd.Index(
    pd.concat(
        [
            df["Source IP"].astype(str),
            df["Destination IP"].astype(str)
        ],
        ignore_index=True
    ).unique()
)

host_to_id = {
    host: index for index, host in enumerate(all_hosts)
}

number_of_nodes = len(all_hosts)
number_of_features = len(feature_columns)

train_sources = (
    train_df["Source IP"].astype(str)
    .map(host_to_id).to_numpy(dtype=np.int64)
)

train_destinations = (
    train_df["Destination IP"].astype(str)
    .map(host_to_id).to_numpy(dtype=np.int64)
)

val_sources = (
    val_df["Source IP"].astype(str)
    .map(host_to_id).to_numpy(dtype=np.int64)
)

val_destinations = (
    val_df["Destination IP"].astype(str)
    .map(host_to_id).to_numpy(dtype=np.int64)
)

test_sources = (
    test_df["Source IP"].astype(str)
    .map(host_to_id).to_numpy(dtype=np.int64)
)

test_destinations = (
    test_df["Destination IP"].astype(str)
    .map(host_to_id).to_numpy(dtype=np.int64)
)

# Aggregate ONLY scaled training-flow features into host features.
node_feature_sums = np.zeros(
    (number_of_nodes, number_of_features),
    dtype=np.float64
)

node_counts = np.zeros(
    number_of_nodes,
    dtype=np.int64
)

for endpoints in (train_sources, train_destinations):
    np.add.at(node_counts, endpoints, 1)

for feature_index in range(number_of_features):
    np.add.at(
        node_feature_sums[:, feature_index],
        train_sources,
        X_train[:, feature_index]
    )
    np.add.at(
        node_feature_sums[:, feature_index],
        train_destinations,
        X_train[:, feature_index]
    )

node_features = (
    node_feature_sums /
    np.maximum(node_counts[:, None], 1)
).astype(np.float32)

# Isolated or previously unseen hosts have zero features.
node_features[node_counts == 0] = 0.0

# Message passing uses TRAINING edges only.
train_edge_index = torch.tensor(
    np.vstack([train_sources, train_destinations]),
    dtype=torch.long,
    device=DEVICE
)

x_nodes = torch.tensor(
    node_features,
    dtype=torch.float32,
    device=DEVICE
)

print("\nTRAINING GRAPH")
print("-" * 45)
print("Host nodes:", f"{number_of_nodes:,}")
print("Training graph edges:", f"{len(train_df):,}")
print("Node feature shape:", tuple(x_nodes.shape))
print("Validation/test edges are excluded from message passing.")


# ============================================================
# 6. DEFINE THE GNN
# ============================================================

class IntrusionGCN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        dropout=0.3
    ):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return x

    def classify_edges(
        self,
        node_embeddings,
        source_nodes,
        destination_nodes
    ):
        source_embeddings = node_embeddings[source_nodes]
        destination_embeddings = node_embeddings[destination_nodes]

        edge_embeddings = torch.cat(
            [source_embeddings, destination_embeddings],
            dim=1
        )

        return self.classifier(edge_embeddings)


model = IntrusionGCN(
    input_dim=number_of_features,
    hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT
).to(DEVICE)

# Account for class imbalance using training labels only.
class_values = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=class_values,
    y=y_train
)

weights = torch.ones(
    2, dtype=torch.float32, device=DEVICE
)

for class_value, class_weight in zip(
    class_values, class_weights
):
    weights[int(class_value)] = float(class_weight)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=5e-4
)


# ============================================================
# 7. HELPER FOR BATCHED EDGE EVALUATION
# ============================================================

def evaluate_edges(source_nodes, destination_nodes, labels):
    model.eval()

    predictions = []

    source_tensor = torch.as_tensor(
        source_nodes, dtype=torch.long, device=DEVICE
    )

    destination_tensor = torch.as_tensor(
        destination_nodes, dtype=torch.long, device=DEVICE
    )

    with torch.no_grad():
        # Encode nodes from training graph only.
        node_embeddings = model.encode(
            x_nodes, train_edge_index
        )

        for start in range(0, len(labels), EDGE_BATCH_SIZE):
            end = min(start + EDGE_BATCH_SIZE, len(labels))

            logits = model.classify_edges(
                node_embeddings,
                source_tensor[start:end],
                destination_tensor[start:end]
            )

            predictions.extend(
                logits.argmax(dim=1).cpu().numpy().tolist()
            )

    predictions = np.asarray(predictions, dtype=np.int64)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "f1": f1_score(
            labels, predictions, zero_division=0
        ),
        "predictions": predictions
    }


# ============================================================
# 8. TRAIN ON ALL TRAINING FLOWS IN EDGE MINI-BATCHES
# ============================================================

best_val_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

model_path = MODELS_DIR / "gnn_intrusion_model.pth"

print("\nTRAINING STARTED")
print("=" * 65)

training_start = time.time()

train_sources_tensor = torch.as_tensor(
    train_sources, dtype=torch.long, device=DEVICE
)

train_destinations_tensor = torch.as_tensor(
    train_destinations, dtype=torch.long, device=DEVICE
)

y_train_tensor = torch.as_tensor(
    y_train, dtype=torch.long, device=DEVICE
)

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    # Calculate node embeddings once per epoch.
    node_embeddings = model.encode(
        x_nodes, train_edge_index
    )

    total_loss = 0.0
    number_of_batches = (
        len(y_train) + EDGE_BATCH_SIZE - 1
    ) // EDGE_BATCH_SIZE

    # Backpropagate each edge batch while retaining the shared
    # node-encoding graph until the last batch.
    for batch_number, start in enumerate(
        range(0, len(y_train), EDGE_BATCH_SIZE)
    ):
        end = min(start + EDGE_BATCH_SIZE, len(y_train))

        logits = model.classify_edges(
            node_embeddings,
            train_sources_tensor[start:end],
            train_destinations_tensor[start:end]
        )

        batch_loss = criterion(
            logits,
            y_train_tensor[start:end]
        )

        # Normalize each batch contribution by total training rows.
        weighted_loss = batch_loss * (
            (end - start) / len(y_train)
        )

        retain_graph = batch_number < number_of_batches - 1

        weighted_loss.backward(retain_graph=retain_graph)

        total_loss += weighted_loss.item()

        del logits, batch_loss, weighted_loss

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=2.0
    )

    optimizer.step()

    # Validation flows do not contribute graph edges or node features.
    val_results = evaluate_edges(
        val_sources, val_destinations, y_val
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {total_loss:.4f} | "
        f"Val Accuracy: {val_results['accuracy']:.4f} | "
        f"Val Precision: {val_results['precision']:.4f} | "
        f"Val Recall: {val_results['recall']:.4f} | "
        f"Val F1: {val_results['f1']:.4f}"
    )

    if val_results["f1"] > best_val_f1:
        best_val_f1 = val_results["f1"]
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": number_of_features,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
                "feature_columns": feature_columns,
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1
            },
            model_path
        )

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping: validation F1 did not improve.")
        break

training_time = time.time() - training_start

# Restore the best validation checkpoint before final testing.
checkpoint = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

print("\nBest epoch:", best_epoch)
print("Best validation F1:", round(best_val_f1, 4))
print("Training time (seconds):", round(training_time, 2))


# ============================================================
# 9. FINAL TEST â€” HELD-OUT FLOWS ONLY
# ============================================================

test_start = time.time()

test_results = evaluate_edges(
    test_sources, test_destinations, y_test
)

test_inference_time = time.time() - test_start

test_predictions = test_results["predictions"]

print("\nFINAL TEST RESULTS")
print("=" * 65)
print("Accuracy:", round(test_results["accuracy"], 4))
print("Precision:", round(test_results["precision"], 4))
print("Recall:", round(test_results["recall"], 4))
print("F1-score:", round(test_results["f1"], 4))
print("Test flows:", f"{len(y_test):,}")
print("Test inference time (seconds):", round(test_inference_time, 2))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
)

print("Confusion Matrix:")
print(cm)


# ============================================================
# 10. SAVE MODEL PREPROCESSING AND FEATURE METADATA
# ============================================================

joblib.dump(
    imputer,
    MODELS_DIR / "gnn_feature_imputer.pkl"
)

joblib.dump(
    scaler,
    MODELS_DIR / "gnn_feature_scaler.pkl"
)

joblib.dump(
    feature_columns,
    MODELS_DIR / "gnn_feature_names.pkl"
)

joblib.dump(
    {host: idx for host, idx in host_to_id.items()},
    MODELS_DIR / "gnn_host_mapping.pkl"
)

# Save training-derived host features for later inference.
np.save(
    MODELS_DIR / "gnn_node_features.npy",
    node_features
)

torch.save(
    train_edge_index.cpu(),
    MODELS_DIR / "gnn_training_edge_index.pt"
)

with open(
    MODELS_DIR / "gnn_training_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {
            "dataset": str(DATA_PATH),
            "valid_flows": int(len(df)),
            "training_flows": int(len(train_df)),
            "validation_flows": int(len(val_df)),
            "test_flows": int(len(test_df)),
            "number_of_nodes": int(number_of_nodes),
            "number_of_features": int(number_of_features),
            "best_epoch": int(best_epoch),
            "best_validation_f1": float(best_val_f1),
            "training_seconds": float(training_time),
            "evaluation_protocol": (
                "Random flow split; train-only preprocessing, "
                "host feature aggregation and message-passing edges"
            )
        },
        file,
        indent=4
    )


# ============================================================
# 11. SAVE TEST PREDICTIONS AND REPORTS
# ============================================================

test_report = test_df[
    [
        column for column in [
            "Timestamp",
            "Source IP",
            "Destination IP",
            "Label"
        ]
        if column in test_df.columns
    ]
].copy()

test_report["actual_binary_label"] = y_test
test_report["predicted_binary_label"] = test_predictions

test_report["prediction"] = np.where(
    test_predictions == 1,
    "PREDICTED_ATTACK",
    "PREDICTED_BENIGN"
)

test_report.to_csv(
    REPORTS_DIR / "gnn_test_predictions.csv",
    index=False
)

suspicious_flows = test_report[
    test_report["predicted_binary_label"] == 1
].copy()

suspicious_flows.to_csv(
    REPORTS_DIR / "gnn_suspicious_flows.csv",
    index=False
)

if len(suspicious_flows):
    suspicious_hosts = pd.concat(
        [
            suspicious_flows[
                ["Source IP"]
            ].rename(columns={"Source IP": "Host IP"}),
            suspicious_flows[
                ["Destination IP"]
            ].rename(columns={"Destination IP": "Host IP"})
        ],
        ignore_index=True
    )

    host_summary = (
        suspicious_hosts.groupby("Host IP")
        .size()
        .reset_index(name="Suspicious Flow Involvement")
        .sort_values(
            "Suspicious Flow Involvement",
            ascending=False
        )
    )

    host_summary.to_csv(
        REPORTS_DIR / "gnn_suspicious_hosts.csv",
        index=False
    )

    incident_columns = [
        "Source IP",
        "Destination IP",
        "Label"
    ]

    if "Timestamp" in suspicious_flows.columns:
        incident_summary = (
            suspicious_flows.groupby(
                incident_columns,
                dropna=False
            )
            .agg(
                Flow_Count=("prediction", "size"),
                First_Observed=("Timestamp", "min"),
                Last_Observed=("Timestamp", "max")
            )
            .reset_index()
        )
    else:
        incident_summary = (
            suspicious_flows.groupby(
                incident_columns,
                dropna=False
            )
            .size()
            .reset_index(name="Flow_Count")
        )

    incident_summary = incident_summary.sort_values(
        "Flow_Count",
        ascending=False
    )

    incident_summary.to_csv(
        REPORTS_DIR / "gnn_incident_summary.csv",
        index=False
    )

else:
    pd.DataFrame(
        columns=["Host IP", "Suspicious Flow Involvement"]
    ).to_csv(
        REPORTS_DIR / "gnn_suspicious_hosts.csv",
        index=False
    )

    pd.DataFrame(
        columns=[
            "Source IP",
            "Destination IP",
            "Label",
            "Flow_Count"
        ]
    ).to_csv(
        REPORTS_DIR / "gnn_incident_summary.csv",
        index=False
    )


# ============================================================
# 12. SAVE METRICS
# ============================================================

metrics = pd.DataFrame(
    [
        {
            "Model": "GCN",
            "Accuracy": test_results["accuracy"],
            "Precision": test_results["precision"],
            "Recall": test_results["recall"],
            "F1_Score": test_results["f1"],
            "Training_Flows": len(train_df),
            "Validation_Flows": len(val_df),
            "Test_Flows": len(test_df),
            "Predicted_Attacks": int(
                (test_predictions == 1).sum()
            ),
            "Best_Epoch": best_epoch,
            "Training_Time_Seconds": training_time,
            "Test_Inference_Time_Seconds": test_inference_time
        }
    ]
)

metrics.to_csv(
    REPORTS_DIR / "gnn_model_metrics.csv",
    index=False
)

print("\nSAVED OUTPUTS")
print("=" * 65)
print("Model:", model_path)
print("Metrics:", REPORTS_DIR / "gnn_model_metrics.csv")
print("Test predictions:", REPORTS_DIR / "gnn_test_predictions.csv")
print("Suspicious flows:", REPORTS_DIR / "gnn_suspicious_flows.csv")
print("Suspicious hosts:", REPORTS_DIR / "gnn_suspicious_hosts.csv")
print("Incident summary:", REPORTS_DIR / "gnn_incident_summary.csv")
print("\nFULL-DATASET GNN TRAINING AND EVALUATION COMPLETED.")
