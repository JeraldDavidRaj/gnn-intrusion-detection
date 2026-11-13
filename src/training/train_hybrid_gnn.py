
"""
HYBRID GNN FOR UNSEEN ATTACK-CATEGORY DETECTION

Branch 1:
    Existing benign-only anomaly detector (V3).
    Its checkpoint and implementation remain unchanged.

Branch 2:
    Supervised GNN trained on:
        - BENIGN traffic
        - All attack categories except PortScan

Evaluation:
    - BENIGN test traffic
    - Completely held-out PortScan traffic

Chronological experiment:
    Training attacks: July 4-5, 2017
    Validation attacks: July 6, 2017
    Testing: July 7, 2017

PortScan is excluded from every training and validation operation.
"""

from pathlib import Path
import json
import random
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from torch_geometric.nn import GCNConv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = ROOT / "data" / "processed" / "cic_ids2017_cleaned.csv"

GRAPH_DIR = ROOT / "data" / "zero_day" / "graphs"
# The hybrid model reuses the saved PortScan V6 benign preprocessing.
PORTSCAN_MODEL_DIR = ROOT / "models" / "portscan_v6"
MODEL_DIR = ROOT / "models" / "hybrid"
REPORT_DIR = ROOT / "reports" / "hybrid"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

HIDDEN_CHANNELS = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001

EPOCHS = 35
PATIENCE = 7

BATCH_SIZE = 8192

MAX_TRAIN_PER_CLASS = 250000
MAX_VALIDATION_PER_CLASS = 100000

BENIGN_LABEL = "BENIGN"
HELD_OUT_LABEL = "PortScan"

SOURCE_COLUMN = "Source IP"
DESTINATION_COLUMN = "Destination IP"
TIMESTAMP_COLUMN = "Timestamp"
LABEL_COLUMN = "Label"

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
    DESTINATION_COLUMN,
    TIMESTAMP_COLUMN,
    LABEL_COLUMN
] + FEATURE_COLUMNS


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 75)
print("HYBRID GNN: UNSEEN ATTACK-CATEGORY DETECTION")
print("=" * 75)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. LOAD EXISTING GRAPH AND PREPROCESSING
# ============================================================

def load_graph(filename):

    path = GRAPH_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    print("Loading:", filename)

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


context_graph = load_graph("benign_context_graph.pt")
benign_train_graph = load_graph("benign_train_graph.pt")
benign_validation_graph = load_graph("benign_validation_graph.pt")
benign_test_graph = load_graph("benign_test_graph.pt")
portscan_test_graph = load_graph("portscan_test_graph.pt")


imputer_path = PORTSCAN_MODEL_DIR / "zero_day_imputer.pkl"
scaler_path = PORTSCAN_MODEL_DIR / "zero_day_scaler.pkl"
mapping_path = GRAPH_DIR / "host_mapping.json"

if not imputer_path.exists():
    raise FileNotFoundError(imputer_path)

if not scaler_path.exists():
    raise FileNotFoundError(scaler_path)

if not mapping_path.exists():
    raise FileNotFoundError(mapping_path)

imputer = joblib.load(imputer_path)
scaler = joblib.load(scaler_path)

with open(mapping_path, "r", encoding="utf-8") as file:
    host_mapping = json.load(file)


# ============================================================
# 4. PREPARE EXISTING BENIGN GRAPH TENSORS
# ============================================================

node_features = context_graph["node_features"].float()

context_edge_index = context_graph["edge_index"].long()

benign_train_edge_index = (
    benign_train_graph["query_edge_index"].long()
)

benign_train_edge_attr = (
    benign_train_graph["query_edge_attr"].float()
)

benign_validation_edge_index = (
    benign_validation_graph["query_edge_index"].long()
)

benign_validation_edge_attr = (
    benign_validation_graph["query_edge_attr"].float()
)

benign_test_edge_index = (
    benign_test_graph["query_edge_index"].long()
)

benign_test_edge_attr = (
    benign_test_graph["query_edge_attr"].float()
)

portscan_test_edge_index = (
    portscan_test_graph["query_edge_index"].long()
)

portscan_test_edge_attr = (
    portscan_test_graph["query_edge_attr"].float()
)

print("\nExisting benign graph loaded.")

print("Nodes:", node_features.shape[0])
print("Node features:", node_features.shape[1])
print("Context edges:", context_edge_index.shape[1])


# ============================================================
# 5. LOAD ORIGINAL CLEANED DATASET
# ============================================================

print("\nLoading original cleaned CIC-IDS2017 dataset...")

if not RAW_DATA.exists():
    raise FileNotFoundError(
        f"Original cleaned dataset not found: {RAW_DATA}"
    )

dtype_map = {
    SOURCE_COLUMN: "string",
    DESTINATION_COLUMN: "string",
    LABEL_COLUMN: "string"
}

for column in FEATURE_COLUMNS:
    dtype_map[column] = "float32"

df = pd.read_csv(
    RAW_DATA,
    usecols=REQUIRED_COLUMNS,
    dtype=dtype_map,
    low_memory=False
)

print("Original rows loaded:", len(df))

df[SOURCE_COLUMN] = df[SOURCE_COLUMN].str.strip()
df[DESTINATION_COLUMN] = df[DESTINATION_COLUMN].str.strip()

df[LABEL_COLUMN] = (
    df[LABEL_COLUMN]
    .astype(str)
    .str.strip()
)

df[TIMESTAMP_COLUMN] = pd.to_datetime(
    df[TIMESTAMP_COLUMN],
    dayfirst=True,
    errors="coerce"
)

df = df.dropna(
    subset=[
        SOURCE_COLUMN,
        DESTINATION_COLUMN,
        TIMESTAMP_COLUMN,
        LABEL_COLUMN
    ]
).copy()

df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
    [np.inf, -np.inf],
    np.nan
)

df.reset_index(drop=True, inplace=True)

df["Date"] = df[TIMESTAMP_COLUMN].dt.strftime("%Y-%m-%d")

print("\nLabel distribution:")
print(df[LABEL_COLUMN].value_counts())


# ============================================================
# 6. STRICTLY EXCLUDE PORTSCAN FROM ATTACK TRAINING
# ============================================================

print("\nPreparing attack-category training data...")

# Only known attack categories other than PortScan.
known_attack_df = df[
    (df[LABEL_COLUMN] != BENIGN_LABEL) &
    (df[LABEL_COLUMN] != HELD_OUT_LABEL)
].copy()

# July 4 and July 5: attack training.
attack_train_df = known_attack_df[
    known_attack_df["Date"].isin([
        "2017-07-04",
        "2017-07-05"
    ])
].copy()

# July 6: attack validation.
attack_validation_df = known_attack_df[
    known_attack_df["Date"] == "2017-07-06"
].copy()

# Safety assertions.
assert not (
    attack_train_df[LABEL_COLUMN] == HELD_OUT_LABEL
).any()

assert not (
    attack_validation_df[LABEL_COLUMN] == HELD_OUT_LABEL
).any()

assert set(attack_train_df["Date"].unique()).issubset({
    "2017-07-04",
    "2017-07-05"
})

assert set(attack_validation_df["Date"].unique()).issubset({
    "2017-07-06"
})

print("Known attack training flows:", len(attack_train_df))
print("Known attack validation flows:", len(attack_validation_df))

print("\nAttack training categories:")
print(attack_train_df[LABEL_COLUMN].value_counts())

print("\nAttack validation categories:")
print(attack_validation_df[LABEL_COLUMN].value_counts())

if len(attack_train_df) == 0:
    raise ValueError("No known attack training rows found.")

if len(attack_validation_df) == 0:
    raise ValueError(
        "No known attack validation rows found for July 6."
    )


# ============================================================
# 7. BALANCE TRAINING AND VALIDATION DATA
# ============================================================

def sample_rows(dataframe, maximum, seed):

    if len(dataframe) <= maximum:
        return dataframe.copy()

    return dataframe.sample(
        n=maximum,
        random_state=seed
    ).copy()


# Existing benign training query flows.
benign_train_df = pd.DataFrame({
    "Source": benign_train_graph["query_edge_index"][0].numpy(),
    "Destination": benign_train_graph["query_edge_index"][1].numpy()
})

# We use graph tensors directly for benign data.
# Attack data is sampled to control memory usage.

attack_train_df = sample_rows(
    attack_train_df,
    MAX_TRAIN_PER_CLASS,
    SEED
)

attack_validation_df = sample_rows(
    attack_validation_df,
    MAX_VALIDATION_PER_CLASS,
    SEED
)

# Sample benign validation indices.
validation_rng = np.random.default_rng(SEED)

benign_validation_count = min(
    len(benign_validation_edge_attr),
    MAX_VALIDATION_PER_CLASS
)

benign_validation_indices = validation_rng.choice(
    len(benign_validation_edge_attr),
    size=benign_validation_count,
    replace=False
)

benign_validation_indices = np.sort(
    benign_validation_indices
)

benign_validation_edge_index = (
    benign_validation_edge_index[
        :, benign_validation_indices
    ]
)

benign_validation_edge_attr = (
    benign_validation_edge_attr[
        benign_validation_indices
    ]
)

# Sample benign training query indices.
train_rng = np.random.default_rng(SEED)

benign_train_count = min(
    len(benign_train_edge_attr),
    MAX_TRAIN_PER_CLASS
)

benign_train_indices = train_rng.choice(
    len(benign_train_edge_attr),
    size=benign_train_count,
    replace=False
)

benign_train_indices = np.sort(
    benign_train_indices
)

benign_train_edge_index = (
    benign_train_edge_index[
        :, benign_train_indices
    ]
)

benign_train_edge_attr = (
    benign_train_edge_attr[
        benign_train_indices
    ]
)

print("\nBalanced sample sizes:")
print("Benign training:", len(benign_train_edge_attr))
print("Known attack training:", len(attack_train_df))
print("Benign validation:", len(benign_validation_edge_attr))
print("Known attack validation:", len(attack_validation_df))


# ============================================================
# 8. EXTEND HOST MAPPING FOR KNOWN ATTACK TRAINING
# ============================================================

print("\nExtending host mapping for known attack traffic...")

all_attack_ips = pd.concat([
    attack_train_df[SOURCE_COLUMN],
    attack_train_df[DESTINATION_COLUMN],
    attack_validation_df[SOURCE_COLUMN],
    attack_validation_df[DESTINATION_COLUMN]
]).dropna().astype(str).unique()

next_node_id = len(host_mapping)

new_host_count = 0

for ip in all_attack_ips:

    if ip not in host_mapping:

        host_mapping[ip] = next_node_id

        next_node_id += 1
        new_host_count += 1

print("New host nodes:", new_host_count)
print("Total graph nodes:", next_node_id)

# New attack-only hosts receive zero benign-context features.
# This prevents their attack traffic from contaminating node features.
if new_host_count > 0:

    extra_features = torch.zeros(
        (new_host_count, node_features.shape[1]),
        dtype=torch.float32
    )

    node_features = torch.cat(
        [node_features, extra_features],
        dim=0
    )


# ============================================================
# 9. CONVERT ATTACK FLOWS TO GRAPH EDGES
# ============================================================

def convert_attack_edges(dataframe):

    source_ids = (
        dataframe[SOURCE_COLUMN]
        .map(host_mapping)
        .to_numpy(dtype=np.int64)
    )

    destination_ids = (
        dataframe[DESTINATION_COLUMN]
        .map(host_mapping)
        .to_numpy(dtype=np.int64)
    )

    if np.any(pd.isna(source_ids)):
        raise ValueError("Unmapped attack source IP found.")

    if np.any(pd.isna(destination_ids)):
        raise ValueError("Unmapped attack destination IP found.")

    return np.vstack([
        source_ids,
        destination_ids
    ])


def transform_attack_features(dataframe):

    raw = dataframe[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    transformed = imputer.transform(raw)

    transformed = scaler.transform(transformed)

    transformed = np.nan_to_num(
        transformed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return transformed.astype(np.float32)


print("\nConverting attack flows into graph edges...")

attack_train_edge_index_np = convert_attack_edges(
    attack_train_df
)

attack_train_edge_attr_np = transform_attack_features(
    attack_train_df
)

attack_validation_edge_index_np = convert_attack_edges(
    attack_validation_df
)

attack_validation_edge_attr_np = transform_attack_features(
    attack_validation_df
)

attack_train_edge_index = torch.from_numpy(
    attack_train_edge_index_np
).long()

attack_train_edge_attr = torch.from_numpy(
    attack_train_edge_attr_np
).float()

attack_validation_edge_index = torch.from_numpy(
    attack_validation_edge_index_np
).long()

attack_validation_edge_attr = torch.from_numpy(
    attack_validation_edge_attr_np
).float()

print("Attack training edges:", attack_train_edge_index.shape[1])
print("Attack validation edges:", attack_validation_edge_index.shape[1])


# ============================================================
# 10. COMBINE TRAINING DATA
# ============================================================

# Label 0 = BENIGN
# Label 1 = known ATTACK

train_edge_index = torch.cat([
    benign_train_edge_index,
    attack_train_edge_index
], dim=1)

train_edge_attr = torch.cat([
    benign_train_edge_attr,
    attack_train_edge_attr
], dim=0)

train_labels = torch.cat([
    torch.zeros(
        len(benign_train_edge_attr),
        dtype=torch.float32
    ),
    torch.ones(
        len(attack_train_edge_attr),
        dtype=torch.float32
    )
])

# Shuffle training samples.
shuffle_generator = torch.Generator()
shuffle_generator.manual_seed(SEED)

shuffle_indices = torch.randperm(
    len(train_labels),
    generator=shuffle_generator
)

train_edge_index = train_edge_index[:, shuffle_indices]
train_edge_attr = train_edge_attr[shuffle_indices]
train_labels = train_labels[shuffle_indices]

# Validation data.
validation_edge_index = torch.cat([
    benign_validation_edge_index,
    attack_validation_edge_index
], dim=1)

validation_edge_attr = torch.cat([
    benign_validation_edge_attr,
    attack_validation_edge_attr
], dim=0)

validation_labels = torch.cat([
    torch.zeros(
        len(benign_validation_edge_attr),
        dtype=torch.float32
    ),
    torch.ones(
        len(attack_validation_edge_attr),
        dtype=torch.float32
    )
])

print("\nCombined training data:")
print("Total:", len(train_labels))
print("Benign:", int((train_labels == 0).sum()))
print("Known attacks:", int((train_labels == 1).sum()))

print("\nCombined validation data:")
print("Total:", len(validation_labels))
print("Benign:", int((validation_labels == 0).sum()))
print("Known attacks:", int((validation_labels == 1).sum()))


# ============================================================
# 11. MOVE GRAPH TENSORS TO DEVICE
# ============================================================

node_features = node_features.to(DEVICE)
context_edge_index = context_edge_index.to(DEVICE)

train_edge_index = train_edge_index.to(DEVICE)
train_edge_attr = train_edge_attr.to(DEVICE)
train_labels = train_labels.to(DEVICE)

validation_edge_index = validation_edge_index.to(DEVICE)
validation_edge_attr = validation_edge_attr.to(DEVICE)
validation_labels = validation_labels.to(DEVICE)

benign_test_edge_index = benign_test_edge_index.to(DEVICE)
benign_test_edge_attr = benign_test_edge_attr.to(DEVICE)

portscan_test_edge_index = portscan_test_edge_index.to(DEVICE)
portscan_test_edge_attr = portscan_test_edge_attr.to(DEVICE)


# ============================================================
# 12. DEFINE HYBRID SUPERVISED GNN
# ============================================================

class HybridAttackGCN(nn.Module):

    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        hidden_dim
    ):

        super().__init__()

        self.conv1 = GCNConv(
            node_input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        # Source embedding
        # Destination embedding
        # Absolute embedding difference
        # Element-wise embedding product
        # Actual standardized flow features

        decoder_input_dim = (
            hidden_dim * 4 + edge_input_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                decoder_input_dim,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(
                64,
                1
            )
        )

    def encode(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = self.relu(x)

        return x

    def decode(
        self,
        node_embeddings,
        edge_index,
        edge_features
    ):

        source_embeddings = node_embeddings[
            edge_index[0]
        ]

        destination_embeddings = node_embeddings[
            edge_index[1]
        ]

        edge_representation = torch.cat([
            source_embeddings,
            destination_embeddings,
            torch.abs(
                source_embeddings - destination_embeddings
            ),
            source_embeddings * destination_embeddings,
            edge_features
        ], dim=1)

        logits = self.classifier(
            edge_representation
        ).squeeze(1)

        return logits


model = HybridAttackGCN(
    node_input_dim=node_features.shape[1],
    edge_input_dim=train_edge_attr.shape[1],
    hidden_dim=HIDDEN_CHANNELS
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.BCEWithLogitsLoss()

print("\nModel architecture:")
print(model)

print(
    "\nTrainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# ============================================================
# 13. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    optimizer.zero_grad()

    node_embeddings = model.encode(
        node_features,
        context_edge_index
    )

    total_samples = len(train_labels)

    total_batches = (
        total_samples + BATCH_SIZE - 1
    ) // BATCH_SIZE

    total_loss = 0.0

    for batch_number, start in enumerate(
        range(0, total_samples, BATCH_SIZE)
    ):

        end = min(
            start + BATCH_SIZE,
            total_samples
        )

        batch_edges = train_edge_index[
            :, start:end
        ]

        batch_features = train_edge_attr[
            start:end
        ]

        batch_labels = train_labels[
            start:end
        ]

        logits = model.decode(
            node_embeddings,
            batch_edges,
            batch_features
        )

        loss = criterion(
            logits,
            batch_labels
        )

        weighted_loss = (
            loss * (end - start) / total_samples
        )

        is_last_batch = (
            batch_number == total_batches - 1
        )

        weighted_loss.backward(
            retain_graph=not is_last_batch
        )

        total_loss += (
            loss.item() * (end - start)
        )

        del logits
        del loss
        del weighted_loss

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=2.0
    )

    optimizer.step()

    return total_loss / total_samples


# ============================================================
# 14. PREDICTION FUNCTION
# ============================================================

def predict_probabilities(
    edge_index,
    edge_features
):

    model.eval()

    probabilities = []

    with torch.no_grad():

        node_embeddings = model.encode(
            node_features,
            context_edge_index
        )

        for start in range(
            0,
            edge_index.shape[1],
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                edge_index.shape[1]
            )

            batch_edges = edge_index[
                :, start:end
            ]

            batch_features = edge_features[
                start:end
            ]

            logits = model.decode(
                node_embeddings,
                batch_edges,
                batch_features
            )

            batch_probabilities = torch.sigmoid(
                logits
            )

            probabilities.append(
                batch_probabilities.cpu().numpy()
            )

    return np.concatenate(probabilities)


# ============================================================
# 15. VALIDATION AND THRESHOLD SELECTION
# ============================================================

def evaluate_validation():

    probabilities = predict_probabilities(
        validation_edge_index,
        validation_edge_attr
    )

    labels = validation_labels.cpu().numpy().astype(int)

    # Choose threshold using validation only.
    # Maximize balanced accuracy on known attack categories.
    thresholds = np.linspace(
        0.05,
        0.95,
        91
    )

    best_threshold = 0.5
    best_balanced_accuracy = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = balanced_accuracy_score(
            labels,
            predictions
        )

        if score > best_balanced_accuracy:

            best_balanced_accuracy = score
            best_threshold = float(threshold)

    return (
        probabilities,
        best_threshold,
        best_balanced_accuracy
    )


# ============================================================
# 16. TRAINING LOOP
# ============================================================

print("\n" + "=" * 75)
print("STARTING HYBRID GNN TRAINING")
print("=" * 75)

best_validation_score = -1.0
best_epoch = 0
epochs_without_improvement = 0

history = []

checkpoint_path = (
    MODEL_DIR / "hybrid_gnn_best.pth"
)

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch()

    (
        validation_probabilities,
        current_threshold,
        validation_score
    ) = evaluate_validation()

    history.append({

        "epoch": epoch,
        "training_loss": train_loss,
        "validation_balanced_accuracy": validation_score,
        "validation_threshold": current_threshold

    })

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Training Loss: {train_loss:.6f} "
        f"Validation Balanced Accuracy: {validation_score:.6f} "
        f"Threshold: {current_threshold:.2f}"
    )

    if validation_score > best_validation_score:

        best_validation_score = validation_score
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save({

            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_validation_score": best_validation_score,
            "threshold": current_threshold,
            "node_features": node_features.shape[1],
            "edge_features": train_edge_attr.shape[1],
            "hidden_channels": HIDDEN_CHANNELS

        }, checkpoint_path)

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 17. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_path = (
    REPORT_DIR / "hybrid_gnn_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

print("\nTraining history saved:", history_path)
print("Best epoch:", best_epoch)
print("Best validation balanced accuracy:", best_validation_score)


# ============================================================
# 18. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

threshold = float(
    checkpoint["threshold"]
)

print("\nBest hybrid model loaded:", checkpoint_path)
print("Selected validation threshold:", threshold)


# ============================================================
# 19. FINAL TEST: BENIGN VS HELD-OUT PORTSCAN
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST: BENIGN VS HELD-OUT PORTSCAN")
print("=" * 75)

print("PortScan was excluded from training and validation.")

benign_probabilities = predict_probabilities(
    benign_test_edge_index,
    benign_test_edge_attr
)

portscan_probabilities = predict_probabilities(
    portscan_test_edge_index,
    portscan_test_edge_attr
)

y_true = np.concatenate([

    np.zeros(
        len(benign_probabilities),
        dtype=int
    ),

    np.ones(
        len(portscan_probabilities),
        dtype=int
    )

])

all_probabilities = np.concatenate([

    benign_probabilities,
    portscan_probabilities

])

y_pred = (
    all_probabilities >= threshold
).astype(int)


# ============================================================
# 20. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

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

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)

false_negative_rate = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else 0.0
)

try:

    roc_auc = roc_auc_score(
        y_true,
        all_probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        all_probabilities
    )

except ValueError:

    roc_auc = None
    pr_auc = None


# ============================================================
# 21. SAVE METRICS
# ============================================================

metrics = {

    "experiment": "Hybrid GNN held-out PortScan",

    "training_attack_categories": sorted(
        attack_train_df[LABEL_COLUMN].unique().tolist()
    ),

    "excluded_attack_category": HELD_OUT_LABEL,

    "training_attack_flows": int(
        len(attack_train_df)
    ),

    "validation_attack_flows": int(
        len(attack_validation_df)
    ),

    "benign_test_flows": int(
        len(benign_probabilities)
    ),

    "portscan_test_flows": int(
        len(portscan_probabilities)
    ),

    "best_epoch": int(best_epoch),

    "validation_balanced_accuracy": float(
        best_validation_score
    ),

    "decision_threshold": float(threshold),

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "balanced_accuracy": float(balanced_accuracy),

    "false_positive_rate": float(false_positive_rate),

    "false_negative_rate": float(false_negative_rate),

    "portscan_detection_rate": float(recall),

    "roc_auc": (
        float(roc_auc)
        if roc_auc is not None
        else None
    ),

    "pr_auc": (
        float(pr_auc)
        if pr_auc is not None
        else None
    ),

    "confusion_matrix": cm.tolist()

}

metrics_path = (
    REPORT_DIR / "hybrid_gnn_evaluation_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# 22. SAVE FLOW-LEVEL PREDICTIONS
# ============================================================

benign_results = pd.DataFrame({

    "Actual_Label": "BENIGN",

    "Attack_Probability": benign_probabilities,

    "Predicted_Label": np.where(
        benign_probabilities >= threshold,
        "ATTACK",
        "BENIGN"
    )

})

portscan_results = pd.DataFrame({

    "Actual_Label": "PORTSCAN",

    "Attack_Probability": portscan_probabilities,

    "Predicted_Label": np.where(
        portscan_probabilities >= threshold,
        "ATTACK",
        "BENIGN"
    )

})

prediction_df = pd.concat(
    [
        benign_results,
        portscan_results
    ],
    ignore_index=True
)

prediction_path = (
    REPORT_DIR / "hybrid_gnn_flow_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 23. FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("HYBRID GNN FINAL RESULTS")
print("=" * 75)

print("Best epoch:", best_epoch)
print("Decision threshold:", threshold)

print("\nAccuracy:", round(accuracy, 6))
print("Precision:", round(precision, 6))
print("PortScan Recall:", round(recall, 6))
print("F1-score:", round(f1, 6))
print("Balanced Accuracy:", round(balanced_accuracy, 6))

print("False Positive Rate:", round(false_positive_rate, 6))
print("False Negative Rate:", round(false_negative_rate, 6))

print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["BENIGN", "PORTSCAN"],
        zero_division=0
    )
)

print("\nPortScan detection:")
print(
    f"{tp} / {len(portscan_probabilities)} "
    f"({recall * 100:.2f}%)"
)

print("\nAccuracy target:")
if accuracy >= 0.80:
    print("80% TARGET ACHIEVED.")
else:
    print("80% TARGET NOT ACHIEVED. Further investigation required.")

print("\nSaved files:")
print(checkpoint_path)
print(history_path)
print(metrics_path)
print(prediction_path)

print("\nExisting benign-only V3 checkpoint was not modified.")
print("Existing V3 reports were not modified.")

print("\nHYBRID GNN TRAINING AND EVALUATION COMPLETED.")
print("=" * 75)

