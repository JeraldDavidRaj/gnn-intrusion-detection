
"""
DoS Hulk Zero-Day Intrusion Detection
Adaptation of the PortScan V6 methodology.

Architecture:
1. Temporal GCN flow reconstruction
2. Benign-only temporal Isolation Forest
3. GNN + Isolation Forest ensemble (behaviour rule excluded)
4. 99th percentile threshold using benign validation only
"""

from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from torch_geometric.nn import GCNConv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "dos_hulk"

DATA = EXP / "data" / "prepared"
GRAPHS = EXP / "data" / "graphs"
MODELS = EXP / "models"
REPORTS = EXP / "reports"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42
CONTEXT_RATIO = 0.80
HIDDEN = 64
EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 2048
LR = 0.0005
WEIGHT_DECAY = 1e-5
DROPOUT = 0.25
MIN_LR = 1e-5

THRESHOLD_PERCENTILE = 99.0
IFOREST_TREES = 500
IFOREST_MAX_SAMPLES = 100000

# DoS Hulk component analysis showed that the temporal Isolation
# Forest was the strongest individual detector. The behaviour rule
# is scan-oriented and is excluded from this DoS-specific ensemble.
W_GNN = 0.20
W_IF = 0.80
W_RULE = 0.00

MIN_LOGVAR = -6.0
MAX_LOGVAR = 4.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

FLOW_COLS = [
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

TEMP_COLS = [
    "src_flows_1m",
    "src_unique_dst_ip_1m",
    "src_unique_dst_port_1m",
    "src_syn_count_1m",
    "src_syn_ratio_1m",
    "src_flows_5m",
    "src_unique_dst_ip_5m",
    "src_unique_dst_port_5m",
    "dst_flows_1m",
    "dst_unique_src_ip_1m",
    "dst_unique_src_port_1m",
    "dst_syn_count_1m",
    "src_destination_port_entropy_1m",
    "pair_flows_prev_1m",
    "pair_unique_dst_ports_prev_1m",
    "pair_unique_dst_ports_prev_5m",
]

TEMP_DIM = len(TEMP_COLS)

CSV_COLS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
] + FLOW_COLS

print("=" * 75)
print("DOS HULK BENIGN-ONLY GNN TRAINING")
print("=" * 75)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# 2. LOAD GRAPH ARTIFACTS
# =========================================================

def load_graph(filename):

    path = GRAPHS / filename

    if not path.exists():
        raise FileNotFoundError(path)

    print("Loading:", filename)

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


context = load_graph("dos_hulk_benign_context_graph.pt")
train_g = load_graph("dos_hulk_benign_train_graph.pt")
val_g = load_graph("dos_hulk_benign_validation_graph.pt")
benign_g = load_graph("dos_hulk_benign_test_graph.pt")
attack_g = load_graph("dos_hulk_dos_hulk_test_graph.pt")

x_cpu = context["node_features"].float().contiguous()
context_edges_cpu = context["edge_index"].long().contiguous()

train_edges = train_g["query_edge_index"].long().contiguous()
train_flow = train_g["query_edge_attr"].float().contiguous()

val_edges = val_g["query_edge_index"].long().contiguous()
val_flow = val_g["query_edge_attr"].float().contiguous()

benign_edges = benign_g["query_edge_index"].long().contiguous()
benign_flow = benign_g["query_edge_attr"].float().contiguous()

attack_edges = attack_g["query_edge_index"].long().contiguous()
attack_flow = attack_g["query_edge_attr"].float().contiguous()

NODE_DIM = x_cpu.shape[1]
FLOW_DIM = train_flow.shape[1]

if NODE_DIM != 44 or FLOW_DIM != 21:
    raise ValueError(
        f"Unexpected graph dimensions: {NODE_DIM} nodes / "
        f"{FLOW_DIM} flow features"
    )

print("Nodes:", x_cpu.shape[0])
print("Node features:", NODE_DIM)
print("Flow features:", FLOW_DIM)
print("Training query edges:", train_edges.shape[1])
print("Validation edges:", val_edges.shape[1])
print("Benign test edges:", benign_edges.shape[1])
print("DoS Hulk test edges:", attack_edges.shape[1])


# =========================================================
# 3. LOAD HOST MAPPING AND PREPROCESSORS
# =========================================================

with open(
    GRAPHS / "dos_hulk_host_mapping.json",
    encoding="utf-8"
) as f:

    host_map = {
        str(k): int(v)
        for k, v in json.load(f).items()
    }

imputer = joblib.load(
    MODELS / "dos_hulk_imputer.pkl"
)

flow_scaler = joblib.load(
    MODELS / "dos_hulk_scaler.pkl"
)


# =========================================================
# 4. LOAD PREPARED CSV FILES
# =========================================================

def read_split(filename):

    path = DATA / filename

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        usecols=list(dict.fromkeys(CSV_COLS)),
        low_memory=False
    )

    for c in ["Source IP", "Destination IP"]:
        df[c] = df[c].astype("string").str.strip()

    # Recover ports and protocol from CIC-IDS2017 Flow ID.
    parts = df["Flow ID"].astype("string").str.split("-", expand=True)

    if parts.shape[1] != 5:
        raise ValueError(
            f"{filename}: unexpected Flow ID format. "
            f"Examples: {df['Flow ID'].head().tolist()}"
        )

    src_port = pd.to_numeric(parts[2], errors="coerce")
    dst_port = pd.to_numeric(parts[3], errors="coerce")
    protocol = pd.to_numeric(parts[4], errors="coerce")

    if src_port.isna().any() or dst_port.isna().any() or protocol.isna().any():
        raise ValueError(
            f"{filename}: invalid ports or protocol in Flow ID."
        )

    df["Source Port"] = src_port.astype(np.int32)
    df["Destination Port"] = dst_port.astype(np.int32)
    df["Protocol"] = protocol.astype(np.int32)

    df["_time"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
        dayfirst=True
    )

    if df[
        ["Source IP", "Destination IP", "_time"]
    ].isna().any().any():
        raise ValueError(
            f"{filename}: missing IP or invalid timestamp."
        )

    for c in FLOW_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[FLOW_COLS] = df[FLOW_COLS].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df.reset_index(drop=True, inplace=True)

    print(filename, "rows:", len(df))

    return df


train_df = read_split(
    "dos_hulk_benign_train_candidates.csv"
)

val_df = read_split(
    "dos_hulk_benign_validation.csv"
)

benign_df = read_split(
    "dos_hulk_benign_test.csv"
)

attack_df = read_split(
    "dos_hulk_test.csv"
)


# =========================================================
# 5. VERIFY CSV AND GRAPH ALIGNMENT
# =========================================================

rng = np.random.default_rng(SEED)

permutation = rng.permutation(len(train_df))

context_size = int(
    len(train_df) * CONTEXT_RATIO
)

query_indices = permutation[context_size:]

train_query_df = train_df.iloc[query_indices].copy()


def make_edges(df, name):

    src = df["Source IP"].map(host_map)
    dst = df["Destination IP"].map(host_map)

    if src.isna().any() or dst.isna().any():
        raise ValueError(name + ": unmapped IP address.")

    return torch.from_numpy(
        np.vstack([
            src.to_numpy(dtype=np.int64),
            dst.to_numpy(dtype=np.int64)
        ])
    ).long()


def verify_edges(name, df, saved_edges):

    rebuilt = make_edges(df, name)

    if (
        rebuilt.shape != saved_edges.shape
        or not torch.equal(rebuilt, saved_edges)
    ):
        raise RuntimeError(
            name + ": CSV and graph edge order mismatch."
        )

    print(name, "edge alignment verified.")


def verify_flow(name, df, saved):

    raw = df[FLOW_COLS].to_numpy(dtype=np.float32)

    scaled = flow_scaler.transform(
        imputer.transform(raw)
    ).astype(np.float32)

    if not np.allclose(
        scaled,
        saved.numpy(),
        rtol=2e-4,
        atol=2e-4
    ):
        raise RuntimeError(
            name + ": CSV and graph flow features mismatch."
        )

    print(name, "flow feature alignment verified.")


verify_edges("Training", train_query_df, train_edges)
verify_edges("Validation", val_df, val_edges)
verify_edges("Benign test", benign_df, benign_edges)
verify_edges("DoS Hulk test", attack_df, attack_edges)

verify_flow("Training", train_query_df, train_flow)
verify_flow("Validation", val_df, val_flow)
verify_flow("Benign test", benign_df, benign_flow)
verify_flow("DoS Hulk test", attack_df, attack_flow)


# =========================================================
# 6. CAUSAL TEMPORAL FEATURE ENGINEERING
# =========================================================

def prepare_temporal_frame(df):

    w = df[
        [
            "Source IP",
            "Destination IP",
            "_time",
            "Source Port",
            "Destination Port",
            "SYN Flag Count"
        ]
    ].copy()

    w["_minute"] = (
        w["_time"].astype("int64") // 60_000_000_000
    ).astype(np.int64)

    w["_five"] = (
        w["_minute"] // 5
    ).astype(np.int64)

    w["_srcport"] = (
        pd.to_numeric(w["Source Port"], errors="coerce")
        .fillna(-1)
        .astype(np.int64)
    )

    w["_dstport"] = (
        pd.to_numeric(w["Destination Port"], errors="coerce")
        .fillna(-1)
        .astype(np.int64)
    )

    w["_syn"] = (
        pd.to_numeric(w["SYN Flag Count"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )

    return w


def entropy_by_source_minute(w):

    keys = ["Source IP", "_minute"]

    counts = (
        w.groupby(
            keys + ["_dstport"],
            sort=False,
            observed=True
        )
        .size()
        .rename("n")
        .reset_index()
    )

    totals = (
        counts.groupby(
            keys,
            sort=False,
            observed=True
        )["n"]
        .transform("sum")
        .to_numpy(dtype=np.float64)
    )

    probabilities = (
        counts["n"].to_numpy(dtype=np.float64)
        / np.maximum(totals, 1.0)
    )

    counts["_entropy_part"] = (
        -probabilities
        * np.log2(np.maximum(probabilities, 1e-12))
    )

    return (
        counts.groupby(
            keys,
            sort=False,
            observed=True
        )["_entropy_part"]
        .sum()
        .rename("src_destination_port_entropy_1m")
        .reset_index()
    )


def causal_temporal_features(history_df, query_df, name):

    print("Deriving causal temporal features:", name)

    q = query_df.copy().reset_index(drop=True)
    q["_qid"] = np.arange(len(q), dtype=np.int64)

    h = history_df.copy()

    if (
        len(h)
        and len(q)
        and h["_time"].max() > q["_time"].min()
    ):
        raise ValueError(
            name + ": future history detected. Temporal leakage prevented."
        )

    combined = pd.concat(
        [h, q],
        ignore_index=True,
        copy=False
    )

    w = prepare_temporal_frame(combined)

    # Previous completed one-minute source statistics.
    src1 = (
        w.groupby(
            ["Source IP", "_minute"],
            sort=False,
            observed=True
        )
        .agg(
            src_flows_1m=("Destination IP", "size"),
            src_unique_dst_ip_1m=("Destination IP", "nunique"),
            src_unique_dst_port_1m=("_dstport", "nunique"),
            src_syn_count_1m=("_syn", "sum")
        )
        .reset_index()
    )

    src1["src_syn_ratio_1m"] = (
        src1["src_syn_count_1m"]
        / src1["src_flows_1m"].clip(lower=1)
    )

    src1 = src1.merge(
        entropy_by_source_minute(w),
        on=["Source IP", "_minute"],
        how="left"
    )

    src1["_src_target_minute"] = src1["_minute"] + 1
    src1 = src1.drop(columns=["_minute"])

    # Previous completed five-minute source statistics.
    src5 = (
        w.groupby(
            ["Source IP", "_five"],
            sort=False,
            observed=True
        )
        .agg(
            src_flows_5m=("Destination IP", "size"),
            src_unique_dst_ip_5m=("Destination IP", "nunique"),
            src_unique_dst_port_5m=("_dstport", "nunique")
        )
        .reset_index()
    )

    src5["_src_target_five"] = src5["_five"] + 1
    src5 = src5.drop(columns=["_five"])

    # Previous completed one-minute destination statistics.
    dst1 = (
        w.groupby(
            ["Destination IP", "_minute"],
            sort=False,
            observed=True
        )
        .agg(
            dst_flows_1m=("Source IP", "size"),
            dst_unique_src_ip_1m=("Source IP", "nunique"),
            dst_unique_src_port_1m=("_srcport", "nunique"),
            dst_syn_count_1m=("_syn", "sum")
        )
        .reset_index()
    )

    dst1["_dst_target_minute"] = dst1["_minute"] + 1
    dst1 = dst1.drop(columns=["_minute"])

    # Previous completed minute for each source-destination pair.
    pair1 = (
        w.groupby(
            ["Source IP", "Destination IP", "_minute"],
            sort=False,
            observed=True
        )
        .agg(
            pair_flows_prev_1m=("_dstport", "size"),
            pair_unique_dst_ports_prev_1m=("_dstport", "nunique")
        )
        .reset_index()
    )

    pair1["_pair_target_minute"] = pair1["_minute"] + 1
    pair1 = pair1.drop(columns=["_minute"])

    # Previous completed five-minute pair statistics.
    pair5 = (
        w.groupby(
            ["Source IP", "Destination IP", "_five"],
            sort=False,
            observed=True
        )
        .agg(
            pair_unique_dst_ports_prev_5m=("_dstport", "nunique")
        )
        .reset_index()
    )

    pair5["_pair_target_five"] = pair5["_five"] + 1
    pair5 = pair5.drop(columns=["_five"])

    out = q[
        ["_qid", "Source IP", "Destination IP", "_time"]
    ].copy()

    out["_minute"] = (
        out["_time"].astype("int64") // 60_000_000_000
    ).astype(np.int64)

    out["_five"] = (
        out["_minute"] // 5
    ).astype(np.int64)

    out = out.merge(
        src1,
        left_on=["Source IP", "_minute"],
        right_on=["Source IP", "_src_target_minute"],
        how="left",
        sort=False
    )

    out = out.merge(
        src5,
        left_on=["Source IP", "_five"],
        right_on=["Source IP", "_src_target_five"],
        how="left",
        sort=False
    )

    out = out.merge(
        dst1,
        left_on=["Destination IP", "_minute"],
        right_on=["Destination IP", "_dst_target_minute"],
        how="left",
        sort=False
    )

    out = out.merge(
        pair1,
        left_on=["Source IP", "Destination IP", "_minute"],
        right_on=[
            "Source IP",
            "Destination IP",
            "_pair_target_minute"
        ],
        how="left",
        sort=False
    )

    out = out.merge(
        pair5,
        left_on=["Source IP", "Destination IP", "_five"],
        right_on=[
            "Source IP",
            "Destination IP",
            "_pair_target_five"
        ],
        how="left",
        sort=False
    )

    out = out.sort_values("_qid", kind="stable")

    arr = (
        out[TEMP_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    if len(arr) != len(query_df) or not np.isfinite(arr).all():
        raise RuntimeError(
            name + ": temporal feature generation failed."
        )

    print("Temporal feature shape:", arr.shape)

    return arr


# Training uses only benign training-period history.
train_temp_all = causal_temporal_features(
    train_df.iloc[0:0],
    train_df,
    "benign_train"
)

train_temp = train_temp_all[query_indices]

# Validation uses earlier benign training and validation observations.
val_temp = causal_temporal_features(
    train_df,
    val_df,
    "benign_validation"
)

# Combine benign test and DoS Hulk test into one chronological stream.
# Labels are not used to create temporal features.
test_stream = pd.concat(
    [benign_df, attack_df],
    ignore_index=True,
    copy=False
)

test_stream["_original_test_row"] = np.arange(
    len(test_stream),
    dtype=np.int64
)

test_stream = (
    test_stream.sort_values("_time", kind="stable")
    .reset_index(drop=True)
)

test_history = pd.concat(
    [train_df, val_df],
    ignore_index=True,
    copy=False
)

test_temp_sorted = causal_temporal_features(
    test_history,
    test_stream,
    "combined_test_stream"
)

test_temp_stream = np.empty_like(test_temp_sorted)

test_temp_stream[
    test_stream["_original_test_row"].to_numpy(dtype=np.int64)
] = test_temp_sorted

benign_temp = test_temp_stream[:len(benign_df)]
attack_temp = test_temp_stream[len(benign_df):]


# =========================================================
# 7. TEMPORAL SCALING - BENIGN TRAINING ONLY
# =========================================================

def log_transform(a):

    return np.log1p(
        np.maximum(a, 0)
    ).astype(np.float32)


temp_scaler = StandardScaler()

temp_scaler.fit(
    log_transform(train_temp_all)
)

train_temp_scaled = np.clip(
    temp_scaler.transform(log_transform(train_temp)),
    -10,
    10
).astype(np.float32)

val_temp_scaled = np.clip(
    temp_scaler.transform(log_transform(val_temp)),
    -10,
    10
).astype(np.float32)

benign_temp_scaled = np.clip(
    temp_scaler.transform(log_transform(benign_temp)),
    -10,
    10
).astype(np.float32)

attack_temp_scaled = np.clip(
    temp_scaler.transform(log_transform(attack_temp)),
    -10,
    10
).astype(np.float32)


# =========================================================
# 8. TRANSPARENT BEHAVIOUR SCORE
# =========================================================

def scan_rule(a):

    d = pd.DataFrame(a, columns=TEMP_COLS)

    return (
        0.16 * np.log1p(d["src_flows_1m"].to_numpy())
        + 0.12 * np.log1p(d["src_unique_dst_ip_1m"].to_numpy())
        + 0.18 * np.log1p(d["src_unique_dst_port_1m"].to_numpy())
        + 0.10 * np.log1p(d["src_unique_dst_port_5m"].to_numpy())
        + 0.08 * d["src_syn_ratio_1m"].to_numpy()
        + 0.06 * np.log1p(d["dst_unique_src_ip_1m"].to_numpy())
        + 0.18 * np.log1p(
            d["pair_unique_dst_ports_prev_1m"].to_numpy()
        )
        + 0.12 * np.log1p(
            d["pair_unique_dst_ports_prev_5m"].to_numpy()
        )
    ).astype(np.float64)


train_rule = scan_rule(train_temp)
val_rule = scan_rule(val_temp)
benign_rule = scan_rule(benign_temp)
attack_rule = scan_rule(attack_temp)


# =========================================================
# 9. BENIGN-ONLY TEMPORAL ISOLATION FOREST
# =========================================================

print("Fitting benign-only temporal Isolation Forest...")

iforest = IsolationForest(
    n_estimators=IFOREST_TREES,
    max_samples=min(
        IFOREST_MAX_SAMPLES,
        len(train_temp_scaled)
    ),
    contamination="auto",
    max_features=1.0,
    bootstrap=False,
    random_state=SEED,
    n_jobs=-1
)

iforest.fit(train_temp_scaled)


def iforest_score(a, batch=50000):

    chunks = []

    for start in range(0, len(a), batch):

        chunks.append(
            -iforest.decision_function(
                a[start:start + batch]
            )
        )

    return np.concatenate(chunks).astype(np.float64)


val_if = iforest_score(val_temp_scaled)
benign_if = iforest_score(benign_temp_scaled)
attack_if = iforest_score(attack_temp_scaled)


# =========================================================
# 10. TEMPORAL GCN RECONSTRUCTION MODEL
# =========================================================

x = x_cpu.to(DEVICE)
context_edges = context_edges_cpu.to(DEVICE)


class TemporalGCN(nn.Module):

    def __init__(self, node_dim, hidden, output_dim):

        super().__init__()

        self.conv1 = GCNConv(node_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)

        self.decoder = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim * 2)
        )

        self.output_dim = output_dim

    def encode(self, features, edge_index):

        h = self.relu(
            self.conv1(features, edge_index)
        )

        h = self.dropout(h)

        return self.relu(
            self.conv2(h, edge_index)
        )

    def decode(self, h, edge_index):

        s = h[edge_index[0]]
        d = h[edge_index[1]]

        z = torch.cat(
            [
                s,
                d,
                torch.abs(s - d),
                s * d
            ],
            dim=1
        )

        out = self.decoder(z)

        mu = out[:, :self.output_dim]

        logvar = torch.clamp(
            out[:, self.output_dim:],
            MIN_LOGVAR,
            MAX_LOGVAR
        )

        return mu, logvar


def nll_per_flow(mu, logvar, target):

    err2 = (target - mu) ** 2

    return (
        0.5 * (
            logvar
            + err2 * torch.exp(-logvar)
        )
    ).mean(dim=1)


OUTPUT_DIM = FLOW_DIM + TEMP_DIM

model = TemporalGCN(
    NODE_DIM,
    HIDDEN,
    OUTPUT_DIM
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=MIN_LR
)

train_targets = torch.cat(
    [
        train_flow,
        torch.from_numpy(train_temp_scaled)
    ],
    dim=1
).contiguous()

val_targets = torch.cat(
    [
        val_flow,
        torch.from_numpy(val_temp_scaled)
    ],
    dim=1
).contiguous()

benign_targets = torch.cat(
    [
        benign_flow,
        torch.from_numpy(benign_temp_scaled)
    ],
    dim=1
).contiguous()

attack_targets = torch.cat(
    [
        attack_flow,
        torch.from_numpy(attack_temp_scaled)
    ],
    dim=1
).contiguous()

print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)

print("Reconstruction dimensions:", OUTPUT_DIM)


def train_epoch():

    model.train()
    optimizer.zero_grad(set_to_none=True)

    h = model.encode(x, context_edges)

    n = train_edges.shape[1]
    batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    total = 0.0

    for bi, start in enumerate(
        range(0, n, BATCH_SIZE)
    ):

        end = min(start + BATCH_SIZE, n)

        ei = train_edges[:, start:end].to(DEVICE)
        target = train_targets[start:end].to(DEVICE)

        mu, lv = model.decode(h, ei)

        losses = nll_per_flow(mu, lv, target)

        loss = losses.mean() * (
            (end - start) / n
        )

        loss.backward(
            retain_graph=(bi < batches - 1)
        )

        total += float(
            losses.detach().sum().item()
        )

        del ei, target, mu, lv, losses, loss

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    del h

    return total / n


@torch.no_grad()
def gnn_scores(edge_index, targets):

    model.eval()

    h = model.encode(x, context_edges)

    out = []

    for start in range(
        0,
        edge_index.shape[1],
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            edge_index.shape[1]
        )

        ei = edge_index[:, start:end].to(DEVICE)
        target = targets[start:end].to(DEVICE)

        mu, lv = model.decode(h, ei)

        out.append(
            nll_per_flow(
                mu,
                lv,
                target
            ).cpu().numpy()
        )

        del ei, target, mu, lv

    del h

    return np.concatenate(out).astype(np.float64)


# =========================================================
# 11. TRAINING AND EARLY STOPPING
# =========================================================

checkpoint_path = MODELS / "dos_hulk_gnn_best.pth"

history = []
best_val = float("inf")
best_epoch = 0
wait = 0

print("\nStarting benign-only GNN training...")

for epoch in range(1, EPOCHS + 1):

    train_loss = train_epoch()

    validation_score = gnn_scores(
        val_edges,
        val_targets
    )

    validation_loss = float(
        validation_score.mean()
    )

    history.append({
        "epoch": epoch,
        "training_nll": train_loss,
        "validation_nll": validation_loss
    })

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train NLL: {train_loss:.6f} "
        f"Validation NLL: {validation_loss:.6f}"
    )

    scheduler.step(validation_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Current learning rate: {current_lr:.7f}")

    if validation_loss < best_val:

        best_val = validation_loss
        best_epoch = epoch
        wait = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "best_validation_nll": best_val,
                "node_features": NODE_DIM,
                "flow_features": FLOW_DIM,
                "temporal_features": TEMP_DIM,
                "temporal_columns": TEMP_COLS,
                "hidden_channels": HIDDEN,
                "seed": SEED
            },
            checkpoint_path
        )

    else:

        wait += 1

    if wait >= PATIENCE:
        print("Early stopping.")
        break

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


history_path = REPORTS / "dos_hulk_training_history.csv"

pd.DataFrame(history).to_csv(
    history_path,
    index=False
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Best epoch:",
    best_epoch,
    "| best validation NLL:",
    best_val
)


# =========================================================
# 12. BENIGN-ONLY VALIDATION CALIBRATION
# =========================================================

print("Scoring benign validation...")

val_gnn = gnn_scores(
    val_edges,
    val_targets
)


def percentile_rank(values, reference):

    ref = np.sort(
        np.asarray(reference, dtype=np.float64)
    )

    vals = np.asarray(
        values,
        dtype=np.float64
    )

    return (
        np.searchsorted(ref, vals, side="right")
        / max(len(ref), 1)
    )


def combine(gnn, iso, rule, refs):

    rg = percentile_rank(
        gnn,
        refs["gnn"]
    )

    ri = percentile_rank(
        iso,
        refs["if"]
    )

    rr = percentile_rank(
        rule,
        refs["rule"]
    )

    score = (
        W_GNN * rg
        + W_IF * ri
        + W_RULE * rr
    )

    return score, rg, ri, rr


refs = {
    "gnn": val_gnn,
    "if": val_if,
    "rule": val_rule
}

val_combined, _, _, _ = combine(
    val_gnn,
    val_if,
    val_rule,
    refs
)

threshold = float(
    np.percentile(
        val_combined,
        THRESHOLD_PERCENTILE
    )
)

print("99th percentile threshold:", threshold)


# =========================================================
# 13. TEST SCORING
# =========================================================

print("Scoring benign test...")

benign_gnn = gnn_scores(
    benign_edges,
    benign_targets
)

(
    benign_combined,
    benign_rg,
    benign_ri,
    benign_rr
) = combine(
    benign_gnn,
    benign_if,
    benign_rule,
    refs
)

print("Scoring held-out DoS Hulk...")

attack_gnn = gnn_scores(
    attack_edges,
    attack_targets
)

(
    attack_combined,
    attack_rg,
    attack_ri,
    attack_rr
) = combine(
    attack_gnn,
    attack_if,
    attack_rule,
    refs
)

benign_pred = (
    benign_combined > threshold
).astype(np.int32)

attack_pred = (
    attack_combined > threshold
).astype(np.int32)

y_true = np.concatenate([
    np.zeros(len(benign_pred), dtype=np.int32),
    np.ones(len(attack_pred), dtype=np.int32)
])

y_pred = np.concatenate([
    benign_pred,
    attack_pred
])

all_scores = np.concatenate([
    benign_combined,
    attack_combined
])


# =========================================================
# 14. EVALUATION METRICS
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

accuracy = float(
    accuracy_score(y_true, y_pred)
)

precision = float(
    precision_score(
        y_true,
        y_pred,
        zero_division=0
    )
)

recall = float(
    recall_score(
        y_true,
        y_pred,
        zero_division=0
    )
)

f1 = float(
    f1_score(
        y_true,
        y_pred,
        zero_division=0
    )
)

fpr = float(
    fp / (fp + tn)
) if (fp + tn) else 0.0

fnr = float(
    fn / (fn + tp)
) if (fn + tp) else 0.0

benign_fpr = float(benign_pred.mean())
attack_tpr = float(attack_pred.mean())

try:
    roc_auc = float(
        roc_auc_score(y_true, all_scores)
    )
except ValueError:
    roc_auc = None

try:
    pr_auc = float(
        average_precision_score(y_true, all_scores)
    )
except ValueError:
    pr_auc = None


threshold_tradeoff = {}

for pct in [95.0, 97.5, 99.0, 99.5, 99.9]:

    th = float(
        np.percentile(val_combined, pct)
    )

    threshold_tradeoff[str(pct)] = {
        "validation_percentile": pct,
        "threshold": th,
        "benign_test_fpr": float(
            np.mean(benign_combined > th)
        ),
        "dos_hulk_detection_rate": float(
            np.mean(attack_combined > th)
        )
    }


# =========================================================
# 15. SAVE MODEL SUPPORT ARTIFACTS
# =========================================================

temporal_scaler_path = (
    MODELS / "dos_hulk_temporal_scaler.pkl"
)

iforest_path = (
    MODELS / "dos_hulk_temporal_iforest.pkl"
)

joblib.dump(
    temp_scaler,
    temporal_scaler_path
)

joblib.dump(
    iforest,
    iforest_path
)


# =========================================================
# 16. SAVE EVALUATION METRICS
# =========================================================

metrics = {
    "experiment": "DoS Hulk Zero-Day Detection",
    "architecture": (
        "GCN flow/causal-temporal reconstruction + "
        "benign-only temporal Isolation Forest; "
        "behaviour rule excluded from the DoS Hulk ensemble"
    ),
    "training_type": "Benign-only",
    "attack_labels_used_for_training": False,
    "best_epoch": int(best_epoch),
    "best_validation_nll": float(best_val),
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "threshold": threshold,
    "weights": {
        "gnn": W_GNN,
        "temporal_iforest": W_IF,
        "behaviour_rule": W_RULE
    },
    "training_configuration": {
        "epochs_max": EPOCHS,
        "patience": PATIENCE,
        "learning_rate_initial": LR,
        "learning_rate_minimum": MIN_LR,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "dropout": DROPOUT,
        "isolation_forest_trees": IFOREST_TREES
    },
    "temporal_features": TEMP_COLS,
    "temporal_semantics": (
        "Previous completed minute and previous completed "
        "five-minute bucket; current bucket excluded"
    ),
    "benign_training_flows": int(len(train_df)),
    "benign_validation_flows": int(len(val_df)),
    "benign_test_flows": int(len(benign_pred)),
    "dos_hulk_test_flows": int(len(attack_pred)),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "false_positive_rate": fpr,
    "false_negative_rate": fnr,
    "benign_test_false_positive_rate": benign_fpr,
    "dos_hulk_detection_rate": attack_tpr,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "confusion_matrix": cm.tolist(),
    "threshold_tradeoff": threshold_tradeoff,
    "limitations": [
        (
            "This is a new training configuration; improved test performance "
            "is not guaranteed and must be established by the held-out test."
        ),
        (
            "DoS Hulk has one benign training date, "
            "2017-07-03, and one validation date, 2017-07-04."
        ),
        (
            "Host mapping is transductive because endpoint identities "
            "from all prepared splits are included."
        ),
        (
            "The behaviour score is retained from PortScan V6 and is "
            "scan-oriented rather than DoS-specific."
        ),
        (
            "Temporal features use completed one-minute and "
            "five-minute buckets, not event-by-event streaming."
        )
    ]
}

metrics_path = (
    REPORTS / "dos_hulk_evaluation_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# =========================================================
# 17. SAVE FLOW-LEVEL PREDICTIONS
# =========================================================

benign_out = pd.DataFrame({
    "Actual_Label": "BENIGN",
    "GNN_NLL": benign_gnn,
    "GNN_Rank": benign_rg,
    "Temporal_IForest_Score": benign_if,
    "Temporal_IForest_Rank": benign_ri,
    "Behaviour_Score": benign_rule,
    "Behaviour_Rank": benign_rr,
    "Combined_Anomaly_Score": benign_combined,
    "Predicted_Label": np.where(
        benign_pred == 1,
        "ANOMALY",
        "NORMAL"
    )
})

attack_out = pd.DataFrame({
    "Actual_Label": "DOS_HULK",
    "GNN_NLL": attack_gnn,
    "GNN_Rank": attack_rg,
    "Temporal_IForest_Score": attack_if,
    "Temporal_IForest_Rank": attack_ri,
    "Behaviour_Score": attack_rule,
    "Behaviour_Rank": attack_rr,
    "Combined_Anomaly_Score": attack_combined,
    "Predicted_Label": np.where(
        attack_pred == 1,
        "ANOMALY",
        "NORMAL"
    )
})

predictions_path = (
    REPORTS / "dos_hulk_flow_predictions.csv"
)

pd.concat(
    [benign_out, attack_out],
    ignore_index=True
).to_csv(
    predictions_path,
    index=False
)


# =========================================================
# 18. SAVE CALIBRATION DETAILS
# =========================================================

calibration_path = (
    REPORTS / "dos_hulk_calibration.json"
)

with open(
    calibration_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "threshold": threshold,
            "validation_component_99th_percentiles": {
                "gnn": float(np.percentile(val_gnn, 99)),
                "iforest": float(np.percentile(val_if, 99)),
                "behaviour_rule": float(np.percentile(val_rule, 99))
            },
            "temporal_scaler": str(temporal_scaler_path),
            "isolation_forest": str(iforest_path)
        },
        f,
        indent=4
    )


# =========================================================
# 19. FINAL RESULTS
# =========================================================

print("\n" + "=" * 75)
print("DOS HULK ZERO-DAY GNN RESULTS")
print("=" * 75)

print("Best epoch:", best_epoch)
print("Best validation NLL:", round(best_val, 6))
print("Threshold:", round(threshold, 6))
print("Accuracy:", round(accuracy, 6))
print("Precision:", round(precision, 6))
print("Recall:", round(recall, 6))
print("F1:", round(f1, 6))
print("Benign test FPR:", round(benign_fpr, 6))
print("DoS Hulk detection rate:", round(attack_tpr, 6))
print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)
print("Confusion matrix:\n", cm)

print("\nThreshold trade-off:")

for pct, item in threshold_tradeoff.items():

    print(
        f"{pct}% threshold | "
        f"Benign FPR: {item['benign_test_fpr']:.4%} | "
        f"DoS Hulk detection: "
        f"{item['dos_hulk_detection_rate']:.4%}"
    )

print("\nSaved files:")

for path in [
    checkpoint_path,
    history_path,
    metrics_path,
    predictions_path,
    calibration_path,
    temporal_scaler_path,
    iforest_path
]:
    print(path)

print("\nDoS Hulk training and evaluation completed.")
print("=" * 75)
