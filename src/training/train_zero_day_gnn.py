"""
V6: Causal temporal and source-destination port-behaviour-aware GNN anomaly detection.
Separate experiment. Preserves all V5 artifacts.

Uses existing graph artifacts and prepared CSVs. Temporal features use only completed prior-minute / prior-five-minute buckets. Attack labels are never used for fitting or calibration; test traffic is treated as an unlabeled chronological stream.
"""

from pathlib import Path
import json, random, gc, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch_geometric.nn import GCNConv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "zero_day"
GRAPHS = DATA / "graphs"
MODELS = ROOT / "models" / "portscan_v6"
REPORTS = ROOT / "reports" / "zero_day" / "portscan_v6" / "evaluation"
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42
CONTEXT_RATIO = 0.80
HIDDEN = 64
EPOCHS = 40
PATIENCE = 7
BATCH_SIZE = 2048
LR = 0.001
WEIGHT_DECAY = 1e-5
THRESHOLD_PERCENTILE = 99.0
IFOREST_TREES = 250
IFOREST_MAX_SAMPLES = 100000
W_GNN, W_IF, W_RULE = 0.35, 0.40, 0.25
MIN_LOGVAR, MAX_LOGVAR = -6.0, 4.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

FLOW_COLS = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Packet Length Mean",
    "Packet Length Std", "SYN Flag Count", "ACK Flag Count",
    "RST Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "Active Mean", "Idle Mean"
]
TEMP_COLS = [
    "src_flows_1m", "src_unique_dst_ip_1m", "src_unique_dst_port_1m",
    "src_syn_count_1m", "src_syn_ratio_1m", "src_flows_5m",
    "src_unique_dst_ip_5m", "src_unique_dst_port_5m",
    "dst_flows_1m", "dst_unique_src_ip_1m",
    "dst_unique_src_port_1m", "dst_syn_count_1m",
    "src_destination_port_entropy_1m",
    "pair_flows_prev_1m", "pair_unique_dst_ports_prev_1m",
    "pair_unique_dst_ports_prev_5m"
]
TEMP_DIM = len(TEMP_COLS)
CSV_COLS = [
    "Flow ID", "Source IP", "Destination IP",
    "Timestamp", "Verified_Timestamp"
] + FLOW_COLS

print("=" * 76)
print("ZERO-DAY GNN V6: CAUSAL TEMPORAL / PAIR-PORT-AWARE")
print("=" * 76)
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), "GB")


# 1. LOAD SAVED GRAPHS

def load_graph(name):
    p = GRAPHS / name
    if not p.exists():
        raise FileNotFoundError(p)
    print("Loading:", name)
    return torch.load(p, map_location="cpu", weights_only=False)

context = load_graph("benign_context_graph.pt")
train_g = load_graph("benign_train_graph.pt")
val_g = load_graph("benign_validation_graph.pt")
benign_g = load_graph("benign_test_graph.pt")
attack_g = load_graph("portscan_test_graph.pt")

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

N_NODES = x_cpu.shape[0]
NODE_DIM = x_cpu.shape[1]
FLOW_DIM = train_flow.shape[1]
if NODE_DIM != 44 or FLOW_DIM != 21:
    raise ValueError(f"Unexpected graph dimensions: {NODE_DIM} node / {FLOW_DIM} edge features.")


# 2. LOAD MAPPING AND EXISTING TRAIN-FIT PREPROCESSORS

with open(GRAPHS / "host_mapping.json", encoding="utf-8") as f:
    host_map = {str(k): int(v) for k, v in json.load(f).items()}

imputer = joblib.load(MODELS / "zero_day_imputer.pkl")
flow_scaler = joblib.load(MODELS / "zero_day_scaler.pkl")


# 3. LOAD PREPARED CSV FILES

def read_split(filename):
    p = DATA / filename
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, usecols=list(dict.fromkeys(CSV_COLS)), low_memory=False)
    for c in ["Source IP", "Destination IP"]:
        df[c] = df[c].astype("string").str.strip()
    # Prepared split CSVs do not contain separate port/protocol columns.
    # CIC-IDS2017 Flow ID format in this dataset:
    # SourceIP-DestinationIP-SourcePort-DestinationPort-Protocol
    # Example: 192.168.10.5-8.254.250.126-49188-80-6
    # The CSV Source IP and Destination IP columns remain authoritative
    # for graph endpoints; Flow ID is used only to recover ports/protocol.
    flow_parts = df["Flow ID"].astype("string").str.split("-", expand=True)

    if flow_parts.shape[1] != 5:
        examples = df["Flow ID"].head(5).tolist()
        raise ValueError(
            f"{filename}: expected five fields in Flow ID. "
            f"Examples: {examples}"
        )

    flow_source_port = pd.to_numeric(flow_parts[2], errors="coerce")
    flow_destination_port = pd.to_numeric(flow_parts[3], errors="coerce")
    flow_protocol = pd.to_numeric(flow_parts[4], errors="coerce")

    invalid = (
        flow_source_port.isna()
        | flow_destination_port.isna()
        | flow_protocol.isna()
    )

    if invalid.any():
        examples = df.loc[invalid, "Flow ID"].head(5).tolist()
        raise ValueError(
            f"{filename}: could not recover numeric ports/protocol "
            f"from Flow ID. Examples: {examples}"
        )

    df["Source Port"] = flow_source_port.astype(np.int32)
    df["Destination Port"] = flow_destination_port.astype(np.int32)
    df["Protocol"] = flow_protocol.astype(np.int32)

    time_col = "Verified_Timestamp" if "Verified_Timestamp" in df.columns else "Timestamp"
    df["_time"] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    if df[["Source IP", "Destination IP", "_time"]].isna().any().any():
        raise ValueError(f"{filename} contains missing IPs or invalid timestamps.")
    for c in FLOW_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[FLOW_COLS] = df[FLOW_COLS].replace([np.inf, -np.inf], np.nan)
    df.reset_index(drop=True, inplace=True)
    print(filename, "rows:", len(df))
    return df

train_df = read_split("benign_train.csv")
val_df = read_split("benign_validation.csv")
benign_df = read_split("benign_test.csv")
attack_df = read_split("portscan_test.csv")


# 4. RECREATE TRAIN/CONTEXT SPLIT AND VERIFY EDGE ORDER

rng = np.random.default_rng(SEED)
permutation = rng.permutation(len(train_df))
context_size = int(len(train_df) * CONTEXT_RATIO)
query_indices = permutation[context_size:]
train_query_df = train_df.iloc[query_indices].copy()

def make_edges(df, label):
    src = df["Source IP"].map(host_map)
    dst = df["Destination IP"].map(host_map)
    if src.isna().any() or dst.isna().any():
        raise ValueError(label + ": IP missing from host mapping.")
    return torch.from_numpy(np.vstack([
        src.to_numpy(dtype=np.int64), dst.to_numpy(dtype=np.int64)
    ])).long()

def verify_edges(name, df, saved_edges):
    rebuilt = make_edges(df, name)
    if rebuilt.shape != saved_edges.shape or not torch.equal(rebuilt, saved_edges):
        raise RuntimeError(
            name + ": CSV row order does not match graph edge order. "
            "Stopped to prevent incorrect temporal-feature alignment."
        )
    print(name, "edge alignment verified.")

def verify_flow(name, df, saved):
    raw = df[FLOW_COLS].to_numpy(dtype=np.float32)
    scaled = flow_scaler.transform(imputer.transform(raw)).astype(np.float32)
    if scaled.shape != tuple(saved.shape) or not np.allclose(
        scaled, saved.numpy(), rtol=2e-4, atol=2e-4
    ):
        raise RuntimeError(name + ": CSV flow features do not align with saved graph.")
    print(name, "flow-feature alignment verified.")

verify_edges("train", train_query_df, train_edges)
verify_edges("validation", val_df, val_edges)
verify_edges("benign test", benign_df, benign_edges)
verify_edges("PortScan test", attack_df, attack_edges)
verify_flow("train", train_query_df, train_flow)
verify_flow("validation", val_df, val_flow)
verify_flow("benign test", benign_df, benign_flow)
verify_flow("PortScan test", attack_df, attack_flow)


# 5. STRICTLY CAUSAL TEMPORAL FEATURES
# A flow in minute M receives statistics only from completed minute M-1.
# Five-minute features use the previous completed five-minute bucket.
# All flows sharing the current minute are excluded from their own features.


def _prepare_temporal_frame(df):
    w = df[["Source IP", "Destination IP", "_time", "Source Port", "Destination Port", "SYN Flag Count"]].copy()
    w["_minute"] = (w["_time"].astype("int64") // 60_000_000_000).astype(np.int64)
    w["_five"] = (w["_minute"] // 5).astype(np.int64)
    w["_srcport"] = pd.to_numeric(w["Source Port"], errors="coerce").fillna(-1).astype(np.int64)
    w["_dstport"] = pd.to_numeric(w["Destination Port"], errors="coerce").fillna(-1).astype(np.int64)
    w["_syn"] = pd.to_numeric(w["SYN Flag Count"], errors="coerce").fillna(0).clip(lower=0)
    return w


def _entropy_by_source_minute(w):
    keys = ["Source IP", "_minute"]
    counts = w.groupby(keys + ["_dstport"], sort=False, observed=True).size().rename("n").reset_index()
    totals = counts.groupby(keys, sort=False, observed=True)["n"].transform("sum").to_numpy(dtype=np.float64)
    probs = counts["n"].to_numpy(dtype=np.float64) / np.maximum(totals, 1.0)
    counts["_entropy_part"] = -probs * np.log2(np.maximum(probs, 1e-12))
    return counts.groupby(keys, sort=False, observed=True)["_entropy_part"].sum().rename("src_destination_port_entropy_1m").reset_index()


def causal_temporal_features(history_df, query_df, name):
    """For each query flow, use only aggregates from earlier completed buckets.

    history_df must contain only observations chronologically prior to query_df.
    query_df may contain mixed benign/attack flows; labels are not read.
    """
    print("Deriving causal temporal features:", name)
    q = query_df.copy().reset_index(drop=True)
    q["_qid"] = np.arange(len(q), dtype=np.int64)
    h = history_df.copy()
    if len(h) and len(q) and h["_time"].max() > q["_time"].min():
        raise ValueError(name + ": history contains timestamps later than query; refusing temporal leakage.")
    combined = pd.concat([h, q], ignore_index=True, copy=False)
    w = _prepare_temporal_frame(combined)

    # Per-source completed one-minute aggregates.
    src1 = w.groupby(["Source IP", "_minute"], sort=False, observed=True).agg(
        src_flows_1m=("Destination IP", "size"),
        src_unique_dst_ip_1m=("Destination IP", "nunique"),
        src_unique_dst_port_1m=("_dstport", "nunique"),
        src_syn_count_1m=("_syn", "sum"),
    ).reset_index()
    src1["src_syn_ratio_1m"] = src1["src_syn_count_1m"] / src1["src_flows_1m"].clip(lower=1)
    src1 = src1.merge(_entropy_by_source_minute(w), on=["Source IP", "_minute"], how="left")
    src1["_src_target_minute"] = src1["_minute"] + 1
    src1 = src1.drop(columns=["_minute"]).rename(columns={
        "src_flows_1m": "src_flows_1m", "src_unique_dst_ip_1m": "src_unique_dst_ip_1m",
        "src_unique_dst_port_1m": "src_unique_dst_port_1m"
    })

    # Per-source previous completed five-minute bucket.
    src5 = w.groupby(["Source IP", "_five"], sort=False, observed=True).agg(
        src_flows_5m=("Destination IP", "size"),
        src_unique_dst_ip_5m=("Destination IP", "nunique"),
        src_unique_dst_port_5m=("_dstport", "nunique"),
    ).reset_index()
    src5["_src_target_five"] = src5["_five"] + 1
    src5 = src5.drop(columns=["_five"])

    # Per-destination previous completed one-minute aggregates.
    dst1 = w.groupby(["Destination IP", "_minute"], sort=False, observed=True).agg(
        dst_flows_1m=("Source IP", "size"),
        dst_unique_src_ip_1m=("Source IP", "nunique"),
        dst_unique_src_port_1m=("_srcport", "nunique"),
        dst_syn_count_1m=("_syn", "sum"),
    ).reset_index()
    dst1["_dst_target_minute"] = dst1["_minute"] + 1
    dst1 = dst1.drop(columns=["_minute"])

    # Crucial pair-level port-scan features: one source probing many ports
    # on the same destination host.
    pair1 = w.groupby(["Source IP", "Destination IP", "_minute"], sort=False, observed=True).agg(
        pair_flows_prev_1m=("_dstport", "size"),
        pair_unique_dst_ports_prev_1m=("_dstport", "nunique"),
    ).reset_index()
    pair1["_pair_target_minute"] = pair1["_minute"] + 1
    pair1 = pair1.drop(columns=["_minute"])

    pair5 = w.groupby(["Source IP", "Destination IP", "_five"], sort=False, observed=True).agg(
        pair_unique_dst_ports_prev_5m=("_dstport", "nunique"),
    ).reset_index()
    pair5["_pair_target_five"] = pair5["_five"] + 1
    pair5 = pair5.drop(columns=["_five"])

    out = q[["_qid", "Source IP", "Destination IP", "_time"]].copy()
    out["_minute"] = (out["_time"].astype("int64") // 60_000_000_000).astype(np.int64)
    out["_five"] = (out["_minute"] // 5).astype(np.int64)
    out = out.merge(src1, left_on=["Source IP", "_minute"], right_on=["Source IP", "_src_target_minute"], how="left", sort=False)
    out = out.merge(src5, left_on=["Source IP", "_five"], right_on=["Source IP", "_src_target_five"], how="left", sort=False)
    out = out.merge(dst1, left_on=["Destination IP", "_minute"], right_on=["Destination IP", "_dst_target_minute"], how="left", sort=False)
    out = out.merge(pair1, left_on=["Source IP", "Destination IP", "_minute"], right_on=["Source IP", "Destination IP", "_pair_target_minute"], how="left", sort=False)
    out = out.merge(pair5, left_on=["Source IP", "Destination IP", "_five"], right_on=["Source IP", "Destination IP", "_pair_target_five"], how="left", sort=False)
    out = out.sort_values("_qid", kind="stable")
    arr = out[TEMP_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    if len(arr) != len(query_df) or not np.isfinite(arr).all():
        raise RuntimeError(name + ": causal temporal feature generation failed.")
    print("Causal temporal shape:", arr.shape,
          "| max prior pair ports/min:", int(arr[:, TEMP_COLS.index("pair_unique_dst_ports_prev_1m")].max()),
          "| max prior source ports/min:", int(arr[:, TEMP_COLS.index("src_unique_dst_port_1m")].max()))
    return arr


# Training features: each training flow sees only earlier completed buckets
# from the benign training period. Preserve the exact graph query-edge order.
train_temp_all = causal_temporal_features(train_df.iloc[0:0], train_df, "benign_train")
train_temp = train_temp_all[query_indices]

# Validation sees the complete earlier benign training period plus earlier
# completed validation buckets. No validation flow sees its own/current bucket.
val_temp = causal_temporal_features(train_df, val_df, "benign_validation")

# Test is evaluated as one chronological traffic stream. Benign and PortScan
# records are combined without reading labels, so prior observed traffic can
# contribute to later flow history. Both test classes remain untouched for scoring.
test_stream = pd.concat([benign_df, attack_df], ignore_index=True, copy=False)
test_stream["_original_test_row"] = np.arange(len(test_stream), dtype=np.int64)
test_stream = test_stream.sort_values("_time", kind="stable").reset_index(drop=True)
test_temp_sorted = causal_temporal_features(
    pd.concat([train_df, val_df], ignore_index=True, copy=False),
    test_stream,
    "combined_test_stream"
)
test_temp_stream = np.empty_like(test_temp_sorted)
test_temp_stream[test_stream["_original_test_row"].to_numpy(dtype=np.int64)] = test_temp_sorted
benign_temp = test_temp_stream[:len(benign_df)]
attack_temp = test_temp_stream[len(benign_df):]


def log_transform(a):
    return np.log1p(np.maximum(a, 0)).astype(np.float32)

temp_scaler = StandardScaler()
temp_scaler.fit(log_transform(train_temp_all))
train_temp_scaled = np.clip(temp_scaler.transform(log_transform(train_temp)), -10, 10).astype(np.float32)
val_temp_scaled = np.clip(temp_scaler.transform(log_transform(val_temp)), -10, 10).astype(np.float32)
benign_temp_scaled = np.clip(temp_scaler.transform(log_transform(benign_temp)), -10, 10).astype(np.float32)
attack_temp_scaled = np.clip(temp_scaler.transform(log_transform(attack_temp)), -10, 10).astype(np.float32)


# 6. TRANSPARENT SCAN BEHAVIOUR SCORE

def scan_rule(a):
    d = pd.DataFrame(a, columns=TEMP_COLS)
    return (
        0.16 * np.log1p(d["src_flows_1m"].to_numpy())
        + 0.12 * np.log1p(d["src_unique_dst_ip_1m"].to_numpy())
        + 0.18 * np.log1p(d["src_unique_dst_port_1m"].to_numpy())
        + 0.10 * np.log1p(d["src_unique_dst_port_5m"].to_numpy())
        + 0.08 * d["src_syn_ratio_1m"].to_numpy()
        + 0.06 * np.log1p(d["dst_unique_src_ip_1m"].to_numpy())
        + 0.18 * np.log1p(d["pair_unique_dst_ports_prev_1m"].to_numpy())
        + 0.12 * np.log1p(d["pair_unique_dst_ports_prev_5m"].to_numpy())
    ).astype(np.float64)

train_rule = scan_rule(train_temp)
val_rule = scan_rule(val_temp)
benign_rule = scan_rule(benign_temp)
attack_rule = scan_rule(attack_temp)


# 7. BENIGN-ONLY TEMPORAL ISOLATION FOREST

print("Fitting benign-only temporal Isolation Forest...")
iforest = IsolationForest(
    n_estimators=IFOREST_TREES,
    max_samples=min(IFOREST_MAX_SAMPLES, len(train_temp_scaled)),
    contamination="auto",
    random_state=SEED,
    n_jobs=-1,
)
iforest.fit(train_temp_scaled)

def iforest_score(a, batch=50000):
    chunks = []
    for start in range(0, len(a), batch):
        chunks.append(-iforest.decision_function(a[start:start + batch]))
    return np.concatenate(chunks).astype(np.float64)

val_if = iforest_score(val_temp_scaled)
benign_if = iforest_score(benign_temp_scaled)
attack_if = iforest_score(attack_temp_scaled)


# 8. TEMPORAL GNN RECONSTRUCTION MODEL

x = x_cpu.to(DEVICE)
context_edges = context_edges_cpu.to(DEVICE)

class TemporalGCN(nn.Module):
    def __init__(self, node_dim, hidden, output_dim):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim * 2),
        )
        self.output_dim = output_dim

    def encode(self, features, edge_index):
        h = self.relu(self.conv1(features, edge_index))
        h = self.dropout(h)
        return self.relu(self.conv2(h, edge_index))

    def decode(self, h, edge_index):
        s, d = h[edge_index[0]], h[edge_index[1]]
        z = torch.cat([s, d, torch.abs(s - d), s * d], dim=1)
        out = self.decoder(z)
        mu = out[:, :self.output_dim]
        logvar = torch.clamp(out[:, self.output_dim:], MIN_LOGVAR, MAX_LOGVAR)
        return mu, logvar

def nll_per_flow(mu, logvar, target):
    err2 = (target - mu) ** 2
    return (0.5 * (logvar + err2 * torch.exp(-logvar))).mean(dim=1)

OUTPUT_DIM = FLOW_DIM + TEMP_DIM
model = TemporalGCN(NODE_DIM, HIDDEN, OUTPUT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

train_targets = torch.cat([train_flow, torch.from_numpy(train_temp_scaled)], dim=1).contiguous()
val_targets = torch.cat([val_flow, torch.from_numpy(val_temp_scaled)], dim=1).contiguous()
benign_targets = torch.cat([benign_flow, torch.from_numpy(benign_temp_scaled)], dim=1).contiguous()
attack_targets = torch.cat([attack_flow, torch.from_numpy(attack_temp_scaled)], dim=1).contiguous()

print("GNN trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Reconstruction dimensions:", OUTPUT_DIM)

def train_epoch():
    model.train()
    optimizer.zero_grad(set_to_none=True)
    h = model.encode(x, context_edges)
    n = train_edges.shape[1]
    batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    total = 0.0

    for bi, start in enumerate(range(0, n, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, n)
        ei = train_edges[:, start:end].to(DEVICE)
        target = train_targets[start:end].to(DEVICE)
        mu, lv = model.decode(h, ei)
        losses = nll_per_flow(mu, lv, target)
        loss = losses.mean() * ((end - start) / n)
        loss.backward(retain_graph=(bi < batches - 1))
        total += float(losses.detach().sum().item())
        del ei, target, mu, lv, losses, loss

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    del h
    return total / n

@torch.no_grad()
def gnn_scores(edge_index, targets):
    model.eval()
    h = model.encode(x, context_edges)
    out = []
    for start in range(0, edge_index.shape[1], BATCH_SIZE):
        end = min(start + BATCH_SIZE, edge_index.shape[1])
        ei = edge_index[:, start:end].to(DEVICE)
        target = targets[start:end].to(DEVICE)
        mu, lv = model.decode(h, ei)
        out.append(nll_per_flow(mu, lv, target).cpu().numpy())
        del ei, target, mu, lv
    del h
    return np.concatenate(out).astype(np.float64)


# 9. TRAIN AND SAVE A DISTINCT V6 CHECKPOINT

checkpoint_path = MODELS / "zero_day_gnn_v6_best.pth"
history = []
best_val = float("inf")
best_epoch = 0
wait = 0

print("\nStarting benign-only V6 training...")
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch()
    validation_score = gnn_scores(val_edges, val_targets)
    validation_loss = float(validation_score.mean())
    history.append({
        "epoch": epoch,
        "training_nll": train_loss,
        "validation_nll": validation_loss,
    })
    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train NLL: {train_loss:.6f} Validation NLL: {validation_loss:.6f}"
    )

    if validation_loss < best_val:
        best_val = validation_loss
        best_epoch = epoch
        wait = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_validation_nll": best_val,
            "node_features": NODE_DIM,
            "flow_features": FLOW_DIM,
            "temporal_features": TEMP_DIM,
            "temporal_columns": TEMP_COLS,
            "hidden_channels": HIDDEN,
            "seed": SEED,
        }, checkpoint_path)
    else:
        wait += 1

    if wait >= PATIENCE:
        print("Early stopping.")
        break
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

history_path = REPORTS / "zero_day_v6_training_history.csv"
pd.DataFrame(history).to_csv(history_path, index=False)

checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Best epoch:", best_epoch, "| best validation NLL:", best_val)


# 10. CALIBRATE COMBINED SCORE ON BENIGN VALIDATION ONLY

print("Scoring benign validation...")
val_gnn = gnn_scores(val_edges, val_targets)

def percentile_rank(values, reference):
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    vals = np.asarray(values, dtype=np.float64)
    return np.searchsorted(ref, vals, side="right") / max(len(ref), 1)

def combine(gnn, iso, rule, refs):
    rg = percentile_rank(gnn, refs["gnn"])
    ri = percentile_rank(iso, refs["if"])
    rr = percentile_rank(rule, refs["rule"])
    score = W_GNN * rg + W_IF * ri + W_RULE * rr
    return score, rg, ri, rr

refs = {"gnn": val_gnn, "if": val_if, "rule": val_rule}
val_combined, _, _, _ = combine(val_gnn, val_if, val_rule, refs)
threshold = float(np.percentile(val_combined, THRESHOLD_PERCENTILE))

print("Scoring benign test...")
benign_gnn = gnn_scores(benign_edges, benign_targets)
benign_combined, benign_rg, benign_ri, benign_rr = combine(
    benign_gnn, benign_if, benign_rule, refs
)

print("Scoring held-out PortScan...")
attack_gnn = gnn_scores(attack_edges, attack_targets)
attack_combined, attack_rg, attack_ri, attack_rr = combine(
    attack_gnn, attack_if, attack_rule, refs
)

benign_pred = (benign_combined > threshold).astype(np.int32)
attack_pred = (attack_combined > threshold).astype(np.int32)
y_true = np.concatenate([
    np.zeros(len(benign_pred), dtype=np.int32),
    np.ones(len(attack_pred), dtype=np.int32),
])
y_pred = np.concatenate([benign_pred, attack_pred])
all_scores = np.concatenate([benign_combined, attack_combined])

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

accuracy = float(accuracy_score(y_true, y_pred))
precision = float(precision_score(y_true, y_pred, zero_division=0))
recall = float(recall_score(y_true, y_pred, zero_division=0))
f1 = float(f1_score(y_true, y_pred, zero_division=0))
fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
benign_fpr = float(benign_pred.mean())
attack_tpr = float(attack_pred.mean())

try:
    roc_auc = float(roc_auc_score(y_true, all_scores))
except ValueError:
    roc_auc = None
try:
    pr_auc = float(average_precision_score(y_true, all_scores))
except ValueError:
    pr_auc = None

threshold_tradeoff = {}
for pct in [95.0, 97.5, 99.0, 99.5, 99.9]:
    th = float(np.percentile(val_combined, pct))
    threshold_tradeoff[str(pct)] = {
        "validation_percentile": pct,
        "threshold": th,
        "benign_test_fpr": float(np.mean(benign_combined > th)),
        "portscan_detection_rate": float(np.mean(attack_combined > th)),
    }


# 11. SAVE ALL V6 OUTPUTS WITHOUT OVERWRITING OLDER VERSIONS

temporal_scaler_path = MODELS / "zero_day_v6_temporal_scaler.pkl"
iforest_path = MODELS / "zero_day_v6_temporal_iforest.pkl"
joblib.dump(temp_scaler, temporal_scaler_path)
joblib.dump(iforest, iforest_path)

metrics = {
    "model_version": "V6",
    "architecture": "GCN flow/causal-temporal reconstruction + benign-only temporal Isolation Forest + source-destination pair port-scan score",
    "training_type": "Benign-only",
    "attack_labels_used_for_training": False,
    "best_epoch": int(best_epoch),
    "best_validation_nll": float(best_val),
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "threshold": threshold,
    "weights": {"gnn": W_GNN, "temporal_iforest": W_IF, "scan_rule": W_RULE},
    "temporal_features": TEMP_COLS,
    "temporal_semantics": "previous completed minute and previous completed five-minute bucket; current bucket excluded",
    "benign_test_flows": int(len(benign_pred)),
    "portscan_test_flows": int(len(attack_pred)),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "false_positive_rate": fpr,
    "false_negative_rate": fnr,
    "benign_test_false_positive_rate": benign_fpr,
    "portscan_detection_rate": attack_tpr,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "confusion_matrix": cm.tolist(),
    "threshold_tradeoff": threshold_tradeoff,
    "limitations": (
        "PortScan is excluded from fitting and threshold calibration. Host mapping is transductive. "
        "Temporal features use previous completed minute/five-minute buckets only. "
        "This is causal bucketed evaluation, not sub-minute event-by-event streaming. "
        "Host mapping remains transductive; PortScan is one held-out attack category."
    ),
}

metrics_path = REPORTS / "zero_day_v6_evaluation_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

benign_out = pd.DataFrame({
    "Actual_Label": "BENIGN",
    "GNN_NLL": benign_gnn,
    "GNN_Rank": benign_rg,
    "Temporal_IForest_Score": benign_if,
    "Temporal_IForest_Rank": benign_ri,
    "Scan_Behaviour_Score": benign_rule,
    "Scan_Behaviour_Rank": benign_rr,
    "Combined_Anomaly_Score": benign_combined,
    "Predicted_Label": np.where(benign_pred == 1, "ANOMALY", "NORMAL"),
})
attack_out = pd.DataFrame({
    "Actual_Label": "PORTSCAN",
    "GNN_NLL": attack_gnn,
    "GNN_Rank": attack_rg,
    "Temporal_IForest_Score": attack_if,
    "Temporal_IForest_Rank": attack_ri,
    "Scan_Behaviour_Score": attack_rule,
    "Scan_Behaviour_Rank": attack_rr,
    "Combined_Anomaly_Score": attack_combined,
    "Predicted_Label": np.where(attack_pred == 1, "ANOMALY", "NORMAL"),
})
predictions_path = REPORTS / "zero_day_v6_flow_predictions.csv"
pd.concat([benign_out, attack_out], ignore_index=True).to_csv(predictions_path, index=False)

calibration_path = REPORTS / "zero_day_v6_calibration.json"
with open(calibration_path, "w", encoding="utf-8") as f:
    json.dump({
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold": threshold,
        "validation_component_99th_percentiles": {
            "gnn": float(np.percentile(val_gnn, 99)),
            "iforest": float(np.percentile(val_if, 99)),
            "scan_rule": float(np.percentile(val_rule, 99)),
        },
        "temporal_scaler": str(temporal_scaler_path),
        "isolation_forest": str(iforest_path),
    }, f, indent=4)

print("\n" + "=" * 76)
print("ZERO-DAY GNN V6 RESULTS")
print("=" * 76)
print("Best epoch:", best_epoch)
print("Threshold:", round(threshold, 6))
print("Accuracy:", round(accuracy, 6))
print("Precision:", round(precision, 6))
print("Recall:", round(recall, 6))
print("F1:", round(f1, 6))
print("Benign test FPR:", round(benign_fpr, 6))
print("PortScan detection rate:", round(attack_tpr, 6))
print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)
print("Confusion matrix:\n", cm)
print("\nThreshold trade-off:")
for pct, item in threshold_tradeoff.items():
    print(
        f"{pct}% validation threshold | benign FPR "
        f"{item['benign_test_fpr']:.4%} | PortScan detection "
        f"{item['portscan_detection_rate']:.4%}"
    )
print("\nSaved:")
for p in [
    checkpoint_path, history_path, metrics_path, predictions_path,
    calibration_path, temporal_scaler_path, iforest_path
]:
    print(p)
print("\nV6 training and evaluation completed.")
print("=" * 76)

