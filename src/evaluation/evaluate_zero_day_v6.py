"""
V6 publication evaluation (NO RETRAINING).

Reconstructs the V6 causal temporal features exactly as in train_zero_day_gnn_v6.py,
loads the saved V6 checkpoint/scaler/Isolation Forest, verifies graph alignment,
calibrates component and combined thresholds on benign validation only, and evaluates
the untouched benign-test + PortScan-test stream.

Outputs are written to reports/zero_day_v6_publication_evaluation/.
"""

from pathlib import Path
import json
import warnings
import gc

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "zero_day"
GRAPHS = DATA / "graphs"
MODELS = ROOT / "models" / "portscan_v6"
REPORT_DIR = ROOT / "reports" / "zero_day" / "portscan_v6" / "publication"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
MIN_LOGVAR, MAX_LOGVAR = -6.0, 4.0
W_GNN, W_IF, W_RULE = 0.35, 0.40, 0.25
CALIBRATION_PERCENTILE = 99.0

FLOW_COLS = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Packet Length Mean",
    "Packet Length Std", "SYN Flag Count", "ACK Flag Count",
    "RST Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "Active Mean", "Idle Mean"
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
CSV_COLS = [
    "Flow ID", "Source IP", "Destination IP", "Timestamp", "Verified_Timestamp"
] + FLOW_COLS


def load_graph(filename):
    path = GRAPHS / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing graph artifact: {path}")
    print("Loading graph:", filename)
    return torch.load(path, map_location="cpu", weights_only=False)


def read_split(filename):
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared dataset: {path}")
    df = pd.read_csv(
        path, usecols=list(dict.fromkeys(CSV_COLS)), low_memory=False
    )
    for col in ["Source IP", "Destination IP"]:
        df[col] = df[col].astype("string").str.strip()

    parts = df["Flow ID"].astype("string").str.split("-", expand=True)
    if parts.shape[1] != 5:
        raise ValueError(
            f"{filename}: expected 5 Flow ID fields; examples: "
            f"{df['Flow ID'].head(5).tolist()}"
        )
    for col, part_idx in [
        ("Source Port", 2), ("Destination Port", 3), ("Protocol", 4)
    ]:
        values = pd.to_numeric(parts[part_idx], errors="coerce")
        if values.isna().any():
            examples = df.loc[values.isna(), "Flow ID"].head(5).tolist()
            raise ValueError(f"{filename}: invalid {col} in Flow ID: {examples}")
        df[col] = values.astype(np.int32)

    time_col = "Verified_Timestamp" if "Verified_Timestamp" in df.columns else "Timestamp"
    df["_time"] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    if df[["Source IP", "Destination IP", "_time"]].isna().any().any():
        raise ValueError(f"{filename}: missing IP or invalid timestamp.")
    for col in FLOW_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FLOW_COLS] = df[FLOW_COLS].replace([np.inf, -np.inf], np.nan)
    df.reset_index(drop=True, inplace=True)
    print(f"{filename}: {len(df):,} rows")
    return df


def make_edges(df, host_map, label):
    src = df["Source IP"].map(host_map)
    dst = df["Destination IP"].map(host_map)
    if src.isna().any() or dst.isna().any():
        raise ValueError(f"{label}: IP missing from saved host mapping.")
    return torch.from_numpy(np.vstack([
        src.to_numpy(dtype=np.int64), dst.to_numpy(dtype=np.int64)
    ])).long()


def verify_edges(label, df, saved_edges, host_map):
    rebuilt = make_edges(df, host_map, label)
    if rebuilt.shape != saved_edges.shape or not torch.equal(rebuilt, saved_edges):
        raise RuntimeError(
            f"{label}: CSV order does not match saved graph edge order. "
            "Stopping to prevent incorrect row-to-edge alignment."
        )
    print(label, "edge alignment verified.")


def verify_flow(label, df, saved_flow, imputer, flow_scaler):
    raw = df[FLOW_COLS].to_numpy(dtype=np.float32)
    scaled = flow_scaler.transform(imputer.transform(raw)).astype(np.float32)
    if scaled.shape != tuple(saved_flow.shape) or not np.allclose(
        scaled, saved_flow.numpy(), rtol=2e-4, atol=2e-4
    ):
        raise RuntimeError(f"{label}: flow-feature alignment check failed.")
    print(label, "flow-feature alignment verified.")


# ---- Exact V6 causal temporal feature implementation ----

def prepare_temporal_frame(df):
    w = df[[
        "Source IP", "Destination IP", "_time", "Source Port",
        "Destination Port", "SYN Flag Count"
    ]].copy()
    w["_minute"] = (w["_time"].astype("int64") // 60_000_000_000).astype(np.int64)
    w["_five"] = (w["_minute"] // 5).astype(np.int64)
    w["_srcport"] = pd.to_numeric(w["Source Port"], errors="coerce").fillna(-1).astype(np.int64)
    w["_dstport"] = pd.to_numeric(w["Destination Port"], errors="coerce").fillna(-1).astype(np.int64)
    w["_syn"] = pd.to_numeric(w["SYN Flag Count"], errors="coerce").fillna(0).clip(lower=0)
    return w


def entropy_by_source_minute(w):
    keys = ["Source IP", "_minute"]
    counts = (
        w.groupby(keys + ["_dstport"], sort=False, observed=True)
        .size().rename("n").reset_index()
    )
    totals = counts.groupby(keys, sort=False, observed=True)["n"].transform("sum").to_numpy(dtype=np.float64)
    probs = counts["n"].to_numpy(dtype=np.float64) / np.maximum(totals, 1.0)
    counts["_entropy_part"] = -probs * np.log2(np.maximum(probs, 1e-12))
    return (
        counts.groupby(keys, sort=False, observed=True)["_entropy_part"]
        .sum().rename("src_destination_port_entropy_1m").reset_index()
    )


def causal_temporal_features(history_df, query_df, name):
    print("Creating causal temporal features:", name)
    q = query_df.copy().reset_index(drop=True)
    q["_qid"] = np.arange(len(q), dtype=np.int64)
    h = history_df.copy()

    if len(h) and len(q) and h["_time"].max() > q["_time"].min():
        raise ValueError(
            f"{name}: history contains timestamps later than query; "
            "refusing temporal leakage."
        )

    combined = pd.concat([h, q], ignore_index=True, copy=False)
    w = prepare_temporal_frame(combined)

    src1 = w.groupby(
        ["Source IP", "_minute"], sort=False, observed=True
    ).agg(
        src_flows_1m=("Destination IP", "size"),
        src_unique_dst_ip_1m=("Destination IP", "nunique"),
        src_unique_dst_port_1m=("_dstport", "nunique"),
        src_syn_count_1m=("_syn", "sum"),
    ).reset_index()
    src1["src_syn_ratio_1m"] = (
        src1["src_syn_count_1m"] / src1["src_flows_1m"].clip(lower=1)
    )
    src1 = src1.merge(
        entropy_by_source_minute(w),
        on=["Source IP", "_minute"], how="left"
    )
    src1["_src_target_minute"] = src1["_minute"] + 1
    src1 = src1.drop(columns=["_minute"])

    src5 = w.groupby(
        ["Source IP", "_five"], sort=False, observed=True
    ).agg(
        src_flows_5m=("Destination IP", "size"),
        src_unique_dst_ip_5m=("Destination IP", "nunique"),
        src_unique_dst_port_5m=("_dstport", "nunique"),
    ).reset_index()
    src5["_src_target_five"] = src5["_five"] + 1
    src5 = src5.drop(columns=["_five"])

    dst1 = w.groupby(
        ["Destination IP", "_minute"], sort=False, observed=True
    ).agg(
        dst_flows_1m=("Source IP", "size"),
        dst_unique_src_ip_1m=("Source IP", "nunique"),
        dst_unique_src_port_1m=("_srcport", "nunique"),
        dst_syn_count_1m=("_syn", "sum"),
    ).reset_index()
    dst1["_dst_target_minute"] = dst1["_minute"] + 1
    dst1 = dst1.drop(columns=["_minute"])

    pair1 = w.groupby(
        ["Source IP", "Destination IP", "_minute"],
        sort=False, observed=True
    ).agg(
        pair_flows_prev_1m=("_dstport", "size"),
        pair_unique_dst_ports_prev_1m=("_dstport", "nunique"),
    ).reset_index()
    pair1["_pair_target_minute"] = pair1["_minute"] + 1
    pair1 = pair1.drop(columns=["_minute"])

    pair5 = w.groupby(
        ["Source IP", "Destination IP", "_five"],
        sort=False, observed=True
    ).agg(
        pair_unique_dst_ports_prev_5m=("_dstport", "nunique"),
    ).reset_index()
    pair5["_pair_target_five"] = pair5["_five"] + 1
    pair5 = pair5.drop(columns=["_five"])

    out = q[["_qid", "Source IP", "Destination IP", "_time"]].copy()
    out["_minute"] = (out["_time"].astype("int64") // 60_000_000_000).astype(np.int64)
    out["_five"] = (out["_minute"] // 5).astype(np.int64)
    out = out.merge(
        src1, left_on=["Source IP", "_minute"],
        right_on=["Source IP", "_src_target_minute"], how="left", sort=False
    )
    out = out.merge(
        src5, left_on=["Source IP", "_five"],
        right_on=["Source IP", "_src_target_five"], how="left", sort=False
    )
    out = out.merge(
        dst1, left_on=["Destination IP", "_minute"],
        right_on=["Destination IP", "_dst_target_minute"], how="left", sort=False
    )
    out = out.merge(
        pair1, left_on=["Source IP", "Destination IP", "_minute"],
        right_on=["Source IP", "Destination IP", "_pair_target_minute"],
        how="left", sort=False
    )
    out = out.merge(
        pair5, left_on=["Source IP", "Destination IP", "_five"],
        right_on=["Source IP", "Destination IP", "_pair_target_five"],
        how="left", sort=False
    )
    out = out.sort_values("_qid", kind="stable")
    arr = (
        out[TEMP_COLS].replace([np.inf, -np.inf], np.nan)
        .fillna(0.0).to_numpy(dtype=np.float32)
    )
    if len(arr) != len(query_df) or not np.isfinite(arr).all():
        raise RuntimeError(f"{name}: causal temporal feature generation failed.")
    print("Temporal shape:", arr.shape)
    del combined, w, out
    gc.collect()
    return arr


def log_transform(a):
    return np.log1p(np.maximum(a, 0)).astype(np.float32)


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


def iforest_scores(model, values, batch=50000):
    chunks = []
    for start in range(0, len(values), batch):
        chunks.append(-model.decision_function(values[start:start + batch]))
    return np.concatenate(chunks).astype(np.float64)


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


@torch.no_grad()
def score_gnn(model, x, context_edges, query_edges, targets):
    model.eval()
    h = model.encode(x, context_edges)
    scores = []
    for start in range(0, query_edges.shape[1], BATCH_SIZE):
        end = min(start + BATCH_SIZE, query_edges.shape[1])
        edge_batch = query_edges[:, start:end].to(DEVICE)
        target_batch = targets[start:end].to(DEVICE)
        mu, logvar = model.decode(h, edge_batch)
        scores.append(nll_per_flow(mu, logvar, target_batch).cpu().numpy())
        del edge_batch, target_batch, mu, logvar
    del h
    return np.concatenate(scores).astype(np.float64)


def percentile_rank(values, reference):
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    vals = np.asarray(values, dtype=np.float64)
    return np.searchsorted(ref, vals, side="right") / max(len(ref), 1)


def combined_score(gnn, iso, rule, refs):
    rg = percentile_rank(gnn, refs["gnn"])
    ri = percentile_rank(iso, refs["if"])
    rr = percentile_rank(rule, refs["rule"])
    score = W_GNN * rg + W_IF * ri + W_RULE * rr
    return score, rg, ri, rr


def evaluate_at_threshold(y_true, scores, threshold):
    y_pred = (scores > threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "benign_fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "portscan_detection_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": cm.tolist(),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        result["roc_auc"] = None
    try:
        result["pr_auc"] = float(average_precision_score(y_true, scores))
    except ValueError:
        result["pr_auc"] = None
    return result


def main():
    print("=" * 78)
    print("V6 PUBLICATION EVALUATION â€” NO RETRAINING")
    print("=" * 78)
    print("Device:", DEVICE)

    checkpoint_path = MODELS / "zero_day_gnn_v6_best.pth"
    temp_scaler_path = MODELS / "zero_day_v6_temporal_scaler.pkl"
    iforest_path = MODELS / "zero_day_v6_temporal_iforest.pkl"
    for p in [checkpoint_path, temp_scaler_path, iforest_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required V6 artifact missing: {p}")

    context = load_graph("benign_context_graph.pt")
    train_graph = load_graph("benign_train_graph.pt")
    val_graph = load_graph("benign_validation_graph.pt")
    benign_graph = load_graph("benign_test_graph.pt")
    attack_graph = load_graph("portscan_test_graph.pt")

    x_cpu = context["node_features"].float().contiguous()
    context_edges_cpu = context["edge_index"].long().contiguous()
    val_edges = val_graph["query_edge_index"].long().contiguous()
    val_flow = val_graph["query_edge_attr"].float().contiguous()
    benign_edges = benign_graph["query_edge_index"].long().contiguous()
    benign_flow = benign_graph["query_edge_attr"].float().contiguous()
    attack_edges = attack_graph["query_edge_index"].long().contiguous()
    attack_flow = attack_graph["query_edge_attr"].float().contiguous()

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if checkpoint.get("temporal_features") != len(TEMP_COLS):
        raise ValueError("Checkpoint temporal feature dimension does not match this evaluator.")
    if checkpoint.get("temporal_columns") != TEMP_COLS:
        raise ValueError("Checkpoint temporal feature names/order do not match evaluator.")
    model = TemporalGCN(
        node_dim=int(checkpoint["node_features"]),
        hidden=int(checkpoint["hidden_channels"]),
        output_dim=int(checkpoint["flow_features"]) + int(checkpoint["temporal_features"]),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("V6 checkpoint loaded. Best epoch:", checkpoint.get("best_epoch"))

    with open(GRAPHS / "host_mapping.json", encoding="utf-8") as f:
        host_map = {str(k): int(v) for k, v in json.load(f).items()}
    imputer = joblib.load(MODELS / "zero_day_imputer.pkl")
    flow_scaler = joblib.load(MODELS / "zero_day_scaler.pkl")
    temp_scaler = joblib.load(temp_scaler_path)
    iforest = joblib.load(iforest_path)

    train_df = read_split("benign_train.csv")
    val_df = read_split("benign_validation.csv")
    benign_df = read_split("benign_test.csv")
    attack_df = read_split("portscan_test.csv")

    # Recreate the exact V6 benign-train query subset and verify its saved edges.
    rng = np.random.default_rng(42)
    permutation = rng.permutation(len(train_df))
    context_size = int(len(train_df) * 0.80)
    query_indices = permutation[context_size:]
    train_query_df = train_df.iloc[query_indices].copy()

    verify_edges("validation", val_df, val_edges, host_map)
    verify_edges("benign test", benign_df, benign_edges, host_map)
    verify_edges("PortScan test", attack_df, attack_edges, host_map)

    verify_flow("validation", val_df, val_flow, imputer, flow_scaler)
    verify_flow("benign test", benign_df, benign_flow, imputer, flow_scaler)
    verify_flow("PortScan test", attack_df, attack_flow, imputer, flow_scaler)

    # V6 scaler was fitted on causal temporal features for all benign_train rows.
    train_temp_all = causal_temporal_features(
        train_df.iloc[0:0], train_df, "benign_train_scaler_reconstruction"
    )
    val_temp = causal_temporal_features(train_df, val_df, "benign_validation")

    # Match V6's test protocol exactly: combine both test CSVs without labels,
    # sort chronologically, generate causal features, then restore original row order.
    benign_count = len(benign_df)
    attack_count = len(attack_df)
    test_stream = pd.concat([benign_df, attack_df], ignore_index=True, copy=False)
    test_stream["_original_test_row"] = np.arange(len(test_stream), dtype=np.int64)
    test_stream = test_stream.sort_values("_time", kind="stable").reset_index(drop=True)
    test_temp_sorted = causal_temporal_features(
        pd.concat([train_df, val_df], ignore_index=True, copy=False),
        test_stream,
        "combined_test_stream"
    )
    test_temp_original = np.empty_like(test_temp_sorted)
    test_temp_original[
        test_stream["_original_test_row"].to_numpy(dtype=np.int64)
    ] = test_temp_sorted
    benign_temp = test_temp_original[:benign_count]
    attack_temp = test_temp_original[benign_count:benign_count + attack_count]

    # Recreate V6's log transform, saved train-fitted scaler and clipping.
    val_temp_scaled = np.clip(
        temp_scaler.transform(log_transform(val_temp)), -10, 10
    ).astype(np.float32)
    benign_temp_scaled = np.clip(
        temp_scaler.transform(log_transform(benign_temp)), -10, 10
    ).astype(np.float32)
    attack_temp_scaled = np.clip(
        temp_scaler.transform(log_transform(attack_temp)), -10, 10
    ).astype(np.float32)

    val_rule = scan_rule(val_temp)
    benign_rule = scan_rule(benign_temp)
    attack_rule = scan_rule(attack_temp)

    val_temp_scaled = np.ascontiguousarray(val_temp_scaled)
    benign_temp_scaled = np.ascontiguousarray(benign_temp_scaled)
    attack_temp_scaled = np.ascontiguousarray(attack_temp_scaled)

    print("Scoring Isolation Forest...")
    val_if = iforest_scores(iforest, val_temp_scaled)
    benign_if = iforest_scores(iforest, benign_temp_scaled)
    attack_if = iforest_scores(iforest, attack_temp_scaled)

    # Reconstruct GNN targets using the saved graph flow features and scaled
    # temporal features in precisely the V6 feature order.
    val_targets = torch.cat([val_flow, torch.from_numpy(val_temp_scaled)], dim=1).contiguous()
    benign_targets = torch.cat([benign_flow, torch.from_numpy(benign_temp_scaled)], dim=1).contiguous()
    attack_targets = torch.cat([attack_flow, torch.from_numpy(attack_temp_scaled)], dim=1).contiguous()

    x = x_cpu.to(DEVICE)
    context_edges = context_edges_cpu.to(DEVICE)
    print("Scoring validation GNN...")
    val_gnn = score_gnn(model, x, context_edges, val_edges, val_targets)
    print("Scoring benign-test GNN...")
    benign_gnn = score_gnn(model, x, context_edges, benign_edges, benign_targets)
    print("Scoring PortScan-test GNN...")
    attack_gnn = score_gnn(model, x, context_edges, attack_edges, attack_targets)

    # All component reference distributions and all thresholds are validation-only.
    refs = {"gnn": val_gnn, "if": val_if, "rule": val_rule}
    val_combined, _, _, _ = combined_score(val_gnn, val_if, val_rule, refs)
    benign_combined, benign_rg, benign_ri, benign_rr = combined_score(
        benign_gnn, benign_if, benign_rule, refs
    )
    attack_combined, attack_rg, attack_ri, attack_rr = combined_score(
        attack_gnn, attack_if, attack_rule, refs
    )

    thresholds = {
        "GNN": float(np.percentile(val_gnn, CALIBRATION_PERCENTILE)),
        "Isolation_Forest": float(np.percentile(val_if, CALIBRATION_PERCENTILE)),
        "Scan_Behaviour": float(np.percentile(val_rule, CALIBRATION_PERCENTILE)),
        "Combined_V6": float(np.percentile(val_combined, CALIBRATION_PERCENTILE)),
    }
    print("\nFrozen benign-validation thresholds:")
    for name, value in thresholds.items():
        print(f"{name}: {value:.8f}")

    y_true = np.concatenate([
        np.zeros(benign_count, dtype=np.int32),
        np.ones(attack_count, dtype=np.int32),
    ])
    scores = {
        "GNN": np.concatenate([benign_gnn, attack_gnn]),
        "Isolation_Forest": np.concatenate([benign_if, attack_if]),
        "Scan_Behaviour": np.concatenate([benign_rule, attack_rule]),
        "Combined_V6": np.concatenate([benign_combined, attack_combined]),
    }

    results = {}
    for name, values in scores.items():
        result = evaluate_at_threshold(y_true, values, thresholds[name])
        results[name] = result
        print("\n", name)
        for key in [
            "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
            "benign_fpr", "portscan_detection_rate", "tn", "fp", "fn", "tp"
        ]:
            print(f"{key}: {result.get(key)}")
        print("Confusion matrix:", result["confusion_matrix"])

    # Combined score's percentile threshold trade-off, still calibrated on validation.
    tradeoff = {}
    for pct in [95.0, 97.5, 99.0, 99.5, 99.9]:
        th = float(np.percentile(val_combined, pct))
        test_pred = (scores["Combined_V6"] > th).astype(np.int32)
        cm = confusion_matrix(y_true, test_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        tradeoff[str(pct)] = {
            "validation_percentile": pct,
            "threshold": th,
            "benign_test_fpr": float(fp / (fp + tn)),
            "portscan_detection_rate": float(tp / (tp + fn)),
            "confusion_matrix": cm.tolist(),
        }

    # Save detailed per-flow output with original identifying fields.
    def prediction_frame(df, split_name, actual_label, gnn, gnn_rank,
                         if_score, if_rank, rule, rule_rank, combined, pred):
        out = df[[
            "Flow ID", "Source IP", "Source Port", "Destination IP",
            "Destination Port", "Protocol", "_time"
        ]].copy()
        out.insert(0, "Split", split_name)
        out["Actual_Label"] = actual_label
        out["GNN_NLL"] = gnn
        out["GNN_Rank"] = gnn_rank
        out["Isolation_Forest_Score"] = if_score
        out["Isolation_Forest_Rank"] = if_rank
        out["Scan_Behaviour_Score"] = rule
        out["Scan_Behaviour_Rank"] = rule_rank
        out["Combined_V6_Score"] = combined
        out["Predicted_Label"] = np.where(pred == 1, "ANOMALY", "NORMAL")
        out["Correct"] = (pred == (1 if actual_label == "PORTSCAN" else 0))
        return out

    benign_pred = (benign_combined > thresholds["Combined_V6"]).astype(np.int32)
    attack_pred = (attack_combined > thresholds["Combined_V6"]).astype(np.int32)
    benign_frame = prediction_frame(
        benign_df, "benign_test", "BENIGN", benign_gnn, benign_rg,
        benign_if, benign_ri, benign_rule, benign_rr,
        benign_combined, benign_pred
    )
    attack_frame = prediction_frame(
        attack_df, "portscan_test", "PORTSCAN", attack_gnn, attack_rg,
        attack_if, attack_ri, attack_rule, attack_rr,
        attack_combined, attack_pred
    )
    all_predictions = pd.concat([benign_frame, attack_frame], ignore_index=True)

    false_positives = benign_frame[benign_pred == 1].copy()
    missed_portscan = attack_frame[attack_pred == 0].copy()
    detected_portscan = attack_frame[attack_pred == 1].copy()

    # Component-level flags use each component's own 99th percentile on validation.
    val_component_thresholds = {
        "GNN_NLL": thresholds["GNN"],
        "Isolation_Forest_Score": thresholds["Isolation_Forest"],
        "Scan_Behaviour_Score": thresholds["Scan_Behaviour"],
    }
    missed_component_analysis = missed_portscan[[
        "Source IP", "Destination IP", "Source Port", "Destination Port",
        "Protocol", "_time", "GNN_NLL", "Isolation_Forest_Score",
        "Scan_Behaviour_Score", "Combined_V6_Score"
    ]].copy()
    missed_component_analysis["GNN_component_flag"] = (
        missed_component_analysis["GNN_NLL"] > thresholds["GNN"]
    )
    missed_component_analysis["Isolation_Forest_component_flag"] = (
        missed_component_analysis["Isolation_Forest_Score"] > thresholds["Isolation_Forest"]
    )
    missed_component_analysis["Scan_Behaviour_component_flag"] = (
        missed_component_analysis["Scan_Behaviour_Score"] > thresholds["Scan_Behaviour"]
    )

    # Save score distributions for reproducible plots/statistics.
    distribution_rows = []
    for component in ["GNN", "Isolation_Forest", "Scan_Behaviour", "Combined_V6"]:
        for split, arr in [
            ("benign_validation", {
                "GNN": val_gnn, "Isolation_Forest": val_if,
                "Scan_Behaviour": val_rule, "Combined_V6": val_combined
            }[component]),
            ("benign_test", {
                "GNN": benign_gnn, "Isolation_Forest": benign_if,
                "Scan_Behaviour": benign_rule, "Combined_V6": benign_combined
            }[component]),
            ("portscan_test", {
                "GNN": attack_gnn, "Isolation_Forest": attack_if,
                "Scan_Behaviour": attack_rule, "Combined_V6": attack_combined
            }[component]),
        ]:
            distribution_rows.append({
                "Component": component, "Split": split,
                "Count": int(len(arr)),
                "Mean": float(np.mean(arr)),
                "Std": float(np.std(arr)),
                "Min": float(np.min(arr)),
                "P50": float(np.percentile(arr, 50)),
                "P90": float(np.percentile(arr, 90)),
                "P95": float(np.percentile(arr, 95)),
                "P99": float(np.percentile(arr, 99)),
                "Max": float(np.max(arr)),
            })

    # A temporal protocol note is included in every summary.
    summary = {
        "model_version": "V6",
        "retraining_performed": False,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "calibration_source": "benign_validation_only",
        "calibration_percentile": CALIBRATION_PERCENTILE,
        "component_thresholds": thresholds,
        "combined_weights": {
            "gnn": W_GNN, "isolation_forest": W_IF, "scan_behaviour": W_RULE
        },
        "test_counts": {
            "benign": int(benign_count), "portscan": int(attack_count)
        },
        "component_ablation": results,
        "combined_threshold_tradeoff": tradeoff,
        "limitations": [
            "PortScan was excluded from model fitting and threshold calibration.",
            "Host mapping is transductive.",
            "Temporal features use previous completed minute/five-minute buckets; this is not sub-minute streaming.",
            "Test features reproduce the V6 combined chronological test stream: earlier benign and PortScan observations can contribute to later test-flow history. This is label-blind but is a mixed-class online-history protocol, not independent per-class replay.",
            "The held-out attack evaluation covers PortScan only, not every possible zero-day attack."
        ],
    }

    outputs = {
        "all_test_flow_predictions.csv": all_predictions,
        "combined_v6_false_positives.csv": false_positives,
        "combined_v6_missed_portscan.csv": missed_portscan,
        "combined_v6_detected_portscan.csv": detected_portscan,
        "missed_attack_component_analysis.csv": missed_component_analysis,
        "test_score_distributions.csv": pd.DataFrame(distribution_rows),
        "publication_ablation_results.csv": pd.DataFrame([
            {"Component": name, **res} for name, res in results.items()
        ]),
    }
    for filename, frame in outputs.items():
        path = REPORT_DIR / filename
        frame.to_csv(path, index=False)
        print("Saved:", path)

    with open(REPORT_DIR / "frozen_validation_thresholds.json", "w", encoding="utf-8") as f:
        json.dump({
            "calibration_source": "benign_validation_only",
            "percentile": CALIBRATION_PERCENTILE,
            "thresholds": thresholds,
            "validation_component_thresholds": val_component_thresholds,
        }, f, indent=4)

    with open(REPORT_DIR / "publication_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 78)
    print("V6 PUBLICATION EVALUATION COMPLETED â€” NO MODEL RETRAINED")
    print("=" * 78)
    print("False positives:", len(false_positives))
    print("Missed PortScan:", len(missed_portscan))
    print("Detected PortScan:", len(detected_portscan))
    print("Output directory:", REPORT_DIR)
    print("Thresholds calibrated on benign validation only.")
    print("Original V5/V6 model artifacts were not modified.")


if __name__ == "__main__":
    main()

