from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from torch_geometric.nn import GCNConv


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "remaining_attacks"
# Selected experiment versions are stored separately.
V3_ATTACKS = {"bot", "infiltration", "web_attack"}
REPORT_ROOT = ROOT / "reports" / "zero_day"
MODEL_ROOT = ROOT / "models"

SEED = 42
CONTEXT_RATIO = 0.80

HIDDEN = 64
EPOCHS = 35
PATIENCE = 7

BATCH_SIZE = 2048

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Balanced threshold calibration
CALIBRATION_PERCENTILES = [
    98.0, 98.5, 99.0, 99.25, 99.5, 99.75, 99.9
]

IF_TREES = 200
IF_MAX_SAMPLES = 100000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

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
    "Idle Mean"
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
    "pair_unique_dst_ports_prev_5m"
]


ATTACKS = {
    "ddos": {
        "labels": ["DDoS"],
        "date_group": "2017-07-07"
    },
    "bot": {
        "labels": ["Bot"],
        "date_group": "2017-07-07"
    },
    "dos_hulk": {
        "labels": ["DoS Hulk"],
        "date_group": "2017-07-05"
    },
    "dos_goldeneye": {
        "labels": ["DoS GoldenEye"],
        "date_group": "2017-07-05"
    },
    "web_attack": {
        "labels": [
            "Web Attack â€“ Brute Force",
            "Web Attack â€“ XSS",
            "Web Attack â€“ Sql Injection"
        ],
        "date_group": "2017-07-06"
    },
    "infiltration": {
        "labels": ["Infiltration"],
        "date_group": "2017-07-06"
    }
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATA LOADING
# ============================================================

def read_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required prepared file not found: {path}\n"
            "Run src/prepare_remaining_attack_data.py first."
        )

    df = pd.read_csv(path, low_memory=False)

    if "_time" not in df.columns:
        raise ValueError(
            f"{path.name} has no _time column."
        )

    df["_time"] = pd.to_datetime(
        df["_time"],
        errors="coerce"
    )

    if df["_time"].isna().any():
        raise ValueError(
            f"{path.name} contains invalid _time values."
        )

    for c in FLOW_COLS:
        if c not in df.columns:
            raise ValueError(
                f"{path.name} is missing feature column: {c}"
            )

    for c in ["Source IP", "Destination IP"]:
        if c not in df.columns:
            raise ValueError(
                f"{path.name} is missing {c}"
            )

    return df


def load_attack_data(name):

    train = read_csv(
        DATA / f"{name}_benign_train_candidates.csv"
    )

    val = read_csv(
        DATA / f"{name}_benign_validation.csv"
    )

    benign_test = read_csv(
        DATA / f"{name}_benign_test.csv"
    )

    attack_test = read_csv(
        DATA / f"{name}_test.csv"
    )

    if (
        len(train) == 0
        or len(val) == 0
        or len(benign_test) == 0
        or len(attack_test) == 0
    ):
        raise ValueError(
            f"{name}: one or more splits are empty. "
            "Review preparation_metadata.json."
        )

    # Chronological split integrity
    if train["_time"].max() >= val["_time"].min():
        raise ValueError(
            f"{name}: training data is not strictly earlier "
            "than validation data."
        )

    if val["_time"].max() >= attack_test["_time"].min():
        raise ValueError(
            f"{name}: validation data overlaps or follows "
            "the attack start."
        )

    test = pd.concat(
        [benign_test, attack_test],
        ignore_index=True
    )

    test = test.sort_values(
        "_time",
        kind="mergesort"
    ).reset_index(drop=True)

    test["_target"] = (
        test["Label"]
        .isin(ATTACKS[name]["labels"])
        .astype(np.int64)
    )

    unexpected = (
        set(test["Label"].astype(str))
        - {"BENIGN"}
        - set(ATTACKS[name]["labels"])
    )

    if unexpected:
        raise ValueError(
            f"{name}: unexpected labels in test data: {unexpected}"
        )

    return train, val, test


# ============================================================
# NUMERIC PREPROCESSING
# ============================================================

def safe_numeric(df, columns):

    arr = (
        df[columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )

    arr[~np.isfinite(arr)] = np.nan

    return arr


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def flow_endpoints(df, host_map):

    src = (
        df["Source IP"]
        .astype(str)
        .map(host_map)
    )

    dst = (
        df["Destination IP"]
        .astype(str)
        .map(host_map)
    )

    if src.isna().any() or dst.isna().any():
        raise ValueError(
            "Found an IP address missing from the host mapping."
        )

    return (
        src.to_numpy(dtype=np.int64),
        dst.to_numpy(dtype=np.int64)
    )


def build_host_map(train, val, test):

    ips = pd.concat([
        train["Source IP"].astype(str),
        train["Destination IP"].astype(str),
        val["Source IP"].astype(str),
        val["Destination IP"].astype(str),
        test["Source IP"].astype(str),
        test["Destination IP"].astype(str)
    ], ignore_index=True)

    unique_ips = sorted(
        ips.dropna().unique().tolist()
    )

    return {
        ip: i
        for i, ip in enumerate(unique_ips)
    }


# ============================================================
# CAUSAL TEMPORAL FEATURES
# ============================================================

def causal_temporal_features(history, query):

    """
    Strict completed-bucket features.

    A query in minute M only receives aggregates from
    completed minute M-1.

    Five-minute features use the preceding completed
    five-minute bucket.

    Labels are never used.
    """

    q = query[[
        "_time",
        "Source IP",
        "Destination IP",
        "Source Port",
        "Destination Port",
        "SYN Flag Count"
    ]].copy().reset_index(drop=True)

    for c in [
        "Source Port",
        "Destination Port",
        "SYN Flag Count"
    ]:

        if c not in q.columns:

            if c == "SYN Flag Count":
                q[c] = 0
            else:
                raise ValueError(
                    f"Missing temporal field: {c}"
                )

        q[c] = pd.to_numeric(
            q[c],
            errors="coerce"
        ).fillna(0)

    if len(history):

        h = history[[
            "_time",
            "Source IP",
            "Destination IP",
            "Source Port",
            "Destination Port",
            "SYN Flag Count"
        ]].copy().reset_index(drop=True)

    else:
        h = q.iloc[0:0].copy()

    for c in [
        "Source Port",
        "Destination Port",
        "SYN Flag Count"
    ]:

        if c not in h.columns:
            h[c] = 0

        h[c] = pd.to_numeric(
            h[c],
            errors="coerce"
        ).fillna(0)

    q["_qid"] = np.arange(len(q))

    q["_minute"] = (
        q["_time"].astype("int64")
        // 60_000_000_000
    )

    q["_five"] = q["_minute"] // 5

    h["_minute"] = (
        h["_time"].astype("int64")
        // 60_000_000_000
    )

    h["_five"] = h["_minute"] // 5

    all_rows = pd.concat(
        [h, q],
        ignore_index=True
    )

    # Source/minute aggregates
    src1 = (
        all_rows
        .groupby(
            ["Source IP", "_minute"],
            sort=False
        )
        .agg(
            src_flows=("Destination IP", "size"),
            src_dst_ips=("Destination IP", "nunique"),
            src_dst_ports=("Destination Port", "nunique"),
            src_syn=("SYN Flag Count", "sum")
        )
        .reset_index()
    )

    src1["_minute"] += 1

    # Source/five-minute aggregates
    src5 = (
        all_rows
        .groupby(
            ["Source IP", "_five"],
            sort=False
        )
        .agg(
            src_flows_5=("Destination IP", "size"),
            src_dst_ips_5=("Destination IP", "nunique"),
            src_dst_ports_5=("Destination Port", "nunique")
        )
        .reset_index()
    )

    src5["_five"] += 1

    # Destination/minute aggregates
    dst1 = (
        all_rows
        .groupby(
            ["Destination IP", "_minute"],
            sort=False
        )
        .agg(
            dst_flows=("Source IP", "size"),
            dst_src_ips=("Source IP", "nunique"),
            dst_src_ports=("Source Port", "nunique"),
            dst_syn=("SYN Flag Count", "sum")
        )
        .reset_index()
    )

    dst1["_minute"] += 1

    # Source-destination pair/minute aggregates
    pair1 = (
        all_rows
        .groupby(
            ["Source IP", "Destination IP", "_minute"],
            sort=False
        )
        .agg(
            pair_flows=("Destination Port", "size"),
            pair_ports=("Destination Port", "nunique")
        )
        .reset_index()
    )

    pair1["_minute"] += 1

    # Source-destination pair/five-minute aggregates
    pair5 = (
        all_rows
        .groupby(
            ["Source IP", "Destination IP", "_five"],
            sort=False
        )
        .agg(
            pair_ports_5=("Destination Port", "nunique")
        )
        .reset_index()
    )

    pair5["_five"] += 1

    result = q[[
        "_qid",
        "Source IP",
        "Destination IP",
        "_minute",
        "_five"
    ]].copy()

    result = result.merge(
        src1,
        on=["Source IP", "_minute"],
        how="left"
    )

    result = result.merge(
        src5,
        on=["Source IP", "_five"],
        how="left"
    )

    result = result.merge(
        dst1,
        on=["Destination IP", "_minute"],
        how="left"
    )

    result = result.merge(
        pair1,
        on=["Source IP", "Destination IP", "_minute"],
        how="left"
    )

    result = result.merge(
        pair5,
        on=["Source IP", "Destination IP", "_five"],
        how="left"
    )

    result["src_syn_ratio"] = (
        result["src_syn"]
        / result["src_flows"].replace(0, np.nan)
    ).fillna(0)

    # Port entropy
    port_counts = (
        all_rows
        .groupby(
            ["Source IP", "_minute", "Destination Port"],
            sort=False
        )
        .size()
        .rename("n")
        .reset_index()
    )

    totals = (
        port_counts
        .groupby(
            ["Source IP", "_minute"],
            sort=False
        )["n"]
        .transform("sum")
    )

    port_counts["p"] = (
        port_counts["n"] / totals
    )

    port_counts["entropy_part"] = (
        -port_counts["p"]
        * np.log(
            port_counts["p"].clip(lower=1e-12)
        )
    )

    entropy = (
        port_counts
        .groupby(
            ["Source IP", "_minute"],
            sort=False
        )["entropy_part"]
        .sum()
        .reset_index(name="src_port_entropy")
    )

    entropy["_minute"] += 1

    result = result.merge(
        entropy,
        on=["Source IP", "_minute"],
        how="left"
    )

    rename = {
        "src_flows": "src_flows_1m",
        "src_dst_ips": "src_unique_dst_ip_1m",
        "src_dst_ports": "src_unique_dst_port_1m",
        "src_syn": "src_syn_count_1m",
        "src_flows_5": "src_flows_5m",
        "src_dst_ips_5": "src_unique_dst_ip_5m",
        "src_dst_ports_5": "src_unique_dst_port_5m",
        "dst_flows": "dst_flows_1m",
        "dst_src_ips": "dst_unique_src_ip_1m",
        "dst_src_ports": "dst_unique_src_port_1m",
        "dst_syn": "dst_syn_count_1m",
        "src_port_entropy": "src_destination_port_entropy_1m",
        "pair_flows": "pair_flows_prev_1m",
        "pair_ports": "pair_unique_dst_ports_prev_1m",
        "pair_ports_5": "pair_unique_dst_ports_prev_5m",
        "src_syn_ratio": "src_syn_ratio_1m"
    }

    result = result.rename(columns=rename)

    result = (
        result
        .set_index("_qid")
        .reindex(range(len(q)))
    )

    result = (
        result[TEMP_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return result.to_numpy(dtype=np.float32)


# ============================================================
# NODE FEATURES
# ============================================================

def make_graph_features(
    src,
    dst,
    edge_features,
    num_nodes
):

    out_sum = np.zeros(
        (num_nodes, edge_features.shape[1]),
        dtype=np.float32
    )

    in_sum = np.zeros_like(out_sum)

    out_count = np.zeros(
        num_nodes,
        dtype=np.float32
    )

    in_count = np.zeros(
        num_nodes,
        dtype=np.float32
    )

    np.add.at(out_sum, src, edge_features)
    np.add.at(in_sum, dst, edge_features)

    np.add.at(out_count, src, 1)
    np.add.at(in_count, dst, 1)

    out_mean = (
        out_sum
        / np.maximum(out_count[:, None], 1)
    )

    in_mean = (
        in_sum
        / np.maximum(in_count[:, None], 1)
    )

    x = np.concatenate([
        out_mean,
        in_mean,
        np.log1p(out_count)[:, None],
        np.log1p(in_count)[:, None]
    ], axis=1)

    return x.astype(np.float32)


# ============================================================
# TEMPORAL GCN MODEL
# ============================================================

class TemporalGCN(nn.Module):

    def __init__(
        self,
        node_dim,
        hidden_dim,
        output_dim,
        dropout=0.2
    ):

        super().__init__()

        self.conv1 = GCNConv(
            node_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.dropout = dropout

        self.decoder = nn.Sequential(
            nn.Linear(
                hidden_dim * 4,
                hidden_dim * 2
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                output_dim * 2
            )
        )

        self.output_dim = output_dim

    def encode(self, x, edge_index):

        h = self.conv1(
            x,
            edge_index
        )

        h = F.relu(h)

        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training
        )

        h = self.conv2(
            h,
            edge_index
        )

        return F.relu(h)

    def decode(self, z, edge_index):

        a = z[edge_index[0]]
        b = z[edge_index[1]]

        h = torch.cat([
            a,
            b,
            torch.abs(a - b),
            a * b
        ], dim=1)

        out = self.decoder(h)

        mean = out[:, :self.output_dim]

        logvar = out[:, self.output_dim:]

        return mean, torch.clamp(
            logvar,
            -6.0,
            4.0
        )


# ============================================================
# LOSS FUNCTION
# ============================================================

def gaussian_nll(mean, logvar, target):

    return 0.5 * (
        logvar
        + (target - mean) ** 2
        * torch.exp(-logvar)
    )


# ============================================================
# GNN TRAINING
# ============================================================

def train_gnn(
    model,
    x,
    context_edges,
    query_edges,
    targets,
    val_edges,
    val_targets
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_loss = float("inf")
    best_state = None

    wait = 0
    history = []

    x = torch.tensor(
        x,
        dtype=torch.float32,
        device=DEVICE
    )

    context_edges = torch.tensor(
        context_edges,
        dtype=torch.long,
        device=DEVICE
    )

    query_edges = torch.tensor(
        query_edges,
        dtype=torch.long,
        device=DEVICE
    )

    targets = torch.tensor(
        targets,
        dtype=torch.float32,
        device=DEVICE
    )

    val_edges = torch.tensor(
        val_edges,
        dtype=torch.long,
        device=DEVICE
    )

    val_targets = torch.tensor(
        val_targets,
        dtype=torch.float32,
        device=DEVICE
    )

    for epoch in range(1, EPOCHS + 1):

        model.train()

        optimizer.zero_grad()

        z = model.encode(
            x,
            context_edges
        )

        train_loss_total = 0.0
        batches = 0

        # Correct edge batching: edge_index has shape (2, N).
        for start in range(
            0,
            query_edges.shape[1],
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                query_edges.shape[1]
            )

            mean, logvar = model.decode(
                z,
                query_edges[:, start:end]
            )

            loss = gaussian_nll(
                mean,
                logvar,
                targets[start:end]
            ).mean()

            loss.backward(
                retain_graph=end < query_edges.shape[1]
            )

            train_loss_total += float(
                loss.detach().cpu()
            )

            batches += 1

        optimizer.step()

        train_loss = (
            train_loss_total
            / max(batches, 1)
        )

        # Validation
        model.eval()

        with torch.no_grad():

            z_val = model.encode(
                x,
                context_edges
            )

            val_total = 0.0
            val_batches = 0

            for start in range(
                0,
                val_edges.shape[1],
                BATCH_SIZE
            ):

                end = min(
                    start + BATCH_SIZE,
                    val_edges.shape[1]
                )

                mean, logvar = model.decode(
                    z_val,
                    val_edges[:, start:end]
                )

                loss = gaussian_nll(
                    mean,
                    logvar,
                    val_targets[start:end]
                ).mean()

                val_total += float(
                    loss.cpu()
                )

                val_batches += 1

            val_loss = (
                val_total
                / max(val_batches, 1)
            )

        history.append({
            "epoch": epoch,
            "train_nll": train_loss,
            "validation_nll": val_loss
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train NLL {train_loss:.6f} | "
            f"val NLL {val_loss:.6f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            wait = 0

        else:

            wait += 1

            if wait >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError(
            "Training failed to produce a model checkpoint."
        )

    model.load_state_dict(best_state)

    return (
        model,
        pd.DataFrame(history),
        best_loss
    )


# ============================================================
# GNN ANOMALY SCORING
# ============================================================

def score_gnn(
    model,
    x,
    context_edges,
    query_edges,
    targets,
    batch_size=BATCH_SIZE
):

    model.eval()

    with torch.no_grad():

        z = model.encode(
            torch.tensor(
                x,
                dtype=torch.float32,
                device=DEVICE
            ),
            torch.tensor(
                context_edges,
                dtype=torch.long,
                device=DEVICE
            )
        )

        edges = torch.tensor(
            query_edges,
            dtype=torch.long,
            device=DEVICE
        )

        target = torch.tensor(
            targets,
            dtype=torch.float32,
            device=DEVICE
        )

        scores = []

        for start in range(
            0,
            edges.shape[1],
            batch_size
        ):

            end = min(
                start + batch_size,
                edges.shape[1]
            )

            mean, logvar = model.decode(
                z,
                edges[:, start:end]
            )

            nll = gaussian_nll(
                mean,
                logvar,
                target[start:end]
            ).mean(dim=1)

            scores.append(
                nll.cpu().numpy()
            )

    return np.concatenate(scores)


# ============================================================
# SCAN BEHAVIOUR RULE
# ============================================================

def scan_rule(temp):

    d = pd.DataFrame(
        temp,
        columns=TEMP_COLS
    )

    return (
        0.16 * np.log1p(
            d["src_flows_1m"]
        )
        + 0.12 * np.log1p(
            d["src_unique_dst_ip_1m"]
        )
        + 0.18 * np.log1p(
            d["src_unique_dst_port_1m"]
        )
        + 0.10 * np.log1p(
            d["src_unique_dst_port_5m"]
        )
        + 0.08 * d["src_syn_ratio_1m"]
        + 0.06 * np.log1p(
            d["dst_unique_src_ip_1m"]
        )
        + 0.18 * np.log1p(
            d["pair_unique_dst_ports_prev_1m"]
        )
        + 0.12 * np.log1p(
            d["pair_unique_dst_ports_prev_5m"]
        )
    ).to_numpy(dtype=np.float64)


# ============================================================
# PERCENTILE SCORING
# ============================================================

def percentile_scores(reference, values):

    ref = np.sort(
        np.asarray(
            reference,
            dtype=np.float64
        )
    )

    return (
        np.searchsorted(
            ref,
            values,
            side="right"
        )
        / max(len(ref), 1)
    )


# ============================================================
# BALANCED THRESHOLD CALIBRATION
# ============================================================

def calibrate_thresholds(
    validation_scores,
    percentiles
):

    """
    Candidate thresholds are calculated using benign
    validation scores only.

    No attack labels or test labels are used here.
    """

    scores = np.asarray(
        validation_scores,
        dtype=np.float64
    )

    if len(scores) == 0:
        raise ValueError(
            "Validation scores are empty."
        )

    results = []

    for percentile in percentiles:

        threshold = float(
            np.percentile(
                scores,
                percentile
            )
        )

        alerts = int(
            np.sum(scores >= threshold)
        )

        validation_fpr = (
            alerts / len(scores)
        )

        results.append({
            "percentile": float(percentile),
            "threshold": threshold,
            "validation_fpr": float(validation_fpr),
            "validation_alerts": alerts
        })

    return results


# ============================================================
# EVALUATION METRICS
# ============================================================

def metrics(
    y_true,
    scores,
    threshold
):

    pred = (
        scores >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    ).ravel()

    result = {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(y_true, pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                pred,
                zero_division=0
            )
        ),

        "false_positive_rate": float(
            fp / max(fp + tn, 1)
        ),

        "true_positive_rate": float(
            tp / max(tp + fn, 1)
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    if len(np.unique(y_true)) == 2:

        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                scores
            )
        )

        result["pr_auc"] = float(
            average_precision_score(
                y_true,
                scores
            )
        )

    else:

        result["roc_auc"] = None
        result["pr_auc"] = None

    return result, pred


# ============================================================
# MAIN ATTACK TRAINING ENGINE
# ============================================================

def run_attack(name):

    if name not in ATTACKS:
        raise ValueError(
            f"Unknown attack name: {name}"
        )

    set_seed()

    print("=" * 78)
    print(
        f"BENIGN-ONLY ZERO-DAY EXPERIMENT: {name.upper()}"
    )
    print("Device:", DEVICE)
    print("=" * 78)

    train_df, val_df, test_df = load_attack_data(name)

    # --------------------------------------------------------
    # FLOW FEATURE PREPROCESSING
    # --------------------------------------------------------

    train_raw = safe_numeric(
        train_df,
        FLOW_COLS
    )

    val_raw = safe_numeric(
        val_df,
        FLOW_COLS
    )

    test_raw = safe_numeric(
        test_df,
        FLOW_COLS
    )

    all_missing = np.isnan(
        train_raw
    ).all(axis=0)

    train_raw[:, all_missing] = 0.0
    val_raw[:, all_missing] = 0.0
    test_raw[:, all_missing] = 0.0

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = StandardScaler()

    train_flow = scaler.fit_transform(
        imputer.fit_transform(train_raw)
    ).astype(np.float32)

    val_flow = scaler.transform(
        imputer.transform(val_raw)
    ).astype(np.float32)

    test_flow = scaler.transform(
        imputer.transform(test_raw)
    ).astype(np.float32)

    # --------------------------------------------------------
    # TEMPORAL FEATURE GENERATION
    # --------------------------------------------------------

    train_temp = causal_temporal_features(
        train_df.iloc[0:0],
        train_df
    )

    val_temp = causal_temporal_features(
        train_df,
        val_df
    )

    test_history = pd.concat(
        [train_df, val_df],
        ignore_index=True
    )

    test_temp = causal_temporal_features(
        test_history,
        test_df
    )

    temp_imputer = SimpleImputer(
        strategy="median"
    )

    temp_scaler = StandardScaler()

    train_temp_clean = temp_imputer.fit_transform(
        train_temp
    )

    val_temp_clean = temp_imputer.transform(
        val_temp
    )

    test_temp_clean = temp_imputer.transform(
        test_temp
    )

    train_temp_scaled = temp_scaler.fit_transform(
        train_temp_clean
    ).astype(np.float32)

    val_temp_scaled = temp_scaler.transform(
        val_temp_clean
    ).astype(np.float32)

    test_temp_scaled = temp_scaler.transform(
        test_temp_clean
    ).astype(np.float32)

    # --------------------------------------------------------
    # ISOLATION FOREST
    # --------------------------------------------------------

    iforest = IsolationForest(
        n_estimators=IF_TREES,
        max_samples=min(
            IF_MAX_SAMPLES,
            len(train_temp_scaled)
        ),
        contamination="auto",
        random_state=SEED,
        n_jobs=-1
    )

    iforest.fit(
        train_temp_scaled
    )

    # --------------------------------------------------------
    # HOST MAPPING AND GRAPH
    # --------------------------------------------------------

    host_map = build_host_map(
        train_df,
        val_df,
        test_df
    )

    tr_src, tr_dst = flow_endpoints(
        train_df,
        host_map
    )

    va_src, va_dst = flow_endpoints(
        val_df,
        host_map
    )

    te_src, te_dst = flow_endpoints(
        test_df,
        host_map
    )

    rng = np.random.default_rng(SEED)

    order = rng.permutation(
        len(train_df)
    )

    context_count = max(
        1,
        int(len(order) * CONTEXT_RATIO)
    )

    context_idx = order[:context_count]

    query_idx = order[context_count:]

    if len(query_idx) == 0:
        raise ValueError(
            "Training split is too small to create query edges."
        )

    context_src = tr_src[context_idx]
    context_dst = tr_dst[context_idx]

    context_edge_index = np.vstack([
        context_src,
        context_dst
    ])

    context_edge_features = train_flow[
        context_idx
    ]

    node_x = make_graph_features(
        context_src,
        context_dst,
        context_edge_features,
        len(host_map)
    )

    train_query_edges = np.vstack([
        tr_src[query_idx],
        tr_dst[query_idx]
    ])

    train_targets = np.concatenate([
        train_flow[query_idx],
        train_temp_scaled[query_idx]
    ], axis=1)

    val_edges = np.vstack([
        va_src,
        va_dst
    ])

    val_targets = np.concatenate([
        val_flow,
        val_temp_scaled
    ], axis=1)

    test_edges = np.vstack([
        te_src,
        te_dst
    ])

    test_targets = np.concatenate([
        test_flow,
        test_temp_scaled
    ], axis=1)

    # --------------------------------------------------------
    # MODEL INITIALIZATION
    # --------------------------------------------------------

    output_dim = train_targets.shape[1]

    model = TemporalGCN(
        node_dim=node_x.shape[1],
        hidden_dim=HIDDEN,
        output_dim=output_dim
    ).to(DEVICE)

    print(
        f"Training flows: {len(train_df):,}"
    )

    print(
        f"Validation benign flows: {len(val_df):,}"
    )

    print(
        f"Test flows: {len(test_df):,}"
    )

    print(
        f"Graph nodes: {len(host_map):,}"
    )

    print(
        f"Context edges: {context_count:,}"
    )

    print(
        f"Training query edges: {len(query_idx):,}"
    )

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    # --------------------------------------------------------
    # TRAIN GNN
    # --------------------------------------------------------

    model, history, best_val_loss = train_gnn(
        model,
        node_x,
        context_edge_index,
        train_query_edges,
        train_targets,
        val_edges,
        val_targets
    )

    # --------------------------------------------------------
    # GNN ANOMALY SCORES
    # --------------------------------------------------------

    gnn_val = score_gnn(
        model,
        node_x,
        context_edge_index,
        val_edges,
        val_targets
    )

    gnn_test = score_gnn(
        model,
        node_x,
        context_edge_index,
        test_edges,
        test_targets
    )

    # --------------------------------------------------------
    # ISOLATION FOREST SCORES
    # --------------------------------------------------------

    if_val = -iforest.score_samples(
        val_temp_scaled
    )

    if_test = -iforest.score_samples(
        test_temp_scaled
    )

    # --------------------------------------------------------
    # SCAN RULE SCORES
    # --------------------------------------------------------

    rule_val = scan_rule(
        val_temp
    )

    rule_test = scan_rule(
        test_temp
    )

    # --------------------------------------------------------
    # COMPONENT PERCENTILE RANKING
    # --------------------------------------------------------

    gnn_val_rank = percentile_scores(
        gnn_val,
        gnn_val
    )

    gnn_test_rank = percentile_scores(
        gnn_val,
        gnn_test
    )

    if_val_rank = percentile_scores(
        if_val,
        if_val
    )

    if_test_rank = percentile_scores(
        if_val,
        if_test
    )

    rule_val_rank = percentile_scores(
        rule_val,
        rule_val
    )

    rule_test_rank = percentile_scores(
        rule_val,
        rule_test
    )

    # --------------------------------------------------------
    # BALANCED ENSEMBLE
    # --------------------------------------------------------

    # Preserve the established weights for a controlled
    # comparison with the original DoS Hulk experiment.

    GNN_WEIGHT = 0.35
    IF_WEIGHT = 0.40
    RULE_WEIGHT = 0.25

    combined_val = (
        GNN_WEIGHT * gnn_val_rank
        + IF_WEIGHT * if_val_rank
        + RULE_WEIGHT * rule_val_rank
    )

    combined_test = (
        GNN_WEIGHT * gnn_test_rank
        + IF_WEIGHT * if_test_rank
        + RULE_WEIGHT * rule_test_rank
    )

    # --------------------------------------------------------
    # BENIGN VALIDATION THRESHOLD CALIBRATION
    # --------------------------------------------------------

    threshold_table = calibrate_thresholds(
        combined_val,
        CALIBRATION_PERCENTILES
    )

    # Select candidate closest to 1% validation FPR.
    selected = min(
        threshold_table,
        key=lambda item: abs(
            item["validation_fpr"] - 0.01
        )
    )

    threshold = selected["threshold"]

    print("\nVALIDATION THRESHOLD CALIBRATION")

    print(
        pd.DataFrame(
            threshold_table
        ).to_string(index=False)
    )

    print(
        f"\nSelected threshold: {threshold:.8f}"
    )

    print(
        "Validation FPR:",
        f"{selected['validation_fpr']:.4%}"
    )

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    y_true = test_df["_target"].to_numpy(
        dtype=np.int64
    )

    result, predictions = metrics(
        y_true,
        combined_test,
        threshold
    )

    # --------------------------------------------------------
    # SAVE MODEL AND REPORTS
    # --------------------------------------------------------

    version = "attack_v3" if name in V3_ATTACKS else "attack_v2"

    model_dir = MODEL_ROOT / version / name
    report_dir = REPORT_ROOT / version / name

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = model_dir / f"zero_day_gnn_{name}_best.pth"

    torch.save({

        "model_state_dict": model.state_dict(),

        "node_dim": int(node_x.shape[1]),

        "hidden_dim": HIDDEN,

        "output_dim": output_dim,

        "flow_columns": FLOW_COLS,

        "temporal_columns": TEMP_COLS,

        "attack_name": name,

        "seed": SEED,

        "best_validation_nll": best_val_loss,

        "threshold_percentile": selected["percentile"],

        "combined_threshold": threshold,

        "ensemble_weights": {
            "gnn": GNN_WEIGHT,
            "isolation_forest": IF_WEIGHT,
            "scan_rule": RULE_WEIGHT
        },

        "validation_fpr": selected["validation_fpr"]

    }, checkpoint)

    # Save preprocessing objects
    joblib.dump(
        imputer,
        model_dir / f"{name}_flow_imputer.pkl"
    )

    joblib.dump(
        scaler,
        model_dir / f"{name}_flow_scaler.pkl"
    )

    joblib.dump(
        temp_imputer,
        model_dir / f"{name}_temporal_imputer.pkl"
    )

    joblib.dump(
        temp_scaler,
        model_dir / f"{name}_temporal_scaler.pkl"
    )

    joblib.dump(
        iforest,
        model_dir / f"{name}_iforest.pkl"
    )

    with open(
        model_dir / f"{name}_host_mapping.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            host_map,
            f
        )

    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    history.to_csv(
        report_dir / "training_history.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE VALIDATION SCORES
    # --------------------------------------------------------

    validation_scores_df = pd.DataFrame({

        "gnn_score": gnn_val,

        "if_score": if_val,

        "scan_score": rule_val,

        "gnn_rank": gnn_val_rank,

        "if_rank": if_val_rank,

        "scan_rank": rule_val_rank,

        "combined_score": combined_val

    })

    validation_scores_df.to_csv(
        report_dir / "validation_scores.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE THRESHOLD ANALYSIS
    # --------------------------------------------------------

    pd.DataFrame(
        threshold_table
    ).to_csv(
        report_dir / "threshold_analysis.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    predictions_df = test_df[[
        "_time",
        "Source IP",
        "Destination IP",
        "Label",
        "_target"
    ]].copy()

    predictions_df["gnn_score"] = gnn_test

    predictions_df["if_score"] = if_test

    predictions_df["scan_score"] = rule_test

    predictions_df["combined_score"] = combined_test

    predictions_df["prediction"] = predictions

    predictions_df.to_csv(
        report_dir / "predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE METRICS JSON
    # --------------------------------------------------------

    with open(
        report_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump({

            "attack": name,

            "attack_labels": ATTACKS[name]["labels"],

            "train_flows_benign_only": len(train_df),

            "validation_flows_benign_only": len(val_df),

            "test_flows": len(test_df),

            "test_attack_flows": int(y_true.sum()),

            "test_benign_flows": int(
                (y_true == 0).sum()
            ),

            "best_validation_nll": float(
                best_val_loss
            ),

            "threshold_percentile": selected["percentile"],

            "combined_threshold": threshold,

            "threshold_analysis": threshold_table,

            "ensemble_weights": {
                "gnn": GNN_WEIGHT,
                "isolation_forest": IF_WEIGHT,
                "scan_rule": RULE_WEIGHT
            },

            "validation_fpr": selected["validation_fpr"],

            "metrics": result,

            "protocol_notes": [

                "GNN, scaler, temporal scaler and Isolation Forest were fitted using benign training data only.",

                "Threshold was calibrated using benign validation data only.",

                "Temporal features use completed minute/five-minute buckets.",

                "Host ID vocabulary includes train, validation and test IP identities; node features use benign training context only.",

                "This is a held-out attack-category experiment on CIC-IDS2017, not proof of detection of arbitrary real-world zero-day attacks."

            ]

        }, f, indent=4)

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("FINAL RESULTS")
    print("=" * 78)

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    print("\nCheckpoint:")
    print(checkpoint)

    print("\nReport directory:")
    print(report_dir)

    print("\nCompleted:")
    print(name.upper())

    return result


# ============================================================
# COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attack",
        required=True,
        choices=sorted(ATTACKS.keys())
    )

    args = parser.parse_args()

    run_attack(args.attack)
