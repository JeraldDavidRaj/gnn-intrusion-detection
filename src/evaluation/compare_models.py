
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
DATA_PATH = Path("data/processed/cic_ids2017_cleaned.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = [
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

EXTRA_FEATURES = [
    "Source Port",
    "Destination Port",
    "Protocol",
]

FEATURE_CONFIGURATIONS = {
    "21_Traffic_Features": BASE_FEATURES,
    "24_Traffic_Plus_Network_Metadata": (
        BASE_FEATURES + EXTRA_FEATURES
    ),
}


# ============================================================
# 2. LOAD AND FILTER DATA
# Match the original train_gnn.py filtering.
# ============================================================

print("Loading cleaned CIC-IDS2017 dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

required_columns = list(
    dict.fromkeys(
        BASE_FEATURES
        + EXTRA_FEATURES
        + ["Source IP", "Destination IP", "Label"]
    )
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

df = df.dropna(
    subset=["Source IP", "Destination IP", "Label"]
).copy()

df["Label"] = df["Label"].astype(str).str.strip()

df = df[
    (df["Label"] != "")
    & (df["Source IP"].astype(str).str.strip() != "")
    & (df["Destination IP"].astype(str).str.strip() != "")
].reset_index(drop=True)

df["binary_label"] = (
    df["Label"].str.upper() != "BENIGN"
).astype(np.int64)

# Convert numeric traffic features and ports.
numeric_columns = BASE_FEATURES + [
    "Source Port",
    "Destination Port",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

# Protocol is categorical metadata. Encode using categories
# learned from the training split only, in each experiment.
df["Protocol"] = df["Protocol"].astype("string").str.strip()
df.loc[df["Protocol"] == "", "Protocol"] = pd.NA

print(f"Valid labelled flows: {len(df):,}")
print(f"Base traffic features: {len(BASE_FEATURES)}")
print(f"Additional metadata features: {len(EXTRA_FEATURES)}")


# ============================================================
# 3. REPRODUCE THE GCN SPLIT
# 70% training, 15% validation, 15% testing.
# ============================================================

all_indices = np.arange(len(df))
all_labels = df["binary_label"].to_numpy()

train_indices, temp_indices = train_test_split(
    all_indices,
    test_size=0.30,
    random_state=SEED,
    stratify=all_labels,
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=all_labels[temp_indices],
)

train_df = df.iloc[train_indices].copy()
val_df = df.iloc[val_indices].copy()
test_df = df.iloc[test_indices].copy()

y_train = train_df["binary_label"].to_numpy()
y_val = val_df["binary_label"].to_numpy()
y_test = test_df["binary_label"].to_numpy()

print("\nDATA SPLIT")
print("-" * 50)
print(f"Training flows:   {len(train_df):,}")
print(f"Validation flows: {len(val_df):,}")
print(f"Testing flows:    {len(test_df):,}")


# ============================================================
# 4. DEFINE MODELS
# ============================================================

def create_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


# ============================================================
# 5. RUN BOTH FEATURE EXPERIMENTS
# ============================================================

all_results = []

for config_name, feature_columns in FEATURE_CONFIGURATIONS.items():

    print("\n" + "#" * 65)
    print("FEATURE CONFIGURATION:", config_name)
    print("Number of features:", len(feature_columns))
    print("#" * 65)

    # Prepare the feature matrices.
    X_train_raw = train_df[feature_columns].copy()
    X_val_raw = val_df[feature_columns].copy()
    X_test_raw = test_df[feature_columns].copy()

    # Fit protocol encoding only on training data.
    if "Protocol" in feature_columns:
        protocol_categories = (
            X_train_raw["Protocol"]
            .dropna()
            .unique()
            .tolist()
        )

        protocol_mapping = {
            category: index
            for index, category in enumerate(protocol_categories)
        }

        for matrix in (
            X_train_raw,
            X_val_raw,
            X_test_raw,
        ):
            matrix["Protocol"] = (
                matrix["Protocol"]
                .map(protocol_mapping)
            )

    # Make all selected inputs numeric.
    X_train_raw = X_train_raw.apply(
        pd.to_numeric, errors="coerce"
    )
    X_val_raw = X_val_raw.apply(
        pd.to_numeric, errors="coerce"
    )
    X_test_raw = X_test_raw.apply(
        pd.to_numeric, errors="coerce"
    )

    # Fit imputation exclusively on training data.
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_val_imputed = imputer.transform(X_val_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    # Fit scaling exclusively on training data.
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

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    models = create_models()

    for model_name, model in models.items():

        print("\n" + "-" * 55)
        print(f"Training {model_name} | {config_name}")

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        predictions = model.predict(X_test)
        inference_time = time.perf_counter() - start_time

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        result = {
            "Model": model_name,
            "Feature_Set": config_name,
            "Feature_Count": len(feature_columns),
            "Accuracy": accuracy,
            "Precision_Attack": precision,
            "Recall_Attack": recall,
            "F1_Attack": f1,
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
            "Training_Time_Seconds": training_time,
            "Test_Inference_Time_Seconds": inference_time,
            "Training_Flows": len(X_train),
            "Validation_Flows": len(X_val),
            "Test_Flows": len(X_test),
        }

        all_results.append(result)

        print(f"Accuracy:         {accuracy:.6f}")
        print(f"Attack precision: {precision:.6f}")
        print(f"Attack recall:    {recall:.6f}")
        print(f"Attack F1-score:  {f1:.6f}")
        print(f"Training time:    {training_time:.2f} seconds")
        print(f"Inference time:   {inference_time:.2f} seconds")
        print(f"False positives:  {fp}")
        print(f"False negatives:  {fn}")

        print("\nConfusion matrix [BENIGN, ATTACK]:")
        print(confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ))

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
        ))

        safe_model_name = model_name.lower().replace(" ", "_")

        prediction_path = REPORTS_DIR / (
            f"{safe_model_name}_{config_name}_predictions.csv"
        )

        pd.DataFrame({
            "Original_Row_Index": test_indices,
            "Actual_Label": y_test,
            "Predicted_Label": predictions,
        }).to_csv(prediction_path, index=False)

        print("Saved predictions:", prediction_path)


# ============================================================
# 6. SAVE COMPARISON REPORT AND METADATA
# ============================================================

results_df = pd.DataFrame(all_results).sort_values(
    ["Feature_Set", "F1_Attack"],
    ascending=[True, False],
)

comparison_path = REPORTS_DIR / "model_comparison.csv"
results_df.to_csv(comparison_path, index=False)

metadata = {
    "dataset_path": str(DATA_PATH),
    "seed": SEED,
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "stratification_target": "binary_label",
    "class_mapping": {
        "BENIGN": 0,
        "ATTACK": 1,
    },
    "feature_configurations": {
        name: features
        for name, features in FEATURE_CONFIGURATIONS.items()
    },
    "training_flows": len(train_df),
    "validation_flows": len(val_df),
    "test_flows": len(test_df),
    "note": (
        "Protocol categories are mapped using training data only. "
        "Results are binary classification results. "
        "Exact alignment with the original GCN depends on identical "
        "dataset contents, row order, filtering, and split."
    ),
}

metadata_path = REPORTS_DIR / "model_comparison_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print("\n" + "=" * 75)
print("FINAL COMPARISON: ALL FEATURE CONFIGURATIONS")
print(results_df.to_string(index=False))
print("\nSaved comparison:", comparison_path)
print("Saved metadata:", metadata_path)
print("\nBoth feature experiments completed.")

