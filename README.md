# GNN-Based Network Intrusion Detection

## 1. Project Overview

This project implements a Graph Neural Network (GNN)-based network intrusion detection system using the CIC-IDS2017 dataset.

It investigates three main detection approaches:

- **PortScan V6:** Benign-only temporal graph reconstruction combined with Isolation Forest and scan-behaviour scoring.
- **DoS Hulk:** Benign-only temporal graph reconstruction combined with Isolation Forest.
- **Hybrid GNN:** Supervised graph-based classification trained using benign traffic and selected known attack categories, with PortScan excluded from attack training for held-out attack-category evaluation.

Network hosts are represented as graph nodes, while network flows are represented as directed edges.

## 2. Dataset

The project uses the CIC-IDS2017 dataset published by the Canadian Institute for Cybersecurity, University of New Brunswick.

- [Official CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Dataset Download Portal](https://cicresearch.ca/CICDataset/CIC-IDS-2017/)

The repository contains the eight labelled CSV files under:

```text
Dataset/
└── TrafficLabelling/
```

The cleaned dataset is stored under:

```text
data/
└── processed/
    └── cic_ids2017_cleaned.csv
```

## 3. Technologies and Dependencies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning implementation |
| PyTorch Geometric | Graph Neural Networks |
| Pandas | Dataset manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, Isolation Forest and evaluation |
| Streamlit | Interactive dashboard |
| Joblib | Saving and loading model artifacts |

The main framework versions specified in `requirements.txt` are:

- PyTorch 2.5.1
- PyTorch Geometric 2.8.1

## 4. Installation and Setup

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd GNN-Intrusion-Detection
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

For PyTorch or PyTorch Geometric compatibility issues, refer to:

- https://pytorch.org/get-started/locally/
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### Step 4: Verify installation

```bash
python -c "import torch, torch_geometric, pandas, numpy, sklearn, streamlit, joblib; print('Dependencies installed successfully')"
```

## 5. Project Structure

```text
GNN-Intrusion-Detection/
│
├── Dataset/
│   └── TrafficLabelling/
│
├── data/
│   ├── processed/
│   ├── zero_day/
│   └── remaining_attacks/
│
├── experiments/
│   └── dos_hulk/
│       ├── data/
│       ├── models/
│       ├── reports/
│       └── src/
│
├── models/
│   ├── baseline/
│   ├── hybrid/
│   └── portscan_v6/
│
├── reports/
│   ├── baseline/
│   ├── hybrid/
│   └── zero_day/
│
├── src/
│   ├── analysis/
│   ├── core/
│   ├── evaluation/
│   ├── graph/
│   ├── preprocessing/
│   └── training/
│
├── tests/
├── archive/
├── app.py
├── requirements.txt
└── README.md
```

## 6. Data Preprocessing

The preprocessing script is located at:

```text
src/preprocessing/data_preprocessing.py
```

Execute:

```bash
python src/preprocessing/data_preprocessing.py
```

The preprocessing stage performs dataset loading, column validation, removal of rows with missing essential fields, numeric conversion and handling of infinite values.

The cleaned dataset is saved in the `data/processed/` directory.

## 7. Model Execution

The repository contains separate scripts for the different model experiments.

### 7.1 PortScan V6

The main training script is:

```text
src/training/train_zero_day_gnn.py
```

Execute:

```bash
python src/training/train_zero_day_gnn.py
```

The trained model and related artifacts are stored under:

```text
models/portscan_v6/
```

Evaluation results are stored under:

```text
reports/zero_day/portscan_v6/
```

### 7.2 DoS Hulk

The experiment-specific training script is:

```text
experiments/dos_hulk/src/train_dos_hulk_gnn.py
```

Execute:

```bash
python experiments/dos_hulk/src/train_dos_hulk_gnn.py
```

Model artifacts and evaluation results are stored in the corresponding `experiments/dos_hulk/` directories.

### 7.3 Hybrid GNN

The training script is:

```text
src/training/train_hybrid_gnn.py
```

Execute:

```bash
python src/training/train_hybrid_gnn.py
```

The trained model and evaluation outputs are stored under:

```text
models/hybrid/
reports/hybrid/
```

### 7.4 Data Preparation

The zero-day data preparation script is:

```text
src/preprocessing/zero_day_data_preparation.py
```

Execute:

```bash
python src/preprocessing/zero_day_data_preparation.py
```

The graph construction and experiment-specific preparation scripts are available under the `src/` directory.

**Important:** Run the data preparation stages before their corresponding training stages. Check the scripts for required input files and arguments before executing them.

## 8. Launching the Dashboard

The project includes a Streamlit dashboard that displays saved experimental results.

From the repository's root directory, execute:

```bash
streamlit run app.py
```

Open the following address in your browser:

```text
http://localhost:8501
```

### Dashboard Sections

- Project Overview
- PortScan V6 – Zero-Day
- DoS Hulk – Zero-Day
- Hybrid GNN
- Model Comparison
- Flow Explorer
- Dataset Inspection

The dashboard uses previously generated model results, predictions, metrics and training histories. It is designed for experimental result exploration rather than live network traffic detection.

## 9. Evaluation Metrics

The experiments use the following evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- False Positive Rate
- ROC-AUC
- PR-AUC

Confusion matrices, ROC curves, precision-recall curves, score distributions and training histories are also used to assess model performance.

## 10. Testing

The repository contains preprocessing-related tests under:

```text
tests/
```

Run the tests using:

```bash
python -m pytest tests/
```

If pytest is not installed:

```bash
pip install pytest
```

## 11. Limitations

- The experiments are evaluated using the CIC-IDS2017 benchmark dataset.
- Unseen-attack evaluation is limited to selected held-out attack categories.
- The graph representation uses transductive host mapping.
- Temporal features use one-minute and five-minute intervals.
- The dashboard displays saved experimental outputs rather than performing live network traffic detection.

## 12. References

Sharafaldin, I., Lashkari, A. H., and Ghorbani, A. A. (2018). "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." Proceedings of the International Conference on Information Systems Security and Privacy (ICISSP).

Dataset: https://www.unb.ca/cic/datasets/ids-2017.html