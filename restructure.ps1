
$ErrorActionPreference = "Stop"

$Root = "D:\ML GNN PROJECT\gnn-intrusion-detection"
Set-Location $Root

# Explicitly approved cleanup: move unwanted files to archive.
# Nothing is permanently deleted.

function Move-FileToArchive {
    param([string]$RelativePath)

    $Source = Join-Path $Root $RelativePath
    $Destination = Join-Path $Root ("archive\" + $RelativePath)
    $Parent = Split-Path $Destination -Parent

    if (!(Test-Path -LiteralPath $Source -PathType Leaf)) {
        return
    }

    New-Item -ItemType Directory -Path $Parent -Force | Out-Null

    if (Test-Path -LiteralPath $Destination) {
        Write-Warning "Already archived; skipped: $RelativePath"
        return
    }

    Move-Item -LiteralPath $Source -Destination $Destination
    Write-Host "ARCHIVED: $RelativePath" -ForegroundColor Yellow
}

function Move-DirectoryToArchive {
    param([string]$RelativePath)

    $Source = Join-Path $Root $RelativePath
    $Destination = Join-Path $Root ("archive\" + $RelativePath)
    $Parent = Split-Path $Destination -Parent

    if (!(Test-Path -LiteralPath $Source -PathType Container)) {
        return
    }

    New-Item -ItemType Directory -Path $Parent -Force | Out-Null

    if (Test-Path -LiteralPath $Destination) {
        throw "Archive destination already exists: $Destination"
    }

    Move-Item -LiteralPath $Source -Destination $Destination
    Write-Host "ARCHIVED DIRECTORY: $RelativePath" -ForegroundColor Yellow
}

Write-Host "`n===== 1. ARCHIVE NON-SELECTED MODEL BUNDLES =====" -ForegroundColor Cyan

# V2 selected: DDoS, DoS Hulk, GoldenEye.
# Archive non-selected V2 category bundles.
$V2Unselected = @("bot", "infiltration", "web_attack")

foreach ($Attack in $V2Unselected) {
    $Folder = Join-Path $Root "models\remaining_attack_experiments"

    if (Test-Path $Folder) {
        Get-ChildItem -LiteralPath $Folder -File |
            Where-Object { $_.Name -like "$($Attack)_*" } |
            ForEach-Object {
                Move-FileToArchive "models\remaining_attack_experiments\$($_.Name)"
            }
    }
}

# V3 selected: Bot, Web Attack, Infiltration.
# Archive non-selected V3 category bundles.
$V3Unselected = @("ddos", "dos_goldeneye", "dos_hulk")

foreach ($Attack in $V3Unselected) {
    $Folder = Join-Path $Root "models\zero_day_v3"

    if (Test-Path $Folder) {
        Get-ChildItem -LiteralPath $Folder -File |
            Where-Object { $_.Name -like "$($Attack)_*" } |
            ForEach-Object {
                Move-FileToArchive "models\zero_day_v3\$($_.Name)"
            }
    }
}

# Archive older global zero-day checkpoints.
# Keep the selected PortScan V6 checkpoint.
$OldCheckpoints = @(
    "zero_day_gnn_best.pth",
    "zero_day_gnn_v2_best.pth",
    "zero_day_gnn_v3_best.pth",
    "zero_day_gnn_v4_best.pth",
    "zero_day_gnn_v5_best.pth",
    "zero_day_v5_temporal_iforest.pkl",
    "zero_day_v5_temporal_scaler.pkl"
)

foreach ($File in $OldCheckpoints) {
    Move-FileToArchive "models\$File"
}

Write-Host "`n===== 2. ARCHIVE NON-SELECTED ATTACK REPORTS =====" -ForegroundColor Cyan

# Keep selected V2 categories.
foreach ($Attack in $V2Unselected) {
    Move-DirectoryToArchive "reports\remaining_attack_experiments\$Attack"
}

# Keep selected V3 categories.
foreach ($Attack in $V3Unselected) {
    Move-DirectoryToArchive "reports\zero_day_v3\$Attack"
}

# Archive old PortScan evaluations V1-V5.
$OldReportFiles = @(
    "zero_day_evaluation_metrics.json",
    "zero_day_flow_predictions.csv",
    "zero_day_training_history.csv",
    "zero_day_v2_evaluation_metrics.json",
    "zero_day_v2_flow_predictions.csv",
    "zero_day_v2_training_history.csv",
    "zero_day_v3_evaluation_metrics.json",
    "zero_day_v3_flow_predictions.csv",
    "zero_day_v3_pair_coverage.json",
    "zero_day_v3_training_history.csv",
    "zero_day_v4_evaluation_metrics.json",
    "zero_day_v4_flow_predictions.csv",
    "zero_day_v4_score_calibration.json",
    "zero_day_v4_training_history.csv",
    "zero_day_v5_calibration.json",
    "zero_day_v5_evaluation_metrics.json",
    "zero_day_v5_flow_predictions.csv",
    "zero_day_v5_training_history.csv"
)

foreach ($File in $OldReportFiles) {
    Move-FileToArchive "reports\$File"
}

# Archive V5 publication evaluation; retain V6 publication evaluation.
Move-DirectoryToArchive "reports\zero_day_v5_publication_evaluation"

# Archive intermediate model-comparison and traditional ML outputs.
$ComparisonReports = @(
    "model_comparison.csv",
    "model_comparison_metadata.json",
    "random_forest_21_Traffic_Features_predictions.csv",
    "random_forest_24_Traffic_Plus_Network_Metadata_predictions.csv",
    "random_forest_predictions.csv",
    "xgboost_21_Traffic_Features_predictions.csv",
    "xgboost_24_Traffic_Plus_Network_Metadata_predictions.csv",
    "xgboost_predictions.csv"
)

foreach ($File in $ComparisonReports) {
    Move-FileToArchive "reports\$File"
}

Write-Host "`n===== 3. ARCHIVE EXPLORATORY PYTHON SCRIPTS =====" -ForegroundColor Cyan

# Keep scripts needed for the selected training/evaluation pipeline.
$UnneededScripts = @(
    "src\analysis\audit_zero_day_data.py",
    "src\analysis\inspect_attack_categories.py",
    "src\analysis\zero_day_data_analysis.py",
    "src\analysis\zero_day_feature_diagnostics.py",
    "src\analysis\zero_day_model_diagnostics.py",
    "src\analysis\zero_day_temporal_analysis.py",
    "src\analysis\zero_day_traffic_analysis.py",
    "src\training\optimize_dos_hulk_v2.py",
    "src\evaluation\evaluate_zero_day_v5.py"
)

foreach ($File in $UnneededScripts) {
    Move-FileToArchive $File
}

# Archive empty analysis directory if all files were moved.
$AnalysisPath = Join-Path $Root "src\analysis"

if ((Test-Path $AnalysisPath) -and
    -not (Get-ChildItem $AnalysisPath -Force)) {
    Remove-Item -LiteralPath $AnalysisPath
}

Write-Host "`n===== 4. FINAL ACTIVE FILE CHECK =====" -ForegroundColor Cyan

$KeepFiles = @(
    "app.py",
    "src\data_preprocessing.py",
    "src\zero_day_attack_engine.py",
    "src\preprocessing\data_preprocessing.py",
    "src\preprocessing\zero_day_data_preparation.py",
    "src\preprocessing\prepare_remaining_attack_data.py",
    "src\graph\zero_day_graph_construction.py",
    "src\core\zero_day_attack_engine.py",
    "src\training\train_gnn.py",
    "src\training\train_hybrid_gnn.py",
    "src\training\train_zero_day_gnn.py",
    "src\training\train_zero_day_gnn_v3.py",
    "src\training\train_zero_day_bot.py",
    "src\training\train_zero_day_ddos.py",
    "src\training\train_zero_day_dos_goldeneye.py",
    "src\training\train_zero_day_dos_hulk.py",
    "src\training\train_zero_day_infiltration.py",
    "src\training\train_zero_day_web_attack.py",
    "src\training\run_all_attack_experiments.py",
    "src\evaluation\evaluate_zero_day_v6.py",
    "src\evaluation\final_benign_only_attack_evaluation.py",
    "src\evaluation\compare_models.py",
    "tests\test_cleaned_dataset.py",
    "tests\test_preprocessing.py"
)

foreach ($File in $KeepFiles) {
    if (Test-Path -LiteralPath (Join-Path $Root $File)) {
        Write-Host "KEPT: $File" -ForegroundColor Green
    }
    else {
        Write-Warning "EXPECTED FILE NOT FOUND: $File"
    }
}

Write-Host "`n===== CLEANUP COMPLETED =====" -ForegroundColor Green
Write-Host "Unselected files were moved into archive."
Write-Host "Selected V2, V3 and V6 experiment artifacts were retained."
Write-Host "Dataset and active root model artifacts were not moved."
