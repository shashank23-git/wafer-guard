##Wafer Anomaly Detector

A machine learning project focused on auditing and repairing a wafer anomaly detection pipeline using SECOM manufacturing telemetry data.

The project investigates data leakage, reconstruction-loss broadcasting errors, legacy sensor handling, and model evaluation under naturally imbalanced defect prevalence.

## Problem Statement

Manufacturing defects are rare, making anomaly detection challenging. Incorrect preprocessing, data leakage, and tensor shape mismatches can silently compromise model training and evaluation.

This project repairs these issues and evaluates whether an autoencoder can distinguish defective wafers from normal wafers on chronologically held-out data.

## Key Improvements

- Chronological train, validation, and test splits.
- Training-only median imputation and feature scaling.
- Removal of continuous features with zero variance in training data.
- Corrected autoencoder reconstruction-loss calculation.
- Categorical handling of legacy firmware codes.
- Separate handling of the legacy-station indicator.
- No oversampling of validation or test data.
- Decision threshold selected using validation labels only.
- Evaluation using naturally imbalanced test data.
- Investigation of temporal distribution shift and missingness patterns.
- Retention of debugging experiments to document the development process.

## Model

A fully connected PyTorch autoencoder reconstructs processed sensor data.

**Architecture:**

`474 → 256 → 64 → 16 → 64 → 256 → 474`

**Training configuration:**

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 50 |
| Loss function | Mean squared error |
| Anomaly score | Mean squared reconstruction error |

Higher reconstruction errors indicate samples that differ more from the patterns learned by the model.

## Dataset

The project uses SECOM manufacturing telemetry data.

| Property | Value |
|---|---:|
| Records | 1,567 |
| Original sensor features | 590 |
| Normal records | 1,463 |
| Defective records | 104 |
| Overall defect prevalence | 6.64% |
| Missing sensor values | 41,948 |

The dataset contains missing values, rare defects, and legacy sensor conventions.

### Legacy sensor handling

For legacy Probe Station PS-L records:

- `sensor_0 == 0.0` is treated as a legacy sentinel rather than a genuine continuous measurement.
- `sensor_1` firmware codes 100, 200, and 300 are represented categorically.
- A separate legacy-station indicator is preserved.

Continuous preprocessing is fitted using training data only.

Obtain the dataset from an authorized source and place the CSV file at the path expected by the notebook.

## Evaluation Method

The dataset is split chronologically to avoid training on future records.

| Split | Records | Defects | Defect prevalence |
|---|---:|---:|---:|
| Training | 1,096 | 78 | 7.12% |
| Validation | 235 | 17 | 7.23% |
| Test | 236 | 9 | 3.81% |

The anomaly threshold is selected by maximizing validation F1-score and then fixed before evaluating the test set.

The test set is not artificially balanced.

## Results

### Validation

| Metric | Result |
|---|---:|
| Selected threshold | 1.116271 |
| Precision | 0.172414 |
| Recall | 0.294118 |
| F1-score | 0.217391 |

### Test

| Metric | Result |
|---|---:|
| Test defect prevalence | 3.81% |
| ROC-AUC | 0.566814 |
| Average Precision | 0.061942 |
| Precision | 0.070175 |
| Recall | 0.444444 |
| F1-score | 0.121212 |

### Test confusion matrix

| | Predicted Normal | Predicted Defect |
|---|---:|---:|
| Actual Normal | 174 | 53 |
| Actual Defect | 5 | 4 |

The model detects 4 of the 9 defective test wafers but also flags 53 normal wafers as defective.

The ROC-AUC indicates weak discrimination, and the low precision reflects a substantial false-positive burden.

The test set contains only nine defects, so the evaluation metrics are sensitive to individual predictions.

These results are reported transparently rather than presenting the detector as production-ready.

## Repository Structure

```text
wafer-guard/
├── wafer_guard_fixed.ipynb
├── AUDIT.md
├── README.md
└── .gitignore
