# Wafer Guard — Wafer Anomaly Detector

A machine learning project focused on auditing and repairing a wafer anomaly detection pipeline using SECOM manufacturing telemetry data.

The project investigates data leakage, reconstruction-loss broadcasting errors, legacy sensor handling, and model evaluation under naturally imbalanced defect prevalence.

## Problem Statement

Manufacturing defects are rare, making anomaly detection challenging. Incorrect preprocessing, data leakage, and tensor shape mismatches can silently compromise model training and evaluation.

This project repairs these issues and evaluates whether an autoencoder can distinguish defective wafers from normal wafers on chronologically held-out data.

## Key Improvements

- Chronological train, validation, and test splits.
- Training-only imputation and feature scaling.
- Corrected autoencoder reconstruction-loss calculation.
- Categorical handling of legacy firmware codes.
- Separate handling of the legacy sensor indicator.
- No oversampling of validation or test data.
- Decision threshold selected using validation data.
- Evaluation using naturally imbalanced test data.
- Investigation of temporal distribution shift and missingness patterns.

## Model

A fully connected PyTorch autoencoder learns reconstruction patterns from normal training samples.

**Architecture:**

`471 → 256 → 128 → 32 → 128 → 256 → 471`

Mean squared reconstruction error is used as the anomaly score.

Higher reconstruction errors indicate samples that differ more from the patterns learned by the model.

## Dataset

The project uses SECOM manufacturing telemetry data.

- Records: 1,567
- Original sensor features: 590
- Defect prevalence: approximately 6.64%

The dataset contains missing values, rare defects, and legacy sensor conventions.

Obtain the dataset from an authorized source and place `secom_telemetry_modified.csv` in the notebook's working directory.

## Results

### Validation

| Metric | Result |
|---|---:|
| ROC-AUC | 0.5754 |
| Average Precision | 0.0942 |
| Precision | 0.1304 |
| Recall | 0.3750 |
| F1-score | 0.1935 |

### Test

| Metric | Result |
|---|---:|
| Defect prevalence | 4.20% |
| ROC-AUC | 0.5136 |
| Average Precision | 0.0502 |
| Precision | 0.0625 |
| Recall | 0.6000 |
| F1-score | 0.1132 |

The test set contains only 10 defective samples. The model detects 6 of them but also generates many false positives.

The results indicate weak generalization to unseen chronological data. They are reported honestly rather than presenting the detector as production-ready.

## Repository Structure

```text
wafer-guard/
├── wafer_guard_fixed.ipynb
├── AUDIT.md
├── README.md
└── .gitignore
