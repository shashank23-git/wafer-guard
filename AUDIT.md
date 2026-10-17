# Audit Report: Wafer Anomaly Detector

## 1. Project Overview

This project audits and repairs a wafer anomaly detection pipeline using
SECOM manufacturing telemetry data.

The dataset contains 1,567 records and 590 sensor features.
The original defect prevalence is approximately 6.64%.

The objective is to identify silent implementation bugs, prevent data
leakage, handle legacy sensor data correctly, and evaluate the repaired
model under realistic class imbalance.

## 2. Silent Bugs Identified

### Bug 1: Data Leakage

**Problem:**

The original pipeline performed preprocessing and oversampling before
splitting the data. This allowed information from validation and test
records to influence training.

A random split also ignored the temporal ordering of manufacturing data.

**Repair:**

- Split the data chronologically into training, validation, and test sets.
- Fit the imputer and scaler using training data only.
- Transform validation and test data using the fitted training objects.
- Avoid oversampling the evaluation sets.
- Select the decision threshold using validation data only.

### Bug 2: Corrupted Reconstruction Loss

**Problem:**

The reconstruction output had shape `[64, 590]`, while the input batch
had shape `[64, 1, 590]`.

PyTorch broadcasting expanded these tensors during loss calculation,
producing an unintended comparison with shape `[64, 64, 590]`.

This corrupted the reconstruction loss and its gradients.

**Repair:**

The repaired autoencoder uses matching input and reconstruction shapes.
The mean squared error is calculated directly between the tensors.

The model is trained on normal training samples to learn their
reconstruction patterns.

### Bug 3: Legacy Sensor and Invariant Blindness

**Problem:**

The legacy PS-L station uses `sensor_0 == 0.0` as a sentinel rather
than a genuine measurement. Its `sensor_1` values represent categorical
firmware codes: 100, 200, and 300.

Treating these values as ordinary continuous measurements can create
misleading numerical relationships.

Features constant in the training period also require careful treatment.
A feature that is constant during training may vary in later data.

**Repair:**

- Separate legacy sentinel information from continuous sensor values.
- Represent firmware codes categorically using one-hot encoding.
- Fit preprocessing transformations using training data only.
- Exclude features constant in the training data from continuous
  reconstruction features.
- Preserve a legacy-station indicator.

This feature-selection decision has a limitation: some sensors constant
in training become variable in the test period, so their changes do not
directly contribute to the continuous reconstruction score.

## 3. Preprocessing Strategy

The repaired pipeline uses:

1. Chronological train, validation, and test splits.
2. Median imputation fitted on training data.
3. Quantile transformation fitted on training data.
4. One-hot encoding for firmware categories.
5. A legacy-station indicator.
6. No oversampling of validation or test data.

The final model input contains 471 features:
466 continuous features, 4 firmware indicator features, and 1 legacy
indicator feature.

## 4. Model

A fully connected autoencoder was implemented using PyTorch.

The architecture is:

471 -> 256 -> 128 -> 32 -> 128 -> 256 -> 471

ReLU activations are used in the hidden layers.

The model is trained using normal training samples and mean squared
reconstruction error.

The reconstruction error is used as the anomaly score: higher errors
indicate that a sample is less consistent with the learned normal
patterns.

## 5. Evaluation Method

The data was divided chronologically:

- Training: 1,095 records
- Validation: 234 records
- Test: 238 records

The test set contains 10 defective samples out of 238, giving a defect
prevalence of approximately 4.20%.

The decision threshold was selected using validation data and then
applied to the test set without further tuning.

Evaluation metrics include ROC-AUC, Average Precision, precision,
recall, F1-score, and the confusion matrix.

## 6. Results

### Validation

- ROC-AUC: 0.5754
- Average Precision: 0.0942
- Selected threshold: 1.339948
- Precision: 0.1304
- Recall: 0.3750
- F1-score: 0.1935

### Test

- Natural defect prevalence: 4.20%
- ROC-AUC: 0.5136
- Average Precision: 0.0502
- Precision: 0.0625
- Recall: 0.6000
- F1-score: 0.1132

Test confusion matrix:

| | Predicted Normal | Predicted Defect |
|---|---:|---:|
| Actual Normal | 138 | 90 |
| Actual Defect | 4 | 6 |

The model identified 6 of the 10 defective test samples but also
flagged 90 normal samples as defective.

## 7. Distribution Shift and Limitations

The model performed better on training data than on validation and
test data.

The validation ROC-AUC of 0.5754 and test ROC-AUC of 0.5136 indicate
weak discrimination on unseen chronological data.

The test Average Precision of 0.0502 is also close to the test defect
prevalence of 0.0420.

Additional diagnostics found changes in sensor distributions and
missingness patterns between training and validation periods.

These findings suggest that temporal distribution shift may contribute
to the weak generalization performance.

The test set contains only 10 defective samples, so the reported metrics
are uncertain.

The repaired pipeline addresses implementation and evaluation problems,
but it does not guarantee a highly accurate detector.

## 8. Conclusion

The audit identified data leakage, an incorrectly broadcast
reconstruction loss, and inappropriate treatment of legacy and
training-constant sensor features.

The repaired pipeline uses chronological splitting, training-only
preprocessing, categorical firmware encoding, and correctly shaped
autoencoder inputs.

Evaluation on naturally imbalanced chronological data shows that the
model still has substantial limitations.

These results demonstrate why fixing implementation bugs and using
realistic evaluation procedures are necessary, even when the resulting
model does not achieve high predictive performance.
