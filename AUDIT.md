# Audit Report:Wafer Anomaly Detector

## 1. Project Overview

WaferGuard audits and repairs a wafer anomaly detection pipeline using
SECOM manufacturing telemetry data.

The dataset contains 1,567 records, 590 sensor features, a label, and a
timestamp. It contains 104 defective records and 1,463 normal records,
giving an overall defect prevalence of approximately 6.64%.

The objective is to identify silent implementation bugs, prevent
temporal data leakage, handle legacy sensor data correctly, and evaluate
the repaired model under naturally imbalanced conditions.

The final implementation uses a PyTorch autoencoder to identify
potentially anomalous wafers through reconstruction error.

## 2. Dataset Audit

The following characteristics were identified during the initial audit:

| Property | Value |
|---|---:|
| Total records | 1,567 |
| Sensor features | 590 |
| Normal records | 1,463 |
| Defective records | 104 |
| Overall defect prevalence | 6.64% |
| Missing sensor values | 41,948 |
| Duplicate timestamps | 33 |
| Legacy records (`sensor_0 == 0.0`) | 146 |

The dataset spans July 19, 2008, through October 17, 2008.

The presence of missing sensor measurements, duplicate timestamps,
class imbalance, and legacy-station records requires careful
preprocessing and evaluation.

Duplicate timestamps are retained, and records are sorted
chronologically using a stable sort.

## 3. Silent Bugs Identified

### Bug 1: Data Leakage

**Problem**

Preprocessing performed before splitting the dataset can allow
information from future or evaluation records to influence the model.

Random splitting is also unsuitable for this task because it can mix
earlier and later manufacturing records across training and evaluation
sets.

Oversampling or threshold tuning using the test set would further
compromise the validity of the evaluation.

**Repair**

The final pipeline addresses these risks by:

- Sorting records chronologically.
- Creating chronological training, validation, and test sets.
- Fitting the median imputer using training data only.
- Determining zero-variance features using training data only.
- Fitting the robust scaler using training data only.
- Applying the fitted preprocessing transformations to validation
  and test data without refitting them.
- Selecting the anomaly threshold using validation labels only.
- Keeping the test set untouched during model and threshold selection.
- Avoiding oversampling of validation and test data.

The evaluator-only legacy reference file is not used as a model input,
for feature selection, for threshold selection, or for evaluation.

### Bug 2: Corrupted Reconstruction Loss

**Problem**

A mismatch between the dimensions of the input batch and the
reconstruction can cause PyTorch broadcasting during loss calculation.

For example, comparing tensors with shapes `[64, 590]` and
`[64, 1, 590]` can produce an unintended comparison with shape
`[64, 64, 590]`.

Such broadcasting changes the loss calculation and can corrupt the
gradients used to train the autoencoder.

**Repair**

The final implementation ensures that the input batch and reconstructed
output have matching dimensions.

The model uses mean squared error directly between these tensors:

`loss = criterion(reconstruction, batch)`

The per-wafer anomaly score is calculated by averaging the squared
reconstruction error across the feature dimension:

`((reconstruction - batch) ** 2).mean(dim=1)`

This produces one anomaly score per wafer rather than one score for
the entire batch.

The final implementation uses an input dimension of 474, and both the
input batch and reconstruction have matching feature dimensions.

### Bug 3: Legacy Sensor Handling and Invariant Blindness

**Problem**

The legacy Probe Station PS-L uses special sensor conventions:

- `sensor_0 == 0.0` is a sentinel indicating a legacy wafer rather than
  a genuine continuous measurement.
- `sensor_1` values of 100, 200, and 300 represent categorical firmware
  codes rather than continuous numerical measurements.

Treating these values as ordinary continuous measurements can introduce
misleading numerical relationships into the model.

Features that are constant during training also require careful
treatment. A feature that is constant in the training period may vary
in a later period.

**Repair**

The final preprocessing pipeline:

- Identifies legacy records using the sentinel.
- Separates legacy sentinel information from continuous sensor values.
- Treats the legacy `sensor_1` firmware codes categorically.
- Uses one-hot indicators for the firmware categories.
- Preserves an explicit legacy-station indicator.
- Determines zero-variance continuous features from training data only.
- Excludes training-constant features from continuous reconstruction
  features.
- Fits all preprocessing transformations on training data only.

The final model input contains 474 features:

- 469 continuous features.
- 4 firmware-category indicators.
- 1 legacy-station indicator.

**Limitation:** A sensor that is constant during training but changes
in a later period is excluded from the continuous reconstruction
features. Consequently, changes in such a sensor do not directly
contribute to the reconstruction error.

This is a deliberate trade-off in the current implementation, not a
guarantee that every possible sensor invariant is detected.

## 4. Preprocessing Strategy

The final preprocessing pipeline follows these steps:

1. Sort the records chronologically.
2. Split the records into training, validation, and test sets.
3. Identify legacy records using `sensor_0 == 0.0`.
4. Separate the legacy sentinel and firmware information from the
   continuous sensor measurements.
5. Fit a median imputer on the training data.
6. Remove continuous features with zero standard deviation in the
   training data.
7. Fit a `RobustScaler` on the training data.
8. Clip scaled continuous feature values to the range `[-10, 10]`.
9. Append the firmware-category indicators and legacy-station
   indicator.
10. Convert the processed arrays into tensors for PyTorch.

Validation and test data are transformed using the preprocessing
objects and feature selection determined from training data.

The final input dimension is 474 for all three splits.

No artificial balancing is applied to the evaluation sets.

## 5. Model Architecture and Training

A fully connected autoencoder is implemented using PyTorch.

The architecture is:

`474 -> 256 -> 64 -> 16 -> 64 -> 256 -> 474`

ReLU activations are used in the hidden layers.

The model learns to reconstruct the processed input features. Its
anomaly score is the mean squared reconstruction error for each wafer.

Higher reconstruction errors indicate that a wafer is less consistent
with the patterns learned by the model.

### Training configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Input features | 474 |
| Hidden encoder dimensions | 256, 64, 16 |
| Hidden decoder dimensions | 64, 256 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 50 |
| Loss function | Mean squared error |

The final recorded losses are:

| Metric | Value |
|---|---:|
| Training loss | 0.503121 |
| Validation loss | 0.835077 |

The validation loss is higher than the training loss, indicating a
generalization gap.

## 6. Chronological Evaluation

The final notebook uses a chronological 70% / 15% / 15% split.

| Split | Records | Defects | Defect prevalence |
|---|---:|---:|---:|
| Training | 1,096 | 78 | 7.12% |
| Validation | 235 | 17 | 7.23% |
| Test | 236 | 9 | 3.81% |
| Total | 1,567 | 104 | 6.64% |

The test period runs from October 5, 2008, through October 17, 2008.

The test defect prevalence is lower than the overall dataset prevalence.
This difference is retained and reported rather than artificially
balancing the test set.

The threshold is selected by maximizing the F1-score on the validation
set. Once selected, it is fixed before test evaluation.

### Validation results

| Metric | Value |
|---|---:|
| Selected threshold | 1.116271 |
| Precision | 0.172414 |
| Recall | 0.294118 |
| F1-score | 0.217391 |

The validation threshold is not selected using test labels.

## 7. Final Test Results

The final model is evaluated on the untouched chronological test set.

| Metric | Value |
|---|---:|
| Test records | 236 |
| Actual defects | 9 |
| Natural test defect prevalence | 3.81% |
| ROC-AUC | 0.566814 |
| Average Precision | 0.061942 |
| Precision | 0.070175 |
| Recall | 0.444444 |
| F1-score | 0.121212 |

### Confusion matrix

Rows represent actual classes, and columns represent predicted classes.
The class order is `[0, 1]`, where `0` is normal and `1` is defective.

| | Predicted Normal | Predicted Defect |
|---|---:|---:|
| Actual Normal | 174 | 53 |
| Actual Defect | 5 | 4 |

The model detected 4 of the 9 defective test wafers and missed 5.

It also flagged 53 normal wafers as defective.

Of the 57 wafers predicted as defective, 53 were actually normal.
Therefore, approximately 92.98% of the model's positive alerts were
false positives.

The test ROC-AUC of 0.566814 indicates weak discrimination.

The Average Precision of 0.061942 is low. Although it is above the
test defect prevalence of approximately 0.0381, the improvement is
limited and does not establish that the detector is reliable.

These results must be interpreted in the context of the small number
of defective test records.

## 8. Limitations and Distribution Shift

The final model has substantial limitations.

### Weak discrimination

The test ROC-AUC is approximately 0.567, indicating that the model
does not reliably separate normal and defective wafers.

### High false-positive rate

The selected threshold detects 4 of 9 defects but produces 53 false
positives. Such a high false-positive burden could make the detector
impractical without further improvement.

### Small number of test defects

Only 9 defective wafers appear in the test set. Consequently, each
additional true positive or false negative has a substantial effect on
recall and F1-score.

### Temporal distribution shift

The data is split chronologically, and the test period has a lower
defect prevalence than the dataset as a whole.

Changes in sensor distributions, missingness patterns, and operating
conditions may contribute to the model's weak generalization. These
are plausible explanations, not proof that distribution shift is the
sole cause of the performance limitations.

### Training-constant feature removal

Removing features that are constant during training avoids including
those features in the continuous reconstruction representation.
However, later changes in these features cannot directly influence
the anomaly score.

### Threshold dependence

The threshold is selected using validation labels to maximize F1-score.
Its performance may differ under other operating conditions or defect
prevalences.

The final implementation repairs important pipeline problems, but
these repairs do not guarantee a highly accurate anomaly detector.

## 9. Debugging History and Reproducibility

The notebook intentionally retains earlier experiments and diagnostic
cells so that the debugging process is visible.

The final corrected implementation follows these experiments and
contains the preprocessing, training, threshold selection, and
evaluation used to report the final results.

For reproducibility:

1. Open `wafer_guard_fixed.ipynb` in a compatible Jupyter environment
   or Google Colab.
2. Provide the main SECOM telemetry CSV at the path expected by the
   notebook.
3. Execute the notebook cells in order.
4. Review the final preprocessing dimensions, training losses,
   validation threshold, and test metrics.

The main dataset is used for model development and evaluation.
The evaluator-only legacy reference file must not be used as an input
to the modeling workflow and must not be included in the public
repository.

## 10. Conclusion

The audit addresses three important implementation risks:

1. Data leakage caused by inappropriate splitting or preprocessing.
2. Corrupted reconstruction loss caused by incompatible tensor shapes
   and unintended broadcasting.
3. Incorrect handling of legacy sentinel values, categorical firmware
   codes, and training-constant features.

The final implementation uses chronological data splitting,
training-only preprocessing, categorical firmware representation,
matching autoencoder input and output dimensions, and validation-only
threshold selection.

Evaluation on naturally imbalanced chronological data produces a test
ROC-AUC of 0.566814 and an F1-score of 0.121212.

The detector still produces many false positives and misses several
defective wafers. These results demonstrate that correcting
implementation and evaluation errors is necessary, but not sufficient,
to build a reliable anomaly detection system.

Further work is required to improve discrimination, reduce false
positives, and evaluate robustness across manufacturing periods.
