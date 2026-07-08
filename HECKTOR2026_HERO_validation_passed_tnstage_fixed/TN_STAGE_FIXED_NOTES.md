# TN staging leakage-fixed package

This folder is based on the previous validation-passed HECKTOR2026 HERO inference package.

## What changed

The TN staging artifacts were rebuilt to remove outcome-derived clinical columns from model features:

- clinical_Relapse
- clinical_RFS

These columns are not available at inference time.

## Active artifacts included here

```text
model/tn_staging/T/T_stage_rf_model.joblib
model/tn_staging/N/N_stage_rf_model.joblib
T-stage

Model type:

tstage_ordinal_rf_probability_ensemble

Structure:

branches:
  - petaware
  - final

ordinal classifiers:
  - T_ge_2
  - T_ge_3
  - T_ge_4

weights:
  petaware = 0.8
  final    = 0.2

thresholds:
  T_ge_2 = 0.64
  T_ge_3 = 0.44
  T_ge_4 = 0.35

Clean OOF run:

n_train = 745
petaware n_features = 217
final n_features = 217
OOF balanced accuracy = 0.5637
OOF macro recall = 0.5637
OOF macro F1 = 0.5316
N-stage

Model type:

RandomForestClassifier package with model + feature_columns

Clean OOF run:

n_rows = 751
n_features = 47
OOF balanced accuracy = 0.6201870091380036
OOF macro recall = 0.6201870091380036
Verification

Both active TN artifacts were checked with binary scan:

clinical_Relapse: 0
clinical_RFS: 0

