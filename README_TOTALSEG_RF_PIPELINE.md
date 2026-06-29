# End-to-end TotalSegmentator + RF TN Staging Pipeline

This document describes the end-to-end TN staging inference pipeline using TotalSegmentator anatomy features and trained Random Forest models.

## Pipeline

```text
Input CT/PET
  -> TotalSegmentator on CT
  -> head-neck anatomy feature extraction
  -> trained RF models
  -> T-stage and N-stage prediction
Required Inputs

For each case:

CASE_ID
CT_NII
PET_NII
GTVP_MASK
GTVN_MASK
OUTPUT_DIR

The current RF models assume that tumor and nodal masks are already available. These can be GT masks for upper-bound analysis or predicted masks for automated inference.

Main Script
bash scripts/run_end_to_end_totalseg_rf.sh CASE_ID CT_NII PET_NII GTVP_MASK GTVN_MASK OUTPUT_DIR

Example:

bash scripts/run_end_to_end_totalseg_rf.sh \
  TEST001 \
  /path/to/TEST001__CT.nii.gz \
  /path/to/TEST001__PT.nii.gz \
  /path/to/TEST001_gtvp.nii.gz \
  /path/to/TEST001_gtvn.nii.gz \
  outputs/end_to_end/TEST001
Scripts
scripts/run_totalseg_headneck.py
scripts/extract_totalseg_headneck_features.py
scripts/infer_tn_staging_totalseg_rf.py
scripts/run_end_to_end_totalseg_rf.sh
Trained RF Models

The default end-to-end script uses:

results/tn_staging_rf_exclude48/totalseg_pred_oof_headneck_anatomy/T/T_stage_rf_model.joblib
results/tn_staging_rf_exclude48/totalseg_pred_oof_headneck_anatomy/N/N_stage_rf_model.joblib

These RF models were trained after excluding the predefined 48 cases from TN staging training/evaluation.

The 48 cases were not removed from nnU-Net segmentation training.

Notes

TotalSegmentator is applied to CT images to generate anatomy segmentations. The extracted anatomy features are then aligned to the feature schema expected by the trained RF model.
