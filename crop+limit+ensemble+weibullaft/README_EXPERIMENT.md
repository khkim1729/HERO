# crop+limit+ensemble+weibullaft

Grand Challenge validation experiment package.

## Experiment definition

This version combines:

- crop: apply head-and-neck crop first for oversized CT cases
- limit: if the cropped case is still larger than 100,000,000 voxels, return zero/default segmentation fallback
- ensemble: use all available nnU-Net fold_* directories
- TN staging leakage fix: T/N artifacts do not contain clinical_Relapse or clinical_RFS
- RFS model: WeibullAFT instead of CoxPH

## Upload tarballs

The actual Grand Challenge upload tarballs are not committed to GitHub because they are large.

Local source package:

- /home/introai17/salamanca/gc_uploads/cropfirst_fallback_foldens_tnstage_fixed_weibullaft_rfs_20260709_205111/container.tar.gz
- /home/introai17/salamanca/gc_uploads/cropfirst_fallback_foldens_tnstage_fixed_weibullaft_rfs_20260709_205111/model.tar.gz

## Included here

- Dockerfile
- inference.py
- requirements.txt
- small TN staging artifacts
- small RFS WeibullAFT artifacts
- experiment notes

## RFS

Configured as WeibullAFT via:

- model/rfs/rfs_model_config.json
- model/rfs/weibull_model.pkl
- model/rfs/scaler.pkl

## TN staging

Cleaned T/N staging artifacts with clinical_Relapse and clinical_RFS removed.

## Segmentation

Oversized case logic:

1. If CT voxel_count > 100,000,000, crop to head-and-neck region first.
2. If cropped volume is still > 100,000,000 voxels, return zero/default segmentation.
3. Otherwise run nnU-Net using all available folds.
