# crop_limit_ensemble_tnfixed_smallpp1000_nordinal_coxph

Grand Challenge validation package for HECKTOR2026 HERO.

This folder overwrites the previous HECKTOR2026_HERO_validation_passed package with the latest experiment.

## Experiment definition

- crop first for oversized CT cases
- if cropped volume is still larger than 100,000,000 voxels, return zero/default segmentation fallback
- nnU-Net fold ensemble using all available fold_* directories
- remove predicted tumor connected components smaller than 1000 voxels
- T-stage leakage fixed
- N-stage ordinal RandomForest probability model
- RFS model: CoxPH
- clinical_Relapse and clinical_RFS are not used in TN staging artifacts or inference placeholders

## Grand Challenge upload files

The actual upload tarballs are not committed to GitHub because they are large.

Local package source:

/home/introai17/salamanca/gc_uploads/crop_limit_ensemble_tnfixed_smallpp1000_nordinal_coxph_20260710_024700

Expected upload files:

- container.tar.gz
- model.tar.gz

## Included in this GitHub folder

- Dockerfile
- inference.py
- requirements.txt
- small TN staging artifacts
- small RFS CoxPH artifacts
- manifests and notes

## N-stage OOF summary

- balanced accuracy: 0.6930878088110322
- macro recall: 0.6930878088110322
- macro F1: 0.6214316163830734
- note: N3 is over-predicted in OOF compared with the true distribution

## Package title

crop_limit_ensemble_tnfixed_smallpp1000_nordinal_coxph
