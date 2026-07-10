# notta_step04_smallpp300_tcompat_nordinal_coxph

Grand Challenge validation package for HECKTOR2026 HERO.

This folder overwrites the previous HECKTOR2026_HERO_validation_passed package with the latest validation experiment.

## Experiment definition

- TTA OFF via --disable_tta
- nnU-Net sliding window step_size 0.4
- small component removal threshold: 300 voxels
- crop first for oversized CT cases
- if cropped volume is still larger than 100,000,000 voxels, return zero/default segmentation fallback
- nnU-Net fold ensemble using all available fold_* directories
- T-stage: inference-compatible 47-feature ordinal RandomForest
- N-stage: ordinal RandomForest probability model
- TN staging leakage fixed
- RFS model: CoxPH, original score direction retained
- clinical_Relapse and clinical_RFS are not used in TN staging artifacts or inference placeholders

## Grand Challenge upload files

The actual upload tarballs are not committed to GitHub because they are large.

Local package source:

/home/introai17/salamanca/gc_uploads/notta_step04_smallpp300_tcompat_nordinal_coxph_20260711_021058

Expected upload files:

- container.tar.gz
- model.tar.gz

## Included in this GitHub folder

- Dockerfile
- inference.py
- requirements.txt
- pack_model.sh if present
- small TN staging artifacts
- small RFS CoxPH artifacts
- manifests and notes

## T-stage OOF summary

- model: tstage_ordinal_rf_inference_compatible
- n_features: 47
- balanced accuracy: 0.576957275334714
- macro recall: 0.576957275334714
- macro F1: 0.5634032784009771

## N-stage OOF summary

- model: nstage_ordinal_rf_probability
- balanced accuracy: 0.6930878088110322
- macro recall: 0.6930878088110322
- macro F1: 0.6214316163830734
- note: N3 is over-predicted in OOF compared with the true distribution

## Package title

notta_step04_smallpp300_tcompat_nordinal_coxph
