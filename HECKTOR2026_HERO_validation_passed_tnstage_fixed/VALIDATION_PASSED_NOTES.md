# HECKTOR2026 HERO validation-passed package

This folder contains the code used to build the Grand Challenge container that passed validation.

## Important

The validation-passed submission used:

- container built from this Dockerfile/inference.py
- model.tar.gz containing CoxPH RFS model
- huge-case skip / failsafe inference logic

## Huge case policy

The validation-passed version skips nnU-Net segmentation when:

```text
CT voxel_count > 100,000,000
```

where:

```text
voxel_count = CT_size_x * CT_size_y * CT_size_z
```

For huge cases, the code writes a valid zero segmentation fallback to avoid CPU RAM failure.

## RFS model

The validation-passed model.tar.gz used CoxPH, not WeibullAFT.

Expected files inside model.tar.gz:

```text
rfs/coxph_model.pkl
rfs/scaler.pkl
rfs/rfs_model_config.json
```

Expected config:

```text
model_type: CoxPH
```

## Build container tar

```bash
./build_container.sh hecktor2026-task hecktor2026-task.tar.gz
```

## Pack model tar

```bash
./pack_model.sh model model.tar.gz
```

Do not commit large tar files to GitHub.
