# HECKTOR2026 HERO validation-passed inference package

This folder contains the container source files used for the validation-passed HERO Grand Challenge submission.

## Contents

```text
Dockerfile
inference.py
requirements.txt
.dockerignore
build_container.sh
pack_model.sh
VALIDATION_PASSED_NOTES.md
artifact_checks/
Build container tar
./build_container.sh hecktor2026-task hecktor2026-task.tar.gz
Pack model tar

Prepare a model/ directory with:

model/nnunet/
model/tn_staging/
model/rfs/

Then run:

./pack_model.sh model model.tar.gz
Validation-passed RFS

The validation-passed submission used CoxPH RFS files:

rfs/coxph_model.pkl
rfs/scaler.pkl
rfs/rfs_model_config.json

It did not use WeibullAFT for the submitted validation-passed model archive.

Huge-case handling

The validation-passed inference code skips nnU-Net when:

CT voxel_count > 100,000,000

This was used to avoid CPU RAM failure on very large validation cases.
