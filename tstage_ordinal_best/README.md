# T-stage Ordinal RF Ensemble

Current best fair OOF T-stage model for the TN staging pipeline.

## Method

Ordinal Random Forest probability ensemble:

- P(T >= T2)
- P(T >= T3)
- P(T >= T4)

Branches:

- Pet-aware OOF predicted-mask branch weight = 0.80
- Final OOF predicted-mask branch weight = 0.20

Decision rule:

    if P(T>=T4) >= 0.35: T4
    elif P(T>=T3) >= 0.44: T3
    elif P(T>=T2) >= 0.64: T2
    else: T1

## Fair OOF performance

| Metric | Value |
|---|---:|
| Balanced accuracy | 0.581272 |
| Macro recall | 0.581272 |
| Macro F1 | 0.564200 |
| T1 recall | 0.724551 |
| T2 recall | 0.508961 |
| T3 recall | 0.450549 |
| T4 recall | 0.641026 |
| T4 precision | 0.543478 |
| T4 F1 | 0.588235 |
| Predicted T4 count | 138 |

## Best setting

- pair: fair_hu_geom_petaware_final
- weight_petaware: 0.80
- weight_final: 0.20
- thr_ge_T2: 0.64
- thr_ge_T3: 0.44
- thr_ge_T4: 0.35

## Usage

Use scripts/predict_from_ordinal_probabilities.py to ensemble two ordinal probability tables.

Required columns in both input CSV files:

- PatientID
- p_ge_T2
- p_ge_T3
- p_ge_T4

Do not commit raw CT/PET images or NIfTI segmentation masks.
