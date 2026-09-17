# Summary tables — LOCO revision (10 seeds, condition-level bootstrap)

Every value is the median over the 10 seeds (42-51) of the per-seed point estimate. 
Bracketed intervals are 2.5/97.5 percentiles of the condition-level bootstrap, 
B = 2000 resamples per seed pooled over the 10 seeds (20 000 draws). 
Conditions, not wafers, are the resampling unit; every model and every training 
mode within a domain sees the identical condition draws, so all differences are paired. 
R² is dimensionless; RMSE and MAE are in nm.

## Table 1 — LOCO before and after the D-optimal augmentation

`before` = 42 OFAT wafers (n_added = 0), `after` = 42 OFAT + 15 D-optimal wafers 
(n_added = 15), both under leave-one-condition-out CV on the 4-position mean thickness. 
IQR is the seed-to-seed interquartile range of the point estimate.

| Model | Metric | Before (n_added = 0) | 95 % CI | seed IQR | After (n_added = 15) | 95 % CI | seed IQR |
|---|---|---|---|---|---|---|---|
| Ridge | R² | 0.44 | [-0.18, 0.74] | 0.41–0.48 | 0.75 | [0.57, 0.86] | 0.74–0.75 |
| Ridge | RMSE (nm) | 5.63 | [3.49, 7.96] | 5.43–5.80 | 4.49 | [3.23, 6.17] | 4.47–4.54 |
| Ridge | MAE (nm) | 3.87 | [2.65, 5.46] | 3.72–3.94 | 3.17 | [2.47, 4.14] | 3.14–3.24 |
| Random Forest | R² | 0.65 | [0.42, 0.79] | 0.64–0.67 | 0.79 | [0.67, 0.89] | 0.78–0.81 |
| Random Forest | RMSE (nm) | 4.44 | [2.90, 6.16] | 4.35–4.50 | 4.03 | [2.84, 5.24] | 3.88–4.13 |
| Random Forest | MAE (nm) | 3.24 | [2.14, 4.32] | 3.18–3.29 | 2.81 | [2.03, 3.67] | 2.68–2.92 |
| XGBoost | R² | 0.57 | [0.32, 0.72] | 0.56–0.58 | 0.76 | [0.62, 0.86] | 0.74–0.76 |
| XGBoost | RMSE (nm) | 4.95 | [3.49, 6.69] | 4.91–5.00 | 4.39 | [3.20, 5.68] | 4.34–4.53 |
| XGBoost | MAE (nm) | 3.78 | [2.78, 4.84] | 3.73–3.82 | 3.23 | [2.37, 4.15] | 3.17–3.34 |
| GPR | R² | 0.23 | [-1.01, 0.69] | 0.23–0.23 | 0.77 | [0.63, 0.87] | 0.77–0.77 |
| GPR | RMSE (nm) | 6.65 | [3.86, 9.60] | 6.65–6.65 | 4.27 | [3.06, 5.72] | 4.27–4.27 |
| GPR | MAE (nm) | 4.26 | [2.90, 6.28] | 4.26–4.26 | 2.98 | [2.27, 3.83] | 2.98–2.98 |

## Table 2 — LOOCV versus LOCO on identical data and pipelines

Same seeds, same nested-CV pipelines and grids; only the outer split differs. 
OFAT = 42 wafers / 24 conditions, Augmented = 57 wafers / 39 conditions.

| Model | Subset | CV | median R² | median RMSE (nm) | ΔR² (LOCO − LOOCV) |
|---|---|---|---|---|---|
| Ridge | OFAT | LOOCV | 0.48 | 5.45 |  |
| Ridge | OFAT | LOCO | 0.44 | 5.63 | -0.03 |
| Ridge | Augmented | LOOCV | 0.75 | 4.46 |  |
| Ridge | Augmented | LOCO | 0.75 | 4.49 | 0.00 |
| Random Forest | OFAT | LOOCV | 0.68 | 4.30 |  |
| Random Forest | OFAT | LOCO | 0.65 | 4.44 | -0.02 |
| Random Forest | Augmented | LOOCV | 0.81 | 3.91 |  |
| Random Forest | Augmented | LOCO | 0.79 | 4.03 | -0.01 |
| XGBoost | OFAT | LOOCV | 0.61 | 4.73 |  |
| XGBoost | OFAT | LOCO | 0.57 | 4.95 | -0.04 |
| XGBoost | Augmented | LOOCV | 0.77 | 4.26 |  |
| XGBoost | Augmented | LOCO | 0.76 | 4.39 | -0.01 |
| GPR | OFAT | LOOCV | 0.25 | 6.54 |  |
| GPR | OFAT | LOCO | 0.23 | 6.65 | -0.02 |
| GPR | Augmented | LOOCV | 0.78 | 4.20 |  |
| GPR | Augmented | LOCO | 0.77 | 4.27 | -0.01 |

## Table 3 — Same-test-set A/B comparison

For every one of the 39 conditions, model A trains on the OFAT wafers outside that 
condition and model B on all 57 wafers outside it; both predict the same held-out rows. 
Metrics are therefore computed on identical test rows.

### 3a. Per-mode performance

| Model | Test domain | Mode | median R² | median RMSE (nm) | median MAE (nm) |
|---|---|---|---|---|---|
| Ridge | OFAT rows | A | 0.44 | 5.63 | 3.87 |
| Ridge | OFAT rows | B | 0.61 | 4.72 | 3.18 |
| Ridge | DOE rows | A | 0.86 | 4.47 | 3.80 |
| Ridge | DOE rows | B | 0.90 | 3.75 | 3.12 |
| Ridge | all 57 | A | 0.61 | 5.54 | 3.90 |
| Ridge | all 57 | B | 0.75 | 4.49 | 3.17 |
| Random Forest | OFAT rows | A | 0.65 | 4.44 | 3.24 |
| Random Forest | OFAT rows | B | 0.67 | 4.32 | 3.07 |
| Random Forest | DOE rows | A | 0.89 | 3.91 | 3.04 |
| Random Forest | DOE rows | B | 0.94 | 2.96 | 2.06 |
| Random Forest | all 57 | A | 0.77 | 4.30 | 3.19 |
| Random Forest | all 57 | B | 0.79 | 4.03 | 2.81 |
| XGBoost | OFAT rows | A | 0.57 | 4.95 | 3.78 |
| XGBoost | OFAT rows | B | 0.62 | 4.67 | 3.33 |
| XGBoost | DOE rows | A | 0.81 | 5.16 | 4.23 |
| XGBoost | DOE rows | B | 0.91 | 3.58 | 2.84 |
| XGBoost | all 57 | A | 0.68 | 4.99 | 3.90 |
| XGBoost | all 57 | B | 0.76 | 4.39 | 3.23 |
| GPR | OFAT rows | A | 0.23 | 6.65 | 4.26 |
| GPR | OFAT rows | B | 0.65 | 4.47 | 3.18 |
| GPR | DOE rows | A | 0.11 | 11.21 | 10.13 |
| GPR | DOE rows | B | 0.91 | 3.64 | 2.40 |
| GPR | all 57 | A | 0.17 | 8.10 | 5.80 |
| GPR | all 57 | B | 0.77 | 4.27 | 2.98 |

### 3b. Paired differences (B − A) on identical resampled conditions

`favour B` is the fraction of the 20 000 paired bootstrap draws in which B beats A 
(higher R², lower RMSE / MAE).

| Model | Test domain | ΔR² | 95 % CI | favour B | ΔRMSE (nm) | 95 % CI | favour B | ΔMAE (nm) | 95 % CI | favour B |
|---|---|---|---|---|---|---|---|---|---|---|
| Ridge | OFAT rows | 0.17 | [-0.02, 0.66] | 0.94 | -0.91 | [-2.47, 0.13] | 0.94 | -0.64 | [-1.67, 0.07] | 0.96 |
| Ridge | DOE rows | 0.05 | [-0.01, 0.27] | 0.95 | -0.78 | [-3.36, 0.15] | 0.95 | -0.73 | [-3.27, 0.28] | 0.93 |
| Ridge | all 57 | 0.13 | [0.01, 0.33] | 0.99 | -1.03 | [-2.15, -0.10] | 0.99 | -0.72 | [-1.71, -0.09] | 0.99 |
| Random Forest | OFAT rows | 0.01 | [-0.05, 0.16] | 0.66 | -0.09 | [-1.00, 0.32] | 0.66 | -0.10 | [-0.78, 0.37] | 0.65 |
| Random Forest | DOE rows | 0.04 | [-0.01, 0.12] | 0.96 | -0.76 | [-1.93, 0.10] | 0.96 | -0.92 | [-1.71, -0.23] | 0.99 |
| Random Forest | all 57 | 0.02 | [-0.02, 0.12] | 0.91 | -0.21 | [-1.05, 0.14] | 0.91 | -0.35 | [-0.93, 0.12] | 0.93 |
| XGBoost | OFAT rows | 0.05 | [-0.05, 0.15] | 0.87 | -0.28 | [-0.82, 0.27] | 0.87 | -0.41 | [-0.98, 0.11] | 0.93 |
| XGBoost | DOE rows | 0.10 | [0.02, 0.20] | 0.99 | -1.64 | [-2.74, -0.27] | 0.99 | -1.39 | [-2.64, -0.02] | 0.98 |
| XGBoost | all 57 | 0.07 | [0.00, 0.14] | 0.98 | -0.60 | [-1.20, -0.04] | 0.98 | -0.63 | [-1.24, -0.14] | 0.99 |
| GPR | OFAT rows | 0.42 | [0.01, 1.59] | 0.98 | -2.18 | [-5.07, -0.05] | 0.98 | -1.08 | [-2.70, -0.04] | 0.98 |
| GPR | DOE rows | 0.80 | [0.62, 1.31] | 1.00 | -7.57 | [-9.99, -5.26] | 1.00 | -7.73 | [-10.47, -4.69] | 1.00 |
| GPR | all 57 | 0.60 | [0.31, 1.12] | 1.00 | -3.84 | [-6.09, -2.04] | 1.00 | -2.83 | [-4.75, -1.46] | 1.00 |

## Table 4 — Five-position target sensitivity

Same LOCO folds and pipelines, target swapped from the 4-position mean thickness to 
the 5-position mean. The 4-position columns are the LOCO rows of `metrics_loocv_ref.csv`.

| Model | Subset | 5-pos R² | 95 % CI | 4-pos R² | 5-pos RMSE (nm) | 95 % CI | 4-pos RMSE (nm) |
|---|---|---|---|---|---|---|---|
| Ridge | OFAT | 0.57 | [0.07, 0.75] | 0.44 | 5.05 | [3.67, 6.43] | 5.63 |
| Ridge | Augmented | 0.76 | [0.63, 0.85] | 0.75 | 4.41 | [3.44, 5.34] | 4.49 |
| Random Forest | OFAT | 0.64 | [0.42, 0.76] | 0.65 | 4.63 | [3.07, 5.95] | 4.44 |
| Random Forest | Augmented | 0.78 | [0.66, 0.88] | 0.79 | 4.21 | [2.99, 5.27] | 4.03 |
| XGBoost | OFAT | 0.55 | [0.30, 0.68] | 0.57 | 5.13 | [3.65, 6.43] | 4.95 |
| XGBoost | Augmented | 0.74 | [0.60, 0.85] | 0.76 | 4.63 | [3.26, 5.80] | 4.39 |
| GPR | OFAT | 0.44 | [-0.16, 0.64] | 0.23 | 5.75 | [4.26, 7.10] | 6.65 |
| GPR | Augmented | 0.74 | [0.62, 0.82] | 0.77 | 4.61 | [3.75, 5.35] | 4.27 |

## Table 5 — Incremental D-optimal trajectory

Median LOCO R² as the 15 D-optimal wafers are appended in D-optimal selection order. 
Bootstrap CIs are given at n_added = 0, 5 and 15.

| Model | n = 0 | n = 1 | n = 3 | n = 5 | n = 8 | n = 10 | n = 15 | closed fraction |
|---|---|---|---|---|---|---|---|---|
| Ridge | 0.44 [-0.18, 0.74] | 0.60 | 0.66 | 0.69 [0.47, 0.84] | 0.69 | 0.73 | 0.75 [0.57, 0.86] | 0.54 |
| Random Forest | 0.65 [0.42, 0.79] | 0.66 | 0.71 | 0.74 [0.59, 0.86] | 0.75 | 0.77 | 0.79 [0.67, 0.89] | 0.41 |
| XGBoost | 0.57 [0.32, 0.72] | 0.61 | 0.66 | 0.69 [0.51, 0.81] | 0.68 | 0.72 | 0.76 [0.62, 0.86] | 0.43 |
| GPR | 0.23 [-1.01, 0.69] | 0.26 | 0.67 | 0.76 [0.60, 0.87] | 0.70 | 0.74 | 0.77 [0.63, 0.87] | 0.70 |

`closed fraction` = (R²(15) − R²(0)) / (1 − R²(0)): the share of the residual 
unexplained variance at n_added = 0 that the 15 D-optimal wafers remove.

## Table 6 — OFAT sequential baseline

Median LOCO R² as OFAT wafers accumulate in file order, with the number of distinct 
process conditions available at each n.

| Model | n = 12 | n = 20 | n = 30 | n = 42 |
|---|---|---|---|---|
| Ridge | 0.30 | 0.04 | 0.29 | 0.44 |
| Random Forest | 0.15 | 0.34 | 0.58 | 0.65 |
| XGBoost | 0.37 | 0.25 | 0.46 | 0.57 |
| GPR | -0.15 | 0.11 | -0.20 | 0.23 |
| **unique conditions** | 12 | 15 | 18 | 24 |

## Observations

Of the 36 paired same-test-set differences, 19 have a 95 % interval that excludes zero: all nine GPR cells, all three Ridge cells on the full 57 rows, the XGBoost cells on the DOE rows and on the full 57 rows, and the Random Forest MAE on the DOE rows. The Ridge differences on the OFAT rows and on the DOE rows taken separately straddle zero even though 94 % and 95 % of draws favour B, which is what a 24-condition and a 15-condition resample buys; only the pooled 39-condition domain separates. Random Forest is the model the augmentation moves least, ΔR² = 0.02 [-0.02, 0.12] on all 57 rows, and it is the only model whose R² interval on the full test set overlaps zero improvement in every domain. GPR is the opposite extreme: its ΔR² upper bounds run past 1.00 (1.59 on the OFAT rows, 1.31 on the DOE rows) because mode A, trained on OFAT wafers alone, produces deeply negative R² on many resamples, so the difference is not bounded by 1. Interval widths track condition count far more than wafer count. The widest interval anywhere is GPR on the 24-condition OFAT subset under LOCO, R² = 0.23 spanning [-1.01, 0.69], a width of 1.70; the same model on the 39-condition augmented subset narrows to a width of 0.25, and every augmented R² interval in Table 1 sits between 0.19 and 0.29 wide. Switching to the five-position target (Table 4) halves that GPR OFAT width to 0.79 and lifts the median from 0.23 to 0.44, the only place where the target definition changes a conclusion; every other five-position value is within 0.04 R² of its four-position counterpart. The incremental trajectory is not monotonic for any model: all four dip at n_added = 8 and again at n_added = 12, and GPR falls furthest, from 0.76 at n_added = 5 to 0.70 at n_added = 8 before recovering to 0.77 at n_added = 15. GPR also closes the largest share of residual variance (0.70) purely because it starts lowest, while Random Forest closes the least (0.41) from the highest starting point. The OFAT sequential baseline (Table 6) is far more erratic than the D-optimal trajectory: GPR is negative at every n from 6 to 15 and again from 21 to 32, and Ridge peaks at 0.34 at n = 13, falls to 0.02 at n = 17, and does not beat that early peak until n = 35. Adding 30 OFAT wafers raises the distinct-condition count only from 12 to 24, which is the plainest reading of why the curve stalls. Finally, LOCO is at or below LOOCV everywhere (Table 2) but by at most 0.04 R², so the optimism of leaving out one wafer at a time is real but small at this sample size.
