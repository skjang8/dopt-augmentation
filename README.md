# D-optimal augmentation of an existing experimental record

Data and code for *D-Optimal Augmentation of OFAT Datasets for Machine
Learning-Based Thickness Prediction in HfO2 PEALD*.

The routine proposes the next experiments for a record whose conditions are
known. **It reads no measured response**, so it runs on any process, and the
whole set of additions can be fixed before the first of them is performed.

## Install

    pip install -e .

numpy and pandas are the only dependencies.

## Propose experiments for your own record

Write a config naming the CSV of experiments already performed and a candidate
range per parameter:

```toml
name = "my process"
conditions = "path/to/my_conditions.csv"
levels = 7                # grid levels per parameter
round_to_integer = true   # round proposals to whole numbers

[[parameters]]
column = "chuck_temperature_C"
min = 100
max = 300
# one block per parameter
```

Columns not listed are ignored, so a file that also carries measured responses
can be used as it stands.

    python -m dopt.select --config config/thiswork_hfo2_peald.toml --n-add 15

After each addition the routine prints the rank, log10 det(X'X), the condition
number and mean |r|, so where to stop is your decision and not the routine's.

Two worked configs are included:

| config | record | result |
|---|---|---|
| `thiswork_hfo2_peald.toml` | 42 OFAT experiments, HfO2 on an 8-inch cross-flow PEALD | rank 13 → 21 in **8** additions |
| `yoon2023_pt_ald.toml` | 17 screening experiments, Pt on a rotary ALD | rank 9 → 15 in **6** additions |

The first starts from the record reported in the paper. The second is a different
material, precursor, substrate and reactor, and exists to show that nothing in
the routine is specific to one process.

## The design quantities

| quantity | basis | threshold |
|---|---|---|
| det(X'X) | 21-term quadratic model matrix | **none** — exactly zero while any model term is not estimable, and it is the quantity the criterion maximises |
| κ = cond(X'X) | standardized intercept + 5 factors | a value must be chosen |
| mean \|r\| | standardized 5 factors | a value must be chosen |

`rank` is reported alongside `det`; they carry the same statement, since det = 0
exactly when rank < P.

    python -m dopt.trajectory

prints these over the 15 additions made in this work, the additions at which
each is first attained, and the steps at which κ and mean |r| rise rather than
fall. The eight terms not identifiable from the 42 OFAT conditions are
x4², x5², x2x3, x2x4, x2x5, x3x4, x3x5, x4x5.

## Cross-validation and uncertainty

Predictive performance is evaluated by leave-one-condition-out cross-validation.
Every wafer that shares a process condition is held out in the same fold, so a
condition that was run more than once never sits on both sides of the split. The
42 OFAT wafers cover 24 conditions and the 57 wafers cover 39. Every fit is
repeated over ten seeds, 42 to 51, and the reported value is the median over
them. Four regressors are compared: Ridge, Random Forest, XGBoost and a Gaussian
process, each tuned by a grid search inside the training fold.

Three further comparisons use the same folds and the same pipelines.

The same-test-set comparison isolates what the added conditions are worth. For
each held-out condition, one model is trained on the remaining OFAT wafers and
one on the remaining OFAT and D-optimal wafers, and both predict the same
held-out wafers. The two are therefore read on identical rows.

The condition-level bootstrap gives the interval. Conditions, not wafers, are
resampled with replacement from the saved out-of-fold predictions, 2 000 draws
per seed pooled over the ten seeds, and the interval is the 2.5 and 97.5
percentiles of the pooled draws. Every model and every training mode within a
domain sees the identical draws, so the differences between them are paired.

The five-position sensitivity repeats the folds with the mean of all five wafer
positions as the target instead of the mean of the four retained ones.

Two further parts repeat the cross-validation as the record grows: `dopt` appends
the 15 D-optimal wafers one at a time in selection order, and `ofatseq` lets the
OFAT wafers accumulate in file order. They are what Figure 3 draws.

    pip install -e ".[cv]"
    python -m cv.pipeline --parts dopt ofatseq sametest fivepos loocv_ref \
        --seeds 42 43 44 45 46 47 48 49 50 51
    python -m cv.bootstrap
    python -m cv.tables
    python -m cv.figure_data
    python -m cv.figures

The first command is the whole cost, about four hours on 24 cores for all parts
and all seeds. Its metrics and out-of-fold predictions are in `results/cv/`, so
the bootstrap, the tables and the figures run from those in seconds and nothing
is refitted. The pipeline appends as it goes and resumes from what is already
written, so an interrupted run can be repeated.

Median R² and RMSE under leave-one-condition-out cross-validation, before the
augmentation (42 wafers, 24 conditions) and after it (57 wafers, 39 conditions):

| model | R² before | R² after | RMSE before (nm) | RMSE after (nm) |
|---|---|---|---|---|
| Ridge | 0.44 | 0.75 | 5.63 | 4.49 |
| Random Forest | 0.65 | 0.79 | 4.44 | 4.03 |
| XGBoost | 0.57 | 0.76 | 4.95 | 4.39 |
| GPR | 0.23 | 0.77 | 6.65 | 4.27 |

`results/cv/SUMMARY_TABLES.md` carries the full tables, including the bootstrap
confidence intervals, the leave-one-out comparison and the paired same-test-set
differences.

    cv/                      the cross-validation, bootstrap and figure code
    results/cv/              metrics, out-of-fold predictions and intervals
    results/cv/figure_data/  the reduced inputs the figures read
    figures/                 the four figures drawn from them

## Data

    data/ofat_clean.csv          42 OFAT conditions with the measured thickness
    data/doe_clean.csv           15 D-optimal conditions with the measured thickness
    data/combined_clean.csv      the 57 together
    data/doe_doptimal_order.csv  the order in which the 15 were selected
    data/thickness_positions.csv the 57 with all five measurement positions
    data/yoon2023_screening.csv  17 screening conditions, Pt rotary ALD
    data/figure_data/            design and attribution inputs the figures read

The HfO2 files carry the five process parameters, the film thickness at four
positions across the wafer, and their mean, which is the prediction target. The
Pt record carries conditions only, which is all the routine reads; it is from
our earlier work, Yoon et al., Langmuir 39 (2023) 4984.

`data/thickness_positions.csv` the same 57 wafers with the thickness at all five
measurement positions T1–T5; T2, adjacent to the precursor outlet, is excluded
from the prediction target (Section 4.1 of the manuscript), so Thickness_mean is
the mean of T1, T3, T4 and T5 and Thickness_5pos the mean of all five.

`dopt.select` scores the full grid and is deterministic. The 15 conditions in
`data/` are the ones actually performed, and the design quantities reported in
the paper are computed from those by `dopt.trajectory`; running `dopt.select`
proposes a design for the same record rather than replaying that selection.

## Licence

Code MIT (`LICENSE`); data CC BY 4.0 (`data/LICENSE`).
