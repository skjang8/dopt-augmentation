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

The first reproduces the design reported in the paper. The second is a different
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

## Data

    data/ofat_clean.csv          42 OFAT conditions with the measured thickness
    data/doe_clean.csv           15 D-optimal conditions with the measured thickness
    data/combined_clean.csv      the 57 together
    data/doe_doptimal_order.csv  the order in which the 15 were selected
    data/yoon2023_screening.csv  17 screening conditions, Pt rotary ALD

The HfO2 files carry the five process parameters, the film thickness at four
positions across the wafer, and their mean, which is the prediction target. The
Pt record carries conditions only, which is all the routine reads; it is from
our earlier work, Yoon et al., Langmuir 39 (2023) 4984.

`dopt.select` scores the full grid and is deterministic, so running it on the
first config proposes a design of the same quality but not necessarily the
identical 15 conditions. The 15 in `data/` are those actually performed, and the
design quantities reported in the paper are computed from them by
`dopt.trajectory`.

## Licence

Code MIT (`LICENSE`); data CC BY 4.0 (`data/LICENSE`).
