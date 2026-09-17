"""Build `results/cv/SUMMARY_TABLES.md` from the metrics and interval files.

    python -m cv.tables
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "cv"
MODELS = ["Ridge", "Random Forest", "XGBoost", "GPR"]

md = pd.read_csv(R / "metrics_dopt.csv")
mo = pd.read_csv(R / "metrics_ofatseq.csv")
ms = pd.read_csv(R / "metrics_sametest.csv")
mf = pd.read_csv(R / "metrics_fivepos.csv")
mr = pd.read_csv(R / "metrics_loocv_ref.csv")
ci = pd.read_csv(R / "bootstrap_ci.csv")
pr = pd.read_csv(R / "bootstrap_paired.csv")


def f2(x):
    if pd.isna(x):
        return "n/a"
    t = f"{x:.2f}"
    return "0.00" if t == "-0.00" else t


def brk(lo, hi):
    return f"[{f2(lo)}, {f2(hi)}]"


def med(df, **sel):
    d = df
    for k, v in sel.items():
        d = d[d[k] == v]
    return d


def ci_row(part, model, subset, cv, step, metric):
    d = ci[(ci.part == part) & (ci.model == model) & (ci.subset.astype(str) == str(subset)) &
           (ci.cv == cv) & (ci.step.astype(str) == str(step)) & (ci.metric == metric)]
    if len(d) != 1:
        raise SystemExit(f"ci lookup {part}/{model}/{subset}/{cv}/{step}/{metric} -> {len(d)} rows")
    return d.iloc[0]


def stats(d, col):
    v = d[col].to_numpy(float)
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


out = []
W = out.append

W("# Summary tables — LOCO revision (10 seeds, condition-level bootstrap)\n")
W("Every value is the median over the 10 seeds (42-51) of the per-seed point estimate. ")
W("Bracketed intervals are 2.5/97.5 percentiles of the condition-level bootstrap, ")
W("B = 2000 resamples per seed pooled over the 10 seeds (20 000 draws). ")
W("Conditions, not wafers, are the resampling unit; every model and every training ")
W("mode within a domain sees the identical condition draws, so all differences are paired. ")
W("R² is dimensionless; RMSE and MAE are in nm.\n")

# ---------------------------------------------------------------- Table 1
W("## Table 1 — LOCO before and after the D-optimal augmentation\n")
W("`before` = 42 OFAT wafers (n_added = 0), `after` = 42 OFAT + 15 D-optimal wafers ")
W("(n_added = 15), both under leave-one-condition-out CV on the 4-position mean thickness. ")
W("IQR is the seed-to-seed interquartile range of the point estimate.\n")
W("| Model | Metric | Before (n_added = 0) | 95 % CI | seed IQR | After (n_added = 15) | 95 % CI | seed IQR |")
W("|---|---|---|---|---|---|---|---|")
for m in MODELS:
    for met in ["R2", "RMSE", "MAE"]:
        cells = []
        for step in (0, 15):
            d = med(md, model=m, n_added=step)
            p, q1, q3 = stats(d, met)
            c = ci_row("dopt", m, "all", "LOCO", step, met)
            cells += [f2(p), brk(c.lo, c.hi), f"{f2(q1)}–{f2(q3)}"]
        lbl = {"R2": "R²", "RMSE": "RMSE (nm)", "MAE": "MAE (nm)"}[met]
        W(f"| {m} | {lbl} | " + " | ".join(cells) + " |")
W("")

# ---------------------------------------------------------------- Table 2
W("## Table 2 — LOOCV versus LOCO on identical data and pipelines\n")
W("Same seeds, same nested-CV pipelines and grids; only the outer split differs. ")
W("OFAT = 42 wafers / 24 conditions, Augmented = 57 wafers / 39 conditions.\n")
W("| Model | Subset | CV | median R² | median RMSE (nm) | ΔR² (LOCO − LOOCV) |")
W("|---|---|---|---|---|---|")
for m in MODELS:
    for sub in ["OFAT", "Augmented"]:
        r2s = {}
        for cvn in ["LOOCV", "LOCO"]:
            d = med(mr, model=m, subset=sub, cv=cvn)
            r2s[cvn] = np.median(d.R2.to_numpy(float))
            W(f"| {m} | {sub} | {cvn} | {f2(r2s[cvn])} | "
              f"{f2(np.median(d.RMSE.to_numpy(float)))} | "
              f"{'' if cvn == 'LOOCV' else f2(r2s['LOCO'] - r2s['LOOCV'])} |")
W("")

# ---------------------------------------------------------------- Table 3
DOMS = [("OFAT rows", "OFAT", 42), ("DOE rows", "DOE", 15), ("all 57", "all", 57)]
W("## Table 3 — Same-test-set A/B comparison\n")
W("For every one of the 39 conditions, model A trains on the OFAT wafers outside that ")
W("condition and model B on all 57 wafers outside it; both predict the same held-out rows. ")
W("Metrics are therefore computed on identical test rows.\n")
W("### 3a. Per-mode performance\n")
W("| Model | Test domain | Mode | median R² | median RMSE (nm) | median MAE (nm) |")
W("|---|---|---|---|---|---|")
for m in MODELS:
    for dom, _, _ in DOMS:
        for mode in ["A", "B"]:
            d = med(ms, model=m, test_domain=dom, variant=mode)
            W(f"| {m} | {dom} | {mode} | " + " | ".join(
                f2(np.median(d[c].to_numpy(float))) for c in ["R2", "RMSE", "MAE"]) + " |")
W("")
W("### 3b. Paired differences (B − A) on identical resampled conditions\n")
W("`favour B` is the fraction of the 20 000 paired bootstrap draws in which B beats A ")
W("(higher R², lower RMSE / MAE).\n")
W("| Model | Test domain | ΔR² | 95 % CI | favour B | ΔRMSE (nm) | 95 % CI | favour B | ΔMAE (nm) | 95 % CI | favour B |")
W("|---|---|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    for dom, key, _ in DOMS:
        cells = []
        for met in ["R2", "RMSE", "MAE"]:
            d = pr[(pr.model == m) & (pr.test_domain == key) & (pr.metric == met)]
            if len(d) != 1:
                raise SystemExit(f"paired lookup {m}/{key}/{met} -> {len(d)} rows")
            d = d.iloc[0]
            cells += [f2(d.delta_point), brk(d.lo, d.hi), f"{d.frac_favor_B:.2f}"]
        W(f"| {m} | {dom} | " + " | ".join(cells) + " |")
W("")

# ---------------------------------------------------------------- Table 4
W("## Table 4 — Five-position target sensitivity\n")
W("Same LOCO folds and pipelines, target swapped from the 4-position mean thickness to ")
W("the 5-position mean. The 4-position columns are the LOCO rows of `metrics_loocv_ref.csv`.\n")
W("| Model | Subset | 5-pos R² | 95 % CI | 4-pos R² | 5-pos RMSE (nm) | 95 % CI | 4-pos RMSE (nm) |")
W("|---|---|---|---|---|---|---|---|")
for m in MODELS:
    for sub in ["OFAT", "Augmented"]:
        d5 = med(mf, model=m, subset=sub)
        d4 = med(mr, model=m, subset=sub, cv="LOCO")
        tgt = d5.target.iloc[0]
        row = [m, sub]
        for met in ["R2", "RMSE"]:
            c = ci_row("fivepos", m, sub, "LOCO", tgt, met)
            row += [f2(np.median(d5[met].to_numpy(float))), brk(c.lo, c.hi),
                    f2(np.median(d4[met].to_numpy(float)))]
        W("| " + " | ".join(row) + " |")
W("")

# ---------------------------------------------------------------- Table 5
STEPS = [0, 1, 3, 5, 8, 10, 15]
W("## Table 5 — Incremental D-optimal trajectory\n")
W("Median LOCO R² as the 15 D-optimal wafers are appended in D-optimal selection order. ")
W("Bootstrap CIs are given at n_added = 0, 5 and 15.\n")
W("| Model | " + " | ".join(f"n = {s}" for s in STEPS) + " | closed fraction |")
W("|---" * (len(STEPS) + 2) + "|")
for m in MODELS:
    vals, cells = {}, []
    for s in STEPS:
        v = np.median(med(md, model=m, n_added=s).R2.to_numpy(float))
        vals[s] = v
        if s in (0, 5, 15):
            c = ci_row("dopt", m, "all", "LOCO", s, "R2")
            cells.append(f"{f2(v)} {brk(c.lo, c.hi)}")
        else:
            cells.append(f2(v))
    closed = (vals[15] - vals[0]) / (1.0 - vals[0])
    W(f"| {m} | " + " | ".join(cells) + f" | {closed:.2f} |")
W("")
W("`closed fraction` = (R²(15) − R²(0)) / (1 − R²(0)): the share of the residual ")
W("unexplained variance at n_added = 0 that the 15 D-optimal wafers remove.\n")

# ---------------------------------------------------------------- Table 6
NS = [12, 20, 30, 42]
W("## Table 6 — OFAT sequential baseline\n")
W("Median LOCO R² as OFAT wafers accumulate in file order, with the number of distinct ")
W("process conditions available at each n.\n")
W("| Model | " + " | ".join(f"n = {n}" for n in NS) + " |")
W("|---" * (len(NS) + 1) + "|")
for m in MODELS:
    W(f"| {m} | " + " | ".join(
        f2(np.median(med(mo, model=m, n=n).R2.to_numpy(float))) for n in NS) + " |")
W("| **unique conditions** | " + " | ".join(
    str(int(med(mo, n=n).n_groups.unique()[0])) for n in NS) + " |")
W("")

# ---------------------------------------------------------- observations
n_excl = int(((pr.lo > 0) | (pr.hi < 0)).sum())
W("## Observations\n")
W(
    f"Of the 36 paired same-test-set differences, {n_excl} have a 95 % interval that "
    "excludes zero: all nine GPR cells, all three Ridge cells on the full 57 rows, the "
    "XGBoost cells on the DOE rows and on the full 57 rows, and the Random Forest MAE on "
    "the DOE rows. The Ridge differences on the OFAT rows and on the DOE rows taken "
    "separately straddle zero even though 94 % and 95 % of draws favour B, which is what a "
    "24-condition and a 15-condition resample buys; only the pooled 39-condition domain "
    "separates. Random Forest is the model the augmentation moves least, ΔR² = 0.02 "
    "[-0.02, 0.12] on all 57 rows, and it is the only model whose R² interval on the full "
    "test set overlaps zero improvement in every domain. GPR is the opposite extreme: its "
    "ΔR² upper bounds run past 1.00 (1.59 on the OFAT rows, 1.31 on the DOE rows) because "
    "mode A, trained on OFAT wafers alone, produces deeply negative R² on many resamples, "
    "so the difference is not bounded by 1. Interval widths track condition count far more "
    "than wafer count. The widest interval anywhere is GPR on the 24-condition OFAT subset "
    "under LOCO, R² = 0.23 spanning [-1.01, 0.69], a width of 1.70; the same model on the "
    "39-condition augmented subset narrows to a width of 0.25, and every augmented R² "
    "interval in Table 1 sits between 0.19 and 0.29 wide. Switching to the five-position "
    "target (Table 4) halves that GPR OFAT width to 0.79 and lifts the median from 0.23 to "
    "0.44, the only place where the target definition changes a conclusion; every other "
    "five-position value is within 0.04 R² of its four-position counterpart. The "
    "incremental trajectory is not monotonic for any model: all four dip at n_added = 8 and "
    "again at n_added = 12, and GPR falls furthest, from 0.76 at n_added = 5 to 0.70 at "
    "n_added = 8 before recovering to 0.77 at n_added = 15. GPR also closes the largest "
    "share of residual variance (0.70) purely because it starts lowest, while Random Forest "
    "closes the least (0.41) from the highest starting point. The OFAT sequential baseline "
    "(Table 6) is far more erratic than the D-optimal trajectory: GPR is negative at every "
    "n from 6 to 15 and again from 21 to 32, and Ridge peaks at 0.34 at n = 13, falls to "
    "0.02 at n = 17, and does not beat that early peak until n = 35. Adding 30 OFAT wafers "
    "raises the distinct-condition count only from 12 to 24, which is the plainest reading "
    "of why the curve stalls. Finally, LOCO is at or below LOOCV everywhere (Table 2) but "
    "by at most 0.04 R², so the optimism of leaving out one wafer at a time is real but "
    "small at this sample size.\n")

path = R / "SUMMARY_TABLES.md"
with open(path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("wrote", path, len(out), "lines")
