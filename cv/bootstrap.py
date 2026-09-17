"""Condition-level bootstrap confidence intervals, pooled across seeds.

The resampling unit is the process condition, not the wafer.  For each
(part, model, subset, step, seed) the distinct conditions of the saved
out-of-fold predictions are drawn with replacement, every wafer of a drawn
condition is kept, and R2, RMSE and MAE are recomputed on the resampled set.
The draws of all seeds are pooled and reported as the 2.5, 50 and 97.5
percentiles; `point` is the across-seed median of the per-seed point estimate.

Draws are cached per (condition set, seed), so every model and every training
mode within a domain sees the identical resample.  That makes the B minus A
differences of the same-test-set comparison and the model-to-model comparisons
paired.

Reads the prediction files written by `cv.pipeline` and writes
`bootstrap_ci.csv` and `bootstrap_paired.csv` beside them.

    python -m cv.bootstrap
    python -m cv.bootstrap --B 2000 --results-dir ... --out-dir ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "cv"

# Columns every prediction file must carry.  The pipeline writes the training
# set size as `n` (not `n_total`), so that is what is required here.
LONG_COLS = ["part", "model", "seed", "n_added", "n", "row_id",
             "group_id", "source", "y_true", "y_pred"]


# --------------------------------------------------------------- metrics
def _r2(y, p):
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - np.sum((y - p) ** 2) / ss_tot


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


METRICS = {"R2": _r2, "RMSE": _rmse, "MAE": _mae}
HIGHER_IS_BETTER = {"R2": True, "RMSE": False, "MAE": False}


# --------------------------------------------------------- resample cache
class Resampler:
    """Condition draws shared by every model and training mode in a domain."""

    def __init__(self, B: int):
        self.B = B
        self._cache: dict[tuple, np.ndarray] = {}

    def draws(self, groups: np.ndarray, seed: int) -> np.ndarray:
        """(B, n_groups) array of positional indices into `groups`."""
        key = (tuple(groups), int(seed))
        if key not in self._cache:
            rng = np.random.default_rng(int(seed))
            self._cache[key] = rng.integers(0, len(groups), size=(self.B, len(groups)))
        return self._cache[key]


def boot_draws(df: pd.DataFrame, rs: Resampler, seed: int) -> dict[str, np.ndarray]:
    """Bootstrap metric values for one (model, subset, step, seed) slice."""
    groups = np.array(sorted(df.group_id.unique()))
    pos = {g: np.flatnonzero((df.group_id == g).to_numpy()) for g in groups}
    y = df.y_true.to_numpy(float)
    p = df.y_pred.to_numpy(float)
    idx = rs.draws(groups, seed)
    out = {m: np.empty(len(idx)) for m in METRICS}
    for b, row in enumerate(idx):
        rows = np.concatenate([pos[groups[j]] for j in row])
        yy, pp = y[rows], p[rows]
        for m, fn in METRICS.items():
            out[m][b] = fn(yy, pp)
    return out


def point_metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df.y_true.to_numpy(float)
    p = df.y_pred.to_numpy(float)
    return {m: fn(y, p) for m, fn in METRICS.items()}


def summarize(part, model, subset, cv, step, slices, rs) -> list[dict]:
    """slices: {seed: dataframe}. Pools bootstrap draws over seeds."""
    pooled = {m: [] for m in METRICS}
    per_seed_point = {m: [] for m in METRICS}
    n_groups = n_rows = 0
    for seed, sub in sorted(slices.items()):
        d = boot_draws(sub, rs, seed)
        for m in METRICS:
            pooled[m].append(d[m])
        pm = point_metrics(sub)
        for m in METRICS:
            per_seed_point[m].append(pm[m])
        n_groups, n_rows = sub.group_id.nunique(), len(sub)
    rows = []
    for m in METRICS:
        v = np.concatenate(pooled[m])
        v = v[np.isfinite(v)]
        lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
        rows.append(dict(
            part=part, model=model, subset=subset, cv=cv, step=step, metric=m,
            point=float(np.median(per_seed_point[m])),
            lo=float(lo), hi=float(hi), median_boot=float(med),
            n_groups=int(n_groups), n_rows=int(n_rows), B_total=int(len(v)),
            n_seeds=len(slices),
        ))
    return rows


# ------------------------------------------------------------------ parts
def load_long(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    missing = [c for c in LONG_COLS if c not in df.columns]
    if missing:
        sys.exit(f"{os.path.basename(path)} is missing columns: {missing}")
    return df


def run_dopt(df, rs, cv="LOCO") -> list[dict]:
    """An interval at every n_added, so the trajectory figure can carry bands."""
    rows = []
    for model, by_model in df.groupby("model", sort=True):
        for n_added, by_step in by_model.groupby("n_added", sort=True):
            slices = {s: g for s, g in by_step.groupby("seed")}
            rows += summarize("dopt", model, "all", cv, int(n_added), slices, rs)
    return rows


def run_sametest(df, rs) -> tuple[list[dict], list[dict]]:
    """Per-training-mode intervals on three test domains, plus paired B - A."""
    domains = {
        "OFAT": lambda d: d[d.source.astype(str).str.upper().str.startswith("OFAT")],
        "DOE": lambda d: d[~d.source.astype(str).str.upper().str.startswith("OFAT")],
        "all": lambda d: d,
    }
    ci_rows, pair_rows = [], []
    for model, by_model in df.groupby("model", sort=True):
        for dom, sel in domains.items():
            dd = sel(by_model)
            if dd.empty:
                continue
            for mode, by_mode in dd.groupby("train_mode", sort=True):
                slices = {s: g for s, g in by_mode.groupby("seed")}
                ci_rows += summarize("sametest", model, f"{mode}@{dom}",
                                     "LOCO", "", slices, rs)
            if set(dd.train_mode.unique()) != {"A", "B"}:
                continue
            pair_rows += paired(model, dom, dd, rs)
    return ci_rows, pair_rows


def paired(model, dom, dd, rs) -> list[dict]:
    """B - A on identical resampled conditions, pooled over seeds."""
    pooled = {m: [] for m in METRICS}
    point = {m: [] for m in METRICS}
    for seed, g in dd.groupby("seed"):
        a = g[g.train_mode == "A"].set_index("row_id").sort_index()
        b = g[g.train_mode == "B"].set_index("row_id").sort_index()
        if not a.index.equals(b.index):
            sys.exit(f"sametest row_id mismatch between A and B ({model}, {dom}, seed {seed})")
        groups = np.array(sorted(a.group_id.unique()))
        pos = {gr: np.flatnonzero((a.group_id == gr).to_numpy()) for gr in groups}
        y = a.y_true.to_numpy(float)
        pa, pb = a.y_pred.to_numpy(float), b.y_pred.to_numpy(float)
        for m, fn in METRICS.items():
            point[m].append(fn(y, pb) - fn(y, pa))
        for row in rs.draws(groups, seed):
            rows = np.concatenate([pos[groups[j]] for j in row])
            yy = y[rows]
            for m, fn in METRICS.items():
                pooled[m].append(fn(yy, pb[rows]) - fn(yy, pa[rows]))
    out = []
    for m in METRICS:
        v = np.asarray(pooled[m])
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        favor = np.mean(v > 0) if HIGHER_IS_BETTER[m] else np.mean(v < 0)
        out.append(dict(part="sametest", model=model, test_domain=dom, metric=m,
                        delta_point=float(np.median(point[m])),
                        lo=float(lo), hi=float(hi), frac_favor_B=float(favor),
                        n_groups=int(dd.group_id.nunique()),
                        B_total=int(len(v))))
    return out


def run_fivepos(df, rs) -> list[dict]:
    rows = []
    has_target = "target" in df.columns
    for model, by_model in df.groupby("model", sort=True):
        for subset, by_sub in by_model.groupby("subset", sort=True):
            targets = by_sub.groupby("target", sort=True) if has_target else [("", by_sub)]
            for tgt, by_t in targets:
                slices = {s: g for s, g in by_t.groupby("seed")}
                rows += summarize("fivepos", model, str(subset), "LOCO",
                                  str(tgt), slices, rs)
    return rows


def run_loocv_ref(df, rs) -> list[dict]:
    """LOOCV and LOCO side by side, both resampled by condition so the two
    intervals are like for like.  This file carries both splits for both
    subsets, so the slices must be keyed by (subset, cv); keying by subset
    alone would pool the two sets of out-of-fold predictions into one metric."""
    rows = []
    sub_col = "subset" if "subset" in df.columns else "n"
    cv_col = "cv" if "cv" in df.columns else None
    for model, by_model in df.groupby("model", sort=True):
        keys = [sub_col] + ([cv_col] if cv_col else [])
        for k, g in by_model.groupby(keys, sort=True):
            k = k if isinstance(k, tuple) else (k,)
            subset = str(k[0])
            cv = str(k[1]) if cv_col else "LOOCV"
            slices = {s: gg for s, gg in g.groupby("seed")}
            rows += summarize("loocv_ref", model, subset, cv, "", slices, rs)
    return rows


# ------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=2000, help="bootstrap draws per seed")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    rs = Resampler(args.B)

    ci_rows, pair_rows = [], []

    dopt = load_long(os.path.join(args.results_dir, "predictions_dopt.csv"))
    if dopt is not None:
        ci_rows += run_dopt(dopt, rs)
    else:
        print("skip: predictions_dopt.csv not found")

    st = load_long(os.path.join(args.results_dir, "predictions_sametest.csv"))
    if st is not None:
        if "train_mode" not in st.columns:
            sys.exit("predictions_sametest.csv needs a train_mode column (A/B)")
        c, pr = run_sametest(st, rs)
        ci_rows += c
        pair_rows += pr
    else:
        print("skip: predictions_sametest.csv not found")

    fp = load_long(os.path.join(args.results_dir, "predictions_fivepos.csv"))
    if fp is not None:
        if "subset" not in fp.columns:
            sys.exit("predictions_fivepos.csv needs a subset column")
        ci_rows += run_fivepos(fp, rs)
    else:
        print("skip: predictions_fivepos.csv not found")

    ref = load_long(os.path.join(args.results_dir, "predictions_loocv_ref.csv"))
    if ref is not None:
        ci_rows += run_loocv_ref(ref, rs)

    if not ci_rows:
        sys.exit("no inputs found; nothing written")

    ci = pd.DataFrame(ci_rows)[
        ["part", "model", "subset", "cv", "step", "metric", "point", "lo", "hi",
         "median_boot", "n_groups", "n_rows", "B_total", "n_seeds"]]
    ci_path = os.path.join(out_dir, "bootstrap_ci.csv")
    ci.to_csv(ci_path, index=False)
    print(f"\nwrote {ci_path}  ({len(ci)} rows)")
    show = ci[ci.metric.isin(["R2", "RMSE"])]
    print(show[["part", "model", "subset", "step", "metric", "point", "lo", "hi",
                "n_groups", "n_rows", "B_total"]].round(4).to_string(index=False))

    if pair_rows:
        pc = pd.DataFrame(pair_rows)[
            ["part", "model", "test_domain", "metric", "delta_point", "lo", "hi",
             "frac_favor_B", "n_groups", "B_total"]]
        pp = os.path.join(out_dir, "bootstrap_paired.csv")
        pc.to_csv(pp, index=False)
        print(f"\nwrote {pp}  ({len(pc)} rows)")
        print(pc[pc.metric.isin(["R2", "RMSE"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
