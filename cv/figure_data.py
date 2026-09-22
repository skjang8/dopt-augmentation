"""Reduce the cross-validation results to the inputs the figures read.

Written to `results/cv/figure_data/`:

    fig3_r2_curves.csv     n_total, median, q25, q75, model
    fig3_rmse_curves.csv   n_total, median, q25, q75, model
    fig3_design.csv        n_total, mean_abs_corr, cond_num, rank, logdet_quad
    fig4_predictions.csv   scenario, actual, predicted

and, beside the other metrics files in `results/cv/`:

    metrics_cv_comparison.csv   one row per (model, subset, cv, seed)

The design quantities of the third panel of Figure 3 do not depend on the
cross-validation split, so they are taken unchanged from
`data/figure_data/design_trajectory.csv`.  The attribution panel of Figure 4
reads `data/figure_data/shap_importance.csv` for the same reason: the
attribution model is fit on all the data and no split enters it.

    python -m cv.figure_data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "cv"
STATIC = ROOT / "data" / "figure_data"
OUT_DEF = RESULTS / "figure_data"

MODEL_ORDER = ["Ridge", "Random Forest", "XGBoost", "GPR"]
METRICS = ["R2", "RMSE", "MAE"]
PLOT_SEED = 48          # the seed whose out-of-fold predictions Figure 4 draws
N_OFAT_MAX = 42
ENDPOINTS = {0: "OFAT", 15: "Augmented"}


def _usable(df, col):
    """True when the column exists and carries at least one value.

    The pipeline writes one wide schema for every part, so a part that does not
    use a given counter still emits the column, filled with blanks: the
    D-optimal predictions have an empty `n` and the reference metrics an empty
    `n_added`.  Selecting such a column by name alone fails on the integer
    cast, so presence is not enough."""
    return col in df.columns and df[col].notna().any()


def _n_total(df, base=0):
    """Resolve the experiment-count column under any of its three names."""
    if _usable(df, "n_total"):
        return df["n_total"].astype(int)
    if _usable(df, "n_added"):
        return df["n_added"].astype(int) + base
    if _usable(df, "n"):
        return df["n"].astype(int)
    raise KeyError(f"no populated n_total / n_added / n column in {list(df.columns)}")


def _n_added(df):
    """Resolve the addition-count column; a total is accepted and converted."""
    if _usable(df, "n_added"):
        return df["n_added"].astype(int)
    if _usable(df, "n_total"):
        return df["n_total"].astype(int) - N_OFAT_MAX
    if _usable(df, "n"):
        return df["n"].astype(int) - N_OFAT_MAX
    raise KeyError(f"no populated n_added / n_total / n column in {list(df.columns)}")


def _require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"{path} not found; run `python -m cv.pipeline` first")
    return path


# ── trajectory curves ───────────────────────────────────────────────────────
def _quantile_curves(allm, metric):
    g = (allm.groupby(["n_total", "model"])[metric]
             .agg(median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75))
             .reset_index())
    return g[["n_total", "median", "q25", "q75", "model"]]


def curves():
    seq = pd.read_csv(_require(RESULTS / "metrics_ofatseq.csv"))
    inc = pd.read_csv(_require(RESULTS / "metrics_dopt.csv"))
    seq = seq.assign(n_total=_n_total(seq))
    inc = inc.assign(n_total=_n_total(inc, base=N_OFAT_MAX))
    keep = ["R2", "RMSE", "n_total", "model", "seed"]
    # The one-factor-at-a-time segment is n_total = 6..42 and comes entirely
    # from the sequential part; the D-optimal segment is n_total = 43..57 and
    # comes from the incremental part.  The incremental part at n_added = 0 is
    # the same 42-wafer fit as the sequential part at n = 42, so the join is
    # seamless either way.
    return pd.concat([seq[seq.n_total <= N_OFAT_MAX][keep],
                      inc[inc.n_total > N_OFAT_MAX][keep]], ignore_index=True)


# ── parity points ───────────────────────────────────────────────────────────
def parity():
    p = pd.read_csv(_require(RESULTS / "predictions_dopt.csv"))
    p = p.assign(n_total=_n_total(p, base=N_OFAT_MAX))
    p = p[(p.model == "XGBoost") & (p.seed == PLOT_SEED)]
    rows = []
    for label, n in (("OFAT", N_OFAT_MAX), ("Augmented", 57)):
        q = p[p.n_total == n]
        if q.empty:
            raise ValueError(
                f"predictions_dopt.csv has no XGBoost/seed {PLOT_SEED}/n_total {n} rows")
        rows.append(pd.DataFrame({"scenario": label,
                                  "actual": q["y_true"].to_numpy(float),
                                  "predicted": q["y_pred"].to_numpy(float)}))
    return pd.concat(rows, ignore_index=True)


# ── the two cross-validation schemes side by side ───────────────────────────
def _endpoints(df, cv):
    """Keep the two endpoints, label the subset, drop everything else."""
    df = df.assign(n_added=_n_added(df))
    df = df[df.n_added.isin(ENDPOINTS) & df.model.isin(MODEL_ORDER)].copy()
    df["subset"] = df.n_added.map(ENDPOINTS)
    df["cv"] = cv
    have = [m for m in METRICS if m in df.columns]
    return df[["model", "subset", "cv", "seed"] + have]


def cv_comparison():
    loco = pd.read_csv(_require(RESULTS / "metrics_dopt.csv"))
    ref = pd.read_csv(_require(RESULTS / "metrics_loocv_ref.csv"))
    # The reference file carries both the leave-one-out and the
    # leave-one-condition-out re-run of the same pipelines, so the
    # leave-one-out rows must be selected explicitly; labelling the whole file
    # would average the two splits together.
    if "cv" in ref.columns:
        ref = ref[ref["cv"] == "LOOCV"].copy()
        if ref.empty:
            sys.exit("no cv == 'LOOCV' rows in metrics_loocv_ref.csv")
    df = pd.concat([_endpoints(loco, "LOCO"), _endpoints(ref, "LOOCV")],
                   ignore_index=True)
    df = df.sort_values(["cv", "subset", "model", "seed"]).reset_index(drop=True)
    missing = [(c, s, m) for c in ("LOOCV", "LOCO") for s in ENDPOINTS.values()
               for m in MODEL_ORDER
               if df[(df.cv == c) & (df.subset == s) & (df.model == m)].empty]
    if missing:
        print(f"  WARNING: no rows for {missing}")
    return df


# ── static inputs ───────────────────────────────────────────────────────────
def static_inputs():
    des = pd.read_csv(_require(STATIC / "design_trajectory.csv"))
    if "logdet_quad" not in des.columns:
        sys.exit("design_trajectory.csv has no logdet_quad column")
    shap = pd.read_csv(_require(STATIC / "shap_importance.csv"))
    return des, shap


def main(out: Path, comparison_out: Path) -> None:
    allm = curves()
    des, shap = static_inputs()
    made = {
        "fig3_r2_curves.csv": _quantile_curves(allm, "R2"),
        "fig3_rmse_curves.csv": _quantile_curves(allm, "RMSE"),
        "fig3_design.csv": des,
        "fig4_predictions.csv": parity(),
        "fig4_shap.csv": shap,
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, df in made.items():
        df.to_csv(out / name, index=False)
        print(f"  wrote {name:24s} {len(df):5d} rows")
    print(f"{len(made)} files -> {out}")

    comp = cv_comparison()
    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(comparison_out, index=False)
    print(comp.groupby(["cv", "subset"])["seed"].nunique().to_string())
    print(f"  wrote {comparison_out.name:24s} {len(comp):5d} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEF,
                    help="directory for the figure inputs")
    ap.add_argument("--comparison-out", type=Path,
                    default=RESULTS / "metrics_cv_comparison.csv")
    a = ap.parse_args()
    main(a.out, a.comparison_out)
