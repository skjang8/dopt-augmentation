"""Draw the cross-validation figures into `figures/`.

    python -m cv.figures                 all four
    python -m cv.figures fig3 figR4      a selection

fig3    Figure 3: the leave-one-condition-out R2 and RMSE trajectories as the
        42 one-factor-at-a-time wafers accumulate and the 15 D-optimal wafers
        are appended, with the design quantities underneath.
fig4    Figure 4: measured against predicted thickness before and after the
        augmentation, and the mean absolute attribution per parameter.
figR3   leave-one-out against leave-one-condition-out cross-validation, before
        and after the augmentation, with condition-level intervals.
figR4   the same-test-set comparison: for every held-out condition, one model
        trained on the remaining one-factor-at-a-time wafers and one trained on
        those plus the D-optimal wafers, both scoring the same wafers.

The figures read the results in `results/cv/` and write 300 dpi PNGs.  Run
`python -m cv.pipeline`, `python -m cv.bootstrap` and `python -m cv.figure_data`
first.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "cv"
FIGDATA = RESULTS / "figure_data"
OUT = ROOT / "figures"

MODEL_ORDER = ["Ridge", "Random Forest", "XGBoost", "GPR"]
MODEL_COLORS = {"Ridge": "#4C72B0", "Random Forest": "#55A868",
                "XGBoost": "#C44E52", "GPR": "#8172B2"}
COLOR_OFAT, COLOR_COMBINED, COLOR_DOE = "#4C72B0", "#55A868", "#C44E52"
FEATURES = ["Plasma_Power_W", "Cycles", "Chuck_T_C", "O2_Flow_sccm", "Oxidant_Dose_s"]
RETAINED = ["T1", "T3", "T4", "T5"]
PLOT_SEED = 48
N_OFAT, N_TOTAL = 42, 57
N_COND_OFAT, N_COND_AUG = 24, 39

# Figures 1 and 2 of the manuscript are drawn in Arial, so these use Arial too.
# On Linux, copy the four Arial faces from a licensed installation into
# ~/.local/share/fonts/, run `fc-cache -f` and clear the matplotlib font cache.
FONT_STACK = ["Arial", "Arimo", "Liberation Sans", "Helvetica",
              "Nimbus Sans", "DejaVu Sans"]


def _style(label_size=12):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 10, "axes.labelsize": label_size,
        "axes.labelweight": "bold", "figure.dpi": 200,
    })


def _require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"{path} not found; run the pipeline, bootstrap and figure_data first")
    return path


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {name}.png")


# ── Figure 3 ────────────────────────────────────────────────────────────────
SHOW_COND_AXIS = False        # True adds the fold-count axis on panel (a)
P_TERMS, R2_FLOOR = 21, -0.3
C_CORR, C_COND, C_DET = "#E07B39", "#4C72B0", "#3F8F5B"
LEGEND_SIZE = 10.5


def unique_conditions(n_total):
    """Distinct five-parameter settings among the first n_total experiments,
    which is the number of leave-one-condition-out folds."""
    d = pd.read_csv(DATA / "combined_clean.csv")
    ofat = d[d.source == "OFAT"].reset_index(drop=True)
    g = pd.factorize(pd.Series([tuple(r) for r in ofat[FEATURES].values]))[0]
    n_ofat_cond = len(set(g))
    return [len(set(g[:n])) if n <= N_OFAT else n_ofat_cond + (n - N_OFAT) for n in n_total]


def fig3():
    _style(12)
    r2 = pd.read_csv(_require(FIGDATA / "fig3_r2_curves.csv"))
    rmse = pd.read_csv(_require(FIGDATA / "fig3_rmse_curves.csv"))
    des = pd.read_csv(_require(FIGDATA / "fig3_design.csv"))
    if "logdet_quad" not in des:
        sys.exit("fig3_design.csv has no logdet_quad column; run `python -m cv.figure_data`")

    fig, (ax_r2, ax_rm, ax_d) = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True,
                                             constrained_layout=True)
    for ax in (ax_r2, ax_rm, ax_d):
        ax.axvspan(N_OFAT, N_TOTAL, color="#f0e4d0", alpha=0.5, zorder=0)
        ax.axvline(N_OFAT, color="gray", lw=0.6, ls="--", alpha=0.6)

    for m in MODEL_ORDER:
        c = MODEL_COLORS[m]
        for ax, src in ((ax_r2, r2), (ax_rm, rmse)):
            d = src[src.model == m]
            ax.fill_between(d.n_total, d.q25, d.q75, color=c, alpha=0.18, lw=0, zorder=1)
            ax.plot(d.n_total, d["median"], "o-", color=c, ms=3.0, lw=1.3,
                    label=m if ax is ax_r2 else None, zorder=2)

    ax_r2.axhline(0.7, color="gray", lw=0.5, ls=":", alpha=0.6)
    ax_r2.axhline(0.0, color="gray", lw=0.4, alpha=0.4)
    ax_r2.set_ylabel("R²")
    ax_r2.set_ylim(max(R2_FLOOR, np.floor(r2.q25.min() * 20) / 20),
                   np.ceil(r2.q75.max() * 20) / 20 + 0.05)
    ax_r2.legend(loc="upper left", fontsize=LEGEND_SIZE, frameon=False, ncol=2)
    ax_rm.set_ylabel("RMSE (nm)")
    for ax in (ax_r2, ax_rm):
        ax.grid(True, axis="y", lw=0.4, alpha=0.4)

    m1 = des.mean_abs_corr.notna()
    l1 = ax_d.plot(des.n_total[m1], des.mean_abs_corr[m1], "o-", color=C_CORR, ms=3.0,
                   lw=1.3, label="Mean |r|")[0]
    ax_d.set_ylabel("Mean |correlation|", color=C_CORR)
    ax_d.tick_params(axis="y", labelcolor=C_CORR)
    ax_d.grid(True, axis="y", lw=0.4, alpha=0.3)

    ax2 = ax_d.twinx()
    m2 = des.cond_num.notna()
    l2 = ax2.plot(des.n_total[m2], des.cond_num[m2], "o-", color=C_COND, ms=3.0, lw=1.3,
                  label="Condition number κ")[0]
    ax2.set_yscale("log")
    ax2.set_ylabel("Condition number κ", color=C_COND)
    ax2.tick_params(axis="y", labelcolor=C_COND)
    ax2.set_yticks([6, 10, 20, 30, 40])
    ax2.set_yticks([], minor=True)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    ax3 = ax_d.twinx()
    ax3.spines["right"].set_position(("axes", 1.078))
    det = des["logdet_quad"]
    defined = det.notna()
    l3 = ax3.plot(des.n_total[defined], det[defined], "o-", color=C_DET, ms=3.0, lw=1.3,
                  label="$\\log_{10}$ det(X′X)")[0]
    # Arial has no unicode subscript glyphs, so the subscript comes from
    # mathtext, but the prime is Arial's own U+2032: the mathtext prime sits
    # too high and detached.
    ax3.set_ylabel("$\\mathbf{log_{10}}$ det(X′X)", color=C_DET, labelpad=2)
    ax3.tick_params(axis="y", labelcolor=C_DET, labelsize=9)
    ax3.set_ylim(float(det[defined].min()) - 1.0, float(det[defined].max()) + 1.0)

    ax_d.legend(handles=[l1, l2, l3], loc="lower left", fontsize=LEGEND_SIZE, frameon=False)

    ax_d.set_xlim(6, N_TOTAL)
    ax_d.set_xlabel("Total experiments (n_total)")
    for ax, s in zip((ax_r2, ax_rm, ax_d), "abc"):
        ax.text(-0.08, 1.0, s, transform=ax.transAxes, fontsize=18,
                fontweight="bold", ha="left", va="bottom")
    top = ax_r2.get_ylim()[1]
    ax_r2.text(N_OFAT - 0.5, top - 0.04, "OFAT sequential", fontsize=9, color="dimgray",
               ha="right", va="top")
    ax_r2.text(N_OFAT + 0.5, top - 0.04, "D-optimal", fontsize=9, color="dimgray",
               ha="left", va="top")

    if SHOW_COND_AXIS:
        ticks = [6, 12, 18, 24, 30, 36, 42, 45, 51, 57]
        ax_top = ax_r2.secondary_xaxis("top")
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels([str(v) for v in unique_conditions(ticks)], fontsize=8)
        ax_top.set_xlabel("Unique process conditions (LOCO folds)", fontsize=9,
                          fontweight="normal", labelpad=3)

    fig.align_ylabels([ax_r2, ax_rm, ax_d])
    _save(fig, "Figure3_LOCO")


# ── Figure 4 ────────────────────────────────────────────────────────────────
def _parity_metrics(y, yp):
    resid = y - yp
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    r2 = 1 - float(np.sum(resid ** 2)) / float(np.sum((y - y.mean()) ** 2))
    return r2, rmse, mae


def fig4():
    _style(12)
    preds = pd.read_csv(_require(FIGDATA / "fig4_predictions.csv"))
    shap_df = pd.read_csv(_require(FIGDATA / "fig4_shap.csv"))

    ofat = preds[preds["scenario"] == "OFAT"]
    aug = preds[preds["scenario"] == "Augmented"]
    y_ofat = ofat["actual"].to_numpy()
    yp_before = ofat["predicted"].to_numpy()
    y_all = aug["actual"].to_numpy()
    yp_after = aug["predicted"].to_numpy()

    lo = float(min(y_ofat.min(), y_all.min(), yp_before.min(), yp_after.min())) - 1
    hi = float(max(y_ofat.max(), y_all.max(), yp_before.max(), yp_after.max())) + 1

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(15, 5),
        gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.3))

    for ax, y_true, y_pred, color, title in [
        (ax_a, y_ofat, yp_before, COLOR_OFAT,
         f"OFAT-only (n = {len(y_ofat)}, {N_COND_OFAT} unique conditions)"),
        (ax_b, y_all, yp_after, COLOR_COMBINED,
         f"Augmented (n = {len(y_all)}, {N_COND_AUG} unique conditions)"),
    ]:
        r2, rmse, mae = _parity_metrics(y_true, y_pred)
        ax.scatter(y_true, y_pred, c=color, s=32, alpha=0.75,
                   edgecolors="black", linewidths=0.35)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        ax.set_xlabel("Actual thickness (nm)")
        ax.set_ylabel("Predicted thickness (nm)")
        ax.set_title(title, fontsize=9, fontweight="normal", pad=6)
        ax.grid(True, lw=0.4, alpha=0.3)
        ax.text(0.05, 0.95,
                f"R² = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor="lightgray"))

    features = shap_df["feature"].to_numpy()
    order = np.argsort(shap_df["augmented"].to_numpy())[::-1]
    x = np.arange(len(features))
    w = 0.38
    labels = features[order]
    vb = shap_df["ofat"].to_numpy()[order]
    va = shap_df["augmented"].to_numpy()[order]
    bars_b = ax_c.bar(x - w / 2, vb, w, color=COLOR_OFAT, alpha=0.85,
                      edgecolor="black", linewidth=0.4, label=f"OFAT (n = {len(y_ofat)})")
    bars_a = ax_c.bar(x + w / 2, va, w, color=COLOR_DOE, alpha=0.85,
                      edgecolor="black", linewidth=0.4,
                      label=f"Augmented (n = {len(y_all)})")
    for bars, vals in ((bars_b, vb), (bars_a, va)):
        for bar, v in zip(bars, vals):
            ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                      f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels)
    ax_c.set_ylabel("Mean |SHAP value| (nm)")
    ax_c.grid(True, axis="y", lw=0.4, alpha=0.3)
    ax_c.set_ylim(0, max(vb.max(), va.max()) * 1.2)
    ax_c.set_box_aspect(1)
    ax_c.legend(loc="upper right", fontsize=8, frameon=False)

    for ax, letter in zip((ax_a, ax_b, ax_c), ("a", "b", "c")):
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes,
                fontsize=18, fontweight="bold", ha="left", va="bottom")

    _save(fig, "Figure4_LOCO")


# ── Figure R3 ───────────────────────────────────────────────────────────────
PURE_ERROR_SD = 3.45      # nm, pooled over repeated conditions
BARS_R3 = [("OFAT", "LOOCV"), ("OFAT", "LOCO"),
           ("Augmented", "LOOCV"), ("Augmented", "LOCO")]
BAR_STYLE_R3 = {("OFAT", "LOOCV"):      dict(alpha=0.35, hatch=""),
                ("OFAT", "LOCO"):       dict(alpha=0.35, hatch="///"),
                ("Augmented", "LOOCV"): dict(alpha=0.95, hatch=""),
                ("Augmented", "LOCO"):  dict(alpha=0.95, hatch="///")}
# The interval file covers every part of the pipeline and the subset labels are
# reused across parts: "OFAT" and "Augmented" appear under the reference part
# (four-position target, both splits) and under the five-position part
# (five-position target, LOCO only).  Selecting on the subset alone would pull
# five-position intervals onto these bars, so the part is pinned here.
BOOT_PART_R3 = "loocv_ref"


def figR3():
    _style(12)
    src = _require(RESULTS / "metrics_cv_comparison.csv")
    df = pd.read_csv(src)
    if "target" in df.columns:
        df = df[df["target"] == "Thickness_mean"]
    val = df.groupby(["subset", "model", "cv"])[["R2", "RMSE"]].median().reset_index()

    boot_path = RESULTS / "bootstrap_ci.csv"
    ci = pd.read_csv(boot_path) if boot_path.exists() else None
    if ci is not None:
        if "part" in ci.columns:
            ci = ci[ci.part == BOOT_PART_R3]
        if ci.empty:
            print(f"  WARNING: no part == '{BOOT_PART_R3}' rows in {boot_path.name}; "
                  "bars drawn without error bars")
            ci = None

    def bounds(model, subset, cv, metric, v):
        """The condition-level interval for one bar, or None.

        The interval file also carries `point`, its own copy of the statistic
        the interval was built around.  It must agree with the bar height,
        which comes from the per-seed medians of the metrics file; a
        disagreement means the two were computed from different runs, so it is
        reported rather than silently plotted."""
        if ci is None:
            return None
        r = ci[(ci.model == model) & (ci.subset == subset)
               & (ci.cv == cv) & (ci.metric == metric)]
        if r.empty:
            return None
        if len(r) > 1:
            sys.exit(f"{boot_path.name}: {len(r)} rows for "
                     f"{BOOT_PART_R3}/{model}/{subset}/{cv}/{metric}; expected one")
        if "point" in r.columns and abs(float(r.point.iloc[0]) - v) > 5e-3:
            print(f"  WARNING: {model}/{subset}/{cv}/{metric}: bar {v:.4f} but "
                  f"interval centre {float(r.point.iloc[0]):.4f}")
        return float(r.lo.iloc[0]), float(r.hi.iloc[0])

    fig, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    w = 0.20
    # Extremes actually drawn on each axis, so no interval is cut off.
    span = {"R2": [0.0, val.R2.max()], "RMSE": [0.0, val.RMSE.max()]}

    for ax, metric, ylab in ((ax_r2, "R2", "R²"), (ax_rmse, "RMSE", "RMSE (nm)")):
        label_top = {}
        for k, (subset, cv) in enumerate(BARS_R3):
            off = (k - 1.5) * w
            vals, errs, tops = [], [[], []], []
            for m in MODEL_ORDER:
                row = val[(val.subset == subset) & (val.model == m) & (val.cv == cv)]
                v = float(row[metric].iloc[0]) if len(row) else np.nan
                vals.append(v)
                b = bounds(m, subset, cv, metric, v)
                errs[0].append(v - b[0] if b else 0.0)
                errs[1].append(b[1] - v if b else 0.0)
                tops.append(b[1] if b else v)
                if b:
                    span[metric][0] = min(span[metric][0], b[0])
                    span[metric][1] = max(span[metric][1], b[1])
            st = BAR_STYLE_R3[(subset, cv)]
            yerr = np.array(errs) if ci is not None and np.any(errs) else None
            ax.bar(x + off, vals, w,
                   color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                   alpha=st["alpha"], hatch=st["hatch"],
                   edgecolor="black", linewidth=0.5,
                   yerr=yerr, capsize=2.5, error_kw=dict(lw=0.8, ecolor="black"))
            label_top[(subset, cv)] = (vals, tops, off)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, fontsize=9)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)

        # Value labels clear the whisker cap, not just the bar top.
        pad = 0.015 * (span[metric][1] - span[metric][0])
        for vals, tops, off in label_top.values():
            for xi, v, t in zip(x + off, vals, tops):
                if np.isfinite(v):
                    ax.text(xi, t + pad, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6.5, rotation=90)

    # Headroom for the rotated labels above the tallest whisker; the R2 axis is
    # allowed below zero because the condition-level intervals reach there.
    r2_lo, r2_hi = span["R2"]
    ax_r2.set_ylim(min(0.0, r2_lo) - 0.06, max(1.0, r2_hi * 1.22))
    if r2_lo < 0:
        ax_r2.axhline(0, color="black", lw=0.7, alpha=0.6, zorder=1)
    ax_rmse.set_ylim(0, span["RMSE"][1] * 1.22)
    ax_rmse.axhline(PURE_ERROR_SD, color="dimgray", lw=1.1, ls="--")
    # The line sits inside the bar field across the whole width, so the
    # annotation goes in the clear space at the top of the panel.
    ax_rmse.text(0.015, 0.985,
                 f"dashed line: pooled pure-error SD, {PURE_ERROR_SD:.2f} nm\n"
                 "(repeated process conditions)",
                 transform=ax_rmse.transAxes, fontsize=8, color="dimgray",
                 ha="left", va="top")

    handles = [Patch(facecolor="0.55", edgecolor="black", linewidth=0.5,
                     alpha=BAR_STYLE_R3[b]["alpha"], hatch=BAR_STYLE_R3[b]["hatch"],
                     label=f"{'OFAT-only' if b[0] == 'OFAT' else 'Augmented'}, "
                           f"{'LOOCV' if b[1] == 'LOOCV' else 'LOCO CV'}")
               for b in BARS_R3]
    # Lower left: the condition-level intervals push the R2 axis well below
    # zero, leaving that corner empty, while the upper left carries the labels.
    ax_r2.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=False, ncol=1)

    for ax, letter in zip((ax_r2, ax_rmse), ("a", "b")):
        ax.text(-0.10, 1.02, letter, transform=ax.transAxes, fontsize=16,
                fontweight="bold", ha="left", va="bottom")

    _save(fig, "FigureR3")


# ── Figure R4 ───────────────────────────────────────────────────────────────
def figR4():
    _style(11)
    preds_path = _require(RESULTS / "predictions_sametest.csv")
    summary_path = RESULTS / "metrics_sametest.csv"

    # The scatter shows a single representative seed, because ten overlaid
    # seeds would be unreadable.  The metric boxes quote the across-seed median
    # from the metrics file, so the numbers agree with the tables rather than
    # with one draw of the inner-fold randomisation.
    d = pd.read_csv(preds_path)
    d = d[d.seed == PLOT_SEED] if "seed" in d.columns else d
    piv = d.pivot_table(index="row_id", columns=["model", "train_mode"],
                        values="y_pred").sort_index()
    base = d.drop_duplicates("row_id").set_index("row_id").sort_index()
    y = base["y_true"].to_numpy(float)
    is_ofat = (base["source"].astype(str).str.upper() == "OFAT").to_numpy()
    pred = {m: (piv[(m, "A")].to_numpy(), piv[(m, "B")].to_numpy()) for m in MODEL_ORDER}

    medians, n_seeds = None, 0
    if summary_path.exists():
        m = pd.read_csv(summary_path)
        if "cv" in m.columns:
            m = m[m.cv == "LOCO"]
        g = m.groupby(["model", "train_mode", "test_domain"])[["R2", "RMSE"]].median()
        n_seeds = int(m.seed.nunique())
        medians = {k: (float(v.R2), float(v.RMSE)) for k, v in g.iterrows()}
        print(f"  metric boxes: {n_seeds}-seed medians from {summary_path.name}")
    else:
        print(f"  metric boxes: seed-{PLOT_SEED} values ({summary_path.name} absent)")

    def metrics(yt, yp):
        r = yt - yp
        rmse = float(np.sqrt(np.mean(r ** 2)))
        r2 = 1 - float(np.sum(r ** 2)) / float(np.sum((yt - yt.mean()) ** 2))
        return r2, rmse

    allv = np.concatenate([y] + [v for p in pred.values() for v in p])
    lo, hi = float(allv.min()) - 2, float(allv.max()) + 2

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.2), constrained_layout=True)
    for ax, model in zip(axes.ravel(), MODEL_ORDER):
        yA, yB = pred[model]
        for mask, c in ((is_ofat, COLOR_OFAT), (~is_ofat, COLOR_DOE)):
            ax.scatter(y[mask], yA[mask], s=34, facecolors="none", edgecolors=c,
                       linewidths=1.0, alpha=0.9, zorder=2)
            ax.scatter(y[mask], yB[mask], s=34, c=c, edgecolors="black",
                       linewidths=0.35, alpha=0.8, zorder=3)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        ax.set_xlabel("Measured thickness (nm)")
        ax.set_ylabel("Predicted thickness (nm)")
        ax.set_title(model, fontsize=11, fontweight="bold", pad=5)
        ax.grid(True, lw=0.4, alpha=0.3)

        lines = [f"median of {n_seeds} seeds"] if medians else [f"seed {PLOT_SEED}"]
        for lab, mask in (("all 57", np.ones(len(y), bool)), ("OFAT rows", is_ofat)):
            if medians:
                rA, eA = medians[(model, "A", lab)]
                rB, eB = medians[(model, "B", lab)]
            else:
                rA, eA = metrics(y[mask], yA[mask])
                rB, eB = metrics(y[mask], yB[mask])
            lines.append(f"{lab} (n = {int(mask.sum())})")
            lines.append(f"  A  R² {rA:5.2f}  RMSE {eA:4.2f}")
            lines.append(f"  B  R² {rB:5.2f}  RMSE {eB:4.2f}")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
                ha="left", fontsize=6.8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9,
                          edgecolor="lightgray"))

    handles = [
        Line2D([], [], ls="", marker="o", mfc="none", mec="0.25", ms=7,
               label="A: trained on OFAT rows only"),
        Line2D([], [], ls="", marker="o", mfc="0.45", mec="black", ms=7,
               label="B: trained on OFAT + D-optimal rows"),
        Line2D([], [], ls="", marker="o", mfc=COLOR_OFAT, mec="black", ms=7,
               label="OFAT-domain test rows (n = 42)"),
        Line2D([], [], ls="", marker="o", mfc=COLOR_DOE, mec="black", ms=7,
               label="D-optimal test rows (n = 15)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.045))

    for ax, letter in zip(axes.ravel(), ("a", "b", "c", "d")):
        ax.text(-0.14, 1.03, letter, transform=ax.transAxes, fontsize=16,
                fontweight="bold", ha="left", va="bottom")

    _save(fig, "FigureR4")


FIGURES = {"fig3": fig3, "fig4": fig4, "figR3": figR3, "figR4": figR4}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", choices=list(FIGURES),
                    help="which figures to draw (default: all)")
    ap.add_argument("--out", type=Path, default=None,
                    help="directory for the PNGs (default: figures/)")
    args = ap.parse_args()
    if args.out is not None:
        global OUT
        OUT = args.out
    for name in (args.which or list(FIGURES)):
        FIGURES[name]()


if __name__ == "__main__":
    main()
