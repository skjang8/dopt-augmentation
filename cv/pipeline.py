"""Leave-one-condition-out cross-validation of the thickness models.

Leave-one-condition-out (LOCO) cross-validation holds out every wafer that
shares a five-parameter process condition in the same fold.  Leave-one-out
cross-validation (LOOCV) holds out one wafer at a time, so a condition that was
run more than once appears on both sides of the split; LOCO removes that.  The
42 one-factor-at-a-time (OFAT) wafers cover 24 distinct conditions and the 57
OFAT plus D-optimal wafers cover 39.

Four regressors are compared: Ridge, Random Forest, XGBoost and a Gaussian
process.  Inside every training fold the hyperparameters are chosen by a
3-fold grid search (the Gaussian process learns its kernel from the marginal
likelihood instead).  Out-of-fold predictions are written for every fold so
that a condition-level bootstrap can be run afterwards without refitting.

Parts
-----
dopt       the 15 D-optimal wafers appended to the 42 OFAT wafers one at a
           time in D-optimal selection order, n_added = 0..15; LOCO.
ofatseq    the OFAT wafers accumulating in file order, n = 6..42; LOCO.  A
           prefix holding fewer than two distinct conditions has no LOCO split
           and is recorded with missing metrics.
sametest   for every one of the 39 conditions, model A trains on the OFAT
           wafers outside that condition and model B on all 57 wafers outside
           it; both predict the same held-out wafers.  Scores are reported for
           the OFAT rows, the D-optimal rows and all 57.
fivepos    LOCO on the 42 and on the 57 wafers with the five-position mean
           thickness as the target instead of the four-position mean.
loocv_ref  LOOCV and LOCO on the 42 and on the 57 wafers with the same seeds
           and pipelines, so the two splits can be compared directly.

Row identity and row order
--------------------------
`row_id` is the 0-based index into `data/combined_clean.csv`: 0..41 are the
OFAT wafers and 42..56 the 15 D-optimal wafers in that file's order, which is
not their selection order.  `group_id` is the condition index obtained by
factorising the five process settings over the same frame, 0..38.  Both are
stable across parts and seeds.

Every design matrix built here is a prefix or a subset of one permutation of
the 57 rows: the 42 OFAT wafers in file order followed by the D-optimal wafers
in selection order (`data/doe_doptimal_order.csv`).  `Data.order57` holds that
permutation and `pos` in the prediction files is the position of a wafer inside
it.  Row order is not inert, because the shuffled inner folds, the Random
Forest bootstrap and the XGBoost row subsample all key off positional index, so
fixing one order keeps the parts comparable with each other.

Results are appended as they are produced and an interrupted run resumes from
what is already on disk.

    python -m cv.pipeline --parts dopt ofatseq sametest fivepos loocv_ref \
        --seeds 42 43 44 45 46 47 48 49 50 51
"""
from __future__ import annotations

import os

# Keep BLAS single-threaded: joblib already saturates the cores with processes
# and nested threading both oversubscribes and perturbs float summation order.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import det
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
COMBINED_PATH = DATA_DIR / "combined_clean.csv"
DOE_PATH = DATA_DIR / "doe_doptimal_order.csv"
POSITIONS_PATH = DATA_DIR / "thickness_positions.csv"
RESULTS_DIR = ROOT / "results" / "cv"

FEATURES = ["Plasma_Power_W", "Cycles", "Chuck_T_C", "O2_Flow_sccm", "Oxidant_Dose_s"]
POINTS = ["point_1_nm", "point_2_nm", "point_3_nm", "point_4_nm"]
RETAINED = ["T1", "T3", "T4", "T5"]
TARGET_4POS = "Thickness_mean"
TARGET_5POS = "Thickness_5pos"

N_MIN = 6
MIN_N_FOR_TUNING = 15
INNER_CV_SPLITS = 3

METRIC_COLS = [
    "part", "step_id", "variant", "model", "seed", "cv", "subset", "target",
    "train_mode", "test_domain", "n_added", "n", "n_total", "n_groups",
    "n_test", "R2", "RMSE", "MAE",
]
PRED_COLS = [
    "part", "step_id", "variant", "model", "seed", "cv", "subset", "target",
    "train_mode", "n_added", "n", "pos", "row_id", "group_id", "source",
    "y_true", "y_pred",
]


# ── Pipelines ───────────────────────────────────────────────────────────────
def make_pipe_ridge(seed):
    return Pipeline([("sc", StandardScaler()), ("est", Ridge())])


def make_pipe_rf(seed):
    return Pipeline([("est", RandomForestRegressor(
        n_estimators=100, random_state=seed, n_jobs=1))])


def make_pipe_xgb(seed):
    return Pipeline([("est", xgb.XGBRegressor(
        n_estimators=100, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, random_state=seed, verbosity=0,
        tree_method="hist", device="cpu", nthread=1))])


def make_pipe_gpr(seed):
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(len(FEATURES)),
              length_scale_bounds=(0.1, 100.0))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
    )
    return Pipeline([("sc", StandardScaler()), ("est", GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=15, alpha=1e-10,
        normalize_y=True, random_state=seed))])


PARAM_GRIDS = {
    "Ridge": {"est__alpha": np.logspace(-3, 3, 13).tolist()},
    "Random Forest": {
        "est__max_depth": [3, 5, None],
        "est__min_samples_leaf": [1, 2, 5],
        "est__max_features": ["sqrt", 1.0],
    },
    "XGBoost": {
        "est__max_depth": [2, 3, 5],
        "est__learning_rate": [0.05, 0.1],
        "est__reg_alpha": [0.0, 0.1, 1.0],
        "est__reg_lambda": [1.0, 10.0],
    },
    "GPR": {},  # kernel hyperparameters learned via marginal likelihood
}

MODEL_FACTORIES = {
    "Ridge": make_pipe_ridge,
    "Random Forest": make_pipe_rf,
    "XGBoost": make_pipe_xgb,
    "GPR": make_pipe_gpr,
}
MODEL_ORDER = list(MODEL_FACTORIES)


# ── Fold worker ─────────────────────────────────────────────────────────────
def _fit_predict(X, y, train_idx, test_idx, model_name, seed):
    """One outer fold: optional inner 3-fold grid search, then predict."""
    warnings.simplefilter("ignore")   # joblib workers reset the filters
    pipe = MODEL_FACTORIES[model_name](seed)
    grid = PARAM_GRIDS[model_name]
    if grid and len(train_idx) >= MIN_N_FOR_TUNING:
        gs = GridSearchCV(
            pipe, grid,
            cv=KFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=seed),
            scoring="neg_mean_squared_error", n_jobs=1,
            error_score=float("-inf"),
        )
        gs.fit(X[train_idx], y[train_idx])
        yhat = gs.predict(X[test_idx])
    else:
        pipe.fit(X[train_idx], y[train_idx])
        yhat = pipe.predict(X[test_idx])
    return np.asarray(yhat, dtype=float)


def _job(step_key, variant, model_name, seed, fold_i, X, y, tr, te):
    return (step_key, variant, model_name, seed, fold_i, te,
            _fit_predict(X, y, tr, te, model_name, seed))


def design_metrics(X):
    """Mean |correlation|, condition number and |det| of the standardized
    intercept-plus-five-factor information matrix."""
    corr = np.corrcoef(X.T)
    triu = np.triu_indices(corr.shape[0], k=1)
    mean_abs_corr = float(np.mean(np.abs(corr[triu])))
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    Xs_aug = np.column_stack([np.ones(len(Xs)), Xs])
    XtX = Xs_aug.T @ Xs_aug
    return mean_abs_corr, float(np.linalg.cond(XtX)), float(np.abs(det(XtX)))


# ── Data ────────────────────────────────────────────────────────────────────
class Data:
    """The 57-row frame, the D-optimal selection order and both targets."""

    def __init__(self):
        comb = pd.read_csv(COMBINED_PATH).reset_index(drop=True)
        doe = pd.read_csv(DOE_PATH)

        # --- consistency checks -------------------------------------------
        is_ofat = (comb["source"].to_numpy(object) == "OFAT")
        n_ofat = int(is_ofat.sum())
        n_doe = len(doe)
        assert len(comb) == n_ofat + n_doe, "combined_clean row count mismatch"
        assert is_ofat[:n_ofat].all() and not is_ofat[n_ofat:].any(), \
            "the OFAT wafers are not the leading block of combined_clean"

        comb_doe = comb.iloc[n_ofat:][FEATURES].to_numpy(float)
        doe_X = doe[FEATURES].to_numpy(float)
        # map each D-optimal selection step onto its row of combined_clean
        order = []
        used = set()
        for r in doe_X:
            hit = [j for j in range(n_doe)
                   if j not in used and np.allclose(comb_doe[j], r)]
            assert len(hit) >= 1, f"D-optimal row {r} absent from combined_clean"
            used.add(hit[0])
            order.append(n_ofat + hit[0])
        assert len(used) == n_doe, "D-optimal rows do not biject onto combined_clean"
        assert np.allclose(doe[TARGET_4POS].to_numpy(float),
                           comb[TARGET_4POS].to_numpy(float)[order]), \
            "D-optimal targets differ from combined_clean"

        # --- five-position target -----------------------------------------
        comb = self._attach_5pos(comb)

        self.df = comb
        self.n_ofat = n_ofat
        self.dopt_row_ids = np.asarray(order, dtype=int)
        # OFAT wafers in file order, then the D-optimal wafers in selection
        # order.  Every design matrix in this module is a prefix or a subset
        # of this permutation.
        self.order57 = np.concatenate(
            [np.arange(n_ofat), self.dopt_row_ids]).astype(int)
        self.X = comb[FEATURES].to_numpy(float)
        self.y4 = comb[TARGET_4POS].to_numpy(float)
        self.y5 = comb[TARGET_5POS].to_numpy(float)
        self.source = comb["source"].to_numpy(object)
        self.group_id = pd.factorize(
            pd.Series([tuple(r) for r in comb[FEATURES].values]))[0]
        self.is_ofat = (self.source == "OFAT")
        assert self.group_id.max() + 1 == 39, "expected 39 unique conditions"

    @staticmethod
    def _attach_5pos(comb):
        """Take the five-position mean from `data/thickness_positions.csv`.

        That file carries the same 57 wafers in the same order, with the
        thickness at all five measurement positions.  T1, T3, T4 and T5 are the
        four readings already in `combined_clean.csv`; T2, adjacent to the
        precursor outlet, is the one left out of the prediction target.  The
        two files are checked row by row before the extra columns are used.

        The target is recomputed from the five readings rather than read from
        the `Thickness_5pos` column, which is a convenience column and carries
        the rounding of its own text representation.
        """
        pos = pd.read_csv(POSITIONS_PATH)
        assert len(pos) == len(comb), "thickness_positions row count mismatch"
        assert np.allclose(pos[FEATURES].to_numpy(float),
                           comb[FEATURES].to_numpy(float)), \
            "thickness_positions process settings differ from combined_clean"
        assert np.allclose(pos[RETAINED].to_numpy(float),
                           comb[POINTS].to_numpy(float)), \
            "thickness_positions retained readings differ from combined_clean"
        assert np.allclose(pos[TARGET_4POS].to_numpy(float),
                           comb[TARGET_4POS].to_numpy(float)), \
            "thickness_positions four-position mean differs from combined_clean"
        assert (pos["source"].to_numpy(object)
                == comb["source"].to_numpy(object)).all(), \
            "thickness_positions row order differs from combined_clean"
        out = comb.copy()
        t2 = pos["T2"].to_numpy(float)
        out["T2"] = t2
        out[TARGET_5POS] = (out[POINTS].sum(axis=1) + out["T2"]) / 5
        assert np.allclose(out[TARGET_5POS].to_numpy(float),
                           pos[TARGET_5POS].to_numpy(float)), \
            "thickness_positions five-position mean differs from combined_clean"
        return out


# ── Step construction ───────────────────────────────────────────────────────
def _loco_folds(groups):
    if len(np.unique(groups)) < 2:
        return None
    return [(tr, te) for tr, te in LeaveOneGroupOut().split(
        np.zeros((len(groups), 1)), groups=groups)]


def _loo_folds(n):
    return [(tr, te) for tr, te in LeaveOneOut().split(np.zeros((n, 1)))]


def _step(part, step_id, row_ids, y, data, variants, meta, cv):
    row_ids = np.asarray(row_ids, dtype=int)
    return {
        "part": part, "step_id": step_id, "cv": cv, "meta": meta,
        "row_ids": row_ids,
        "X": data.X[row_ids], "y": y[row_ids],
        "group_id": data.group_id[row_ids], "source": data.source[row_ids],
        "variants": variants,
    }


def build_steps(part, data):
    """Return a list of step dicts for `part` (data only, no models or seeds)."""
    steps = []
    n_ofat = data.n_ofat

    if part == "dopt":
        for n_added in range(0, len(data.dopt_row_ids) + 1):
            row_ids = data.order57[:n_ofat + n_added]
            g = data.group_id[row_ids]
            folds = _loco_folds(g)
            meta = {"n_added": n_added, "n_total": len(row_ids),
                    "n_groups": len(np.unique(g)), "subset": "dopt",
                    "target": TARGET_4POS}
            steps.append(_step("dopt", f"n_added={n_added}", row_ids, data.y4,
                               data, {"": folds}, meta, "LOCO"))

    elif part == "ofatseq":
        for n in range(N_MIN, n_ofat + 1):
            row_ids = data.order57[:n]
            g = data.group_id[row_ids]
            folds = _loco_folds(g)
            meta = {"n": n, "n_total": n, "n_groups": len(np.unique(g)),
                    "subset": "OFAT", "target": TARGET_4POS}
            steps.append(_step("ofatseq", f"n={n}", row_ids, data.y4,
                               data, {"": folds}, meta, "LOCO"))

    elif part == "sametest":
        row_ids = data.order57
        g = data.group_id[row_ids]                  # local index space
        is_ofat = data.is_ofat[row_ids]
        variants = {}
        for mode in ("A", "B"):
            folds = []
            for c in np.unique(g):
                te = np.where(g == c)[0]
                keep = is_ofat if mode == "A" else np.ones(len(g), bool)
                tr = np.where((g != c) & keep)[0]
                folds.append((tr, te))
            variants[mode] = folds
        meta = {"n_total": len(row_ids), "n_groups": len(np.unique(g)),
                "subset": "Augmented", "target": TARGET_4POS}
        steps.append(_step("sametest", "all57", row_ids, data.y4,
                           data, variants, meta, "LOCO"))

    elif part == "fivepos":
        for subset, row_ids in (("OFAT", data.order57[:n_ofat]),
                                ("Augmented", data.order57)):
            g = data.group_id[row_ids]
            meta = {"n_total": len(row_ids), "n_groups": len(np.unique(g)),
                    "subset": subset, "target": TARGET_5POS}
            steps.append(_step("fivepos", subset, row_ids, data.y5, data,
                               {"": _loco_folds(g)}, meta, "LOCO"))

    elif part == "loocv_ref":
        for subset, row_ids in (("OFAT", data.order57[:n_ofat]),
                                ("Augmented", data.order57)):
            g = data.group_id[row_ids]
            meta = {"n_total": len(row_ids), "n_groups": len(np.unique(g)),
                    "subset": subset, "target": TARGET_4POS}
            steps.append(_step("loocv_ref", f"{subset}|LOOCV", row_ids, data.y4,
                               data, {"": _loo_folds(len(row_ids))}, meta, "LOOCV"))
            steps.append(_step("loocv_ref", f"{subset}|LOCO", row_ids, data.y4,
                               data, {"": _loco_folds(g)}, meta, "LOCO"))

    else:
        raise ValueError(f"unknown part {part!r}")

    return steps


# ── Metrics assembly ────────────────────────────────────────────────────────
def _metric_rows(step, variant, model, seed, y_pred, covered, data):
    base = dict.fromkeys(METRIC_COLS, "")
    base.update({"part": step["part"], "step_id": step["step_id"],
                 "variant": variant, "model": model, "seed": seed,
                 "cv": step["cv"]})
    base.update({k: v for k, v in step["meta"].items()})
    if variant in ("A", "B"):
        base["train_mode"] = variant

    y = step["y"]
    rows = []
    if step["part"] == "sametest":
        is_ofat = step["source"] == "OFAT"
        domains = [("OFAT rows", is_ofat), ("DOE rows", ~is_ofat),
                   ("all 57", np.ones(len(y), bool))]
    else:
        domains = [("all", np.ones(len(y), bool))]

    for dom, mask in domains:
        m = mask & covered
        r = dict(base)
        r["test_domain"] = dom
        r["n_test"] = int(m.sum())
        r["R2"] = r2_score(y[m], y_pred[m])
        r["RMSE"] = float(np.sqrt(mean_squared_error(y[m], y_pred[m])))
        r["MAE"] = float(mean_absolute_error(y[m], y_pred[m]))
        rows.append(r)
    return rows


def _pred_rows(step, variant, model, seed, y_pred, covered):
    base = dict.fromkeys(PRED_COLS, "")
    base.update({"part": step["part"], "step_id": step["step_id"],
                 "variant": variant, "model": model, "seed": seed,
                 "cv": step["cv"]})
    for k in ("subset", "target", "n_added", "n"):
        if k in step["meta"]:
            base[k] = step["meta"][k]
    if variant in ("A", "B"):
        base["train_mode"] = variant
    out = []
    for i in np.where(covered)[0]:
        r = dict(base)
        r.update({"pos": int(i),
                  "row_id": int(step["row_ids"][i]),
                  "group_id": int(step["group_id"][i]),
                  "source": step["source"][i],
                  "y_true": float(step["y"][i]),
                  "y_pred": float(y_pred[i])})
        out.append(r)
    return out


def _skipped_rows(step, models, seeds, reason):
    rows = []
    for model in models:
        for seed in seeds:
            base = dict.fromkeys(METRIC_COLS, "")
            base.update({"part": step["part"], "step_id": step["step_id"],
                         "variant": "", "model": model, "seed": seed,
                         "cv": step["cv"], "test_domain": "all",
                         "n_test": 0, "R2": np.nan, "RMSE": np.nan,
                         "MAE": np.nan})
            base.update(step["meta"])
            rows.append(base)
    print(f"  [{step['part']} {step['step_id']}] skipped: {reason}", flush=True)
    return rows


# ── Incremental output ──────────────────────────────────────────────────────
class Sink:
    def __init__(self, out_dir, part):
        self.metrics_path = out_dir / f"metrics_{part}.csv"
        self.preds_path = out_dir / f"predictions_{part}.csv"
        self.done = set()
        if self.metrics_path.exists():
            try:
                # keep_default_na=False: the empty `variant` field must read
                # back as "" (not NaN) or the resume key never matches.
                prev = pd.read_csv(self.metrics_path, keep_default_na=False)
                for _, r in prev.iterrows():
                    self.done.add((str(r["step_id"]), str(r.get("variant", "")),
                                   str(r["model"]), int(r["seed"])))
            except Exception as exc:          # corrupt/partial file -> start over
                print(f"  ! could not read {self.metrics_path.name}: {exc}")

    def is_done(self, step_id, variant, model, seed):
        return (str(step_id), str(variant), str(model), int(seed)) in self.done

    def _append(self, path, rows, cols):
        if not rows:
            return
        df = pd.DataFrame(rows)[cols]
        header = not path.exists()
        with open(path, "a", newline="") as fh:
            df.to_csv(fh, index=False, header=header)
            fh.flush()
            os.fsync(fh.fileno())

    def write(self, metric_rows, pred_rows):
        self._append(self.metrics_path, metric_rows, METRIC_COLS)
        self._append(self.preds_path, pred_rows, PRED_COLS)
        for r in metric_rows:
            self.done.add((str(r["step_id"]), str(r["variant"]),
                           str(r["model"]), int(r["seed"])))


# ── Runner ──────────────────────────────────────────────────────────────────
def run_part(part, data, seeds, models, out_dir, n_jobs, steps_filter=None):
    sink = Sink(out_dir, part)
    steps = build_steps(part, data)
    if steps_filter is not None:
        steps = [s for s in steps if s["step_id"] in steps_filter]
    t0 = time.time()

    for step in steps:
        # Flatten every (variant, model, seed, fold) of this step into one
        # batch so all cores stay busy even when a step has few folds.
        jobs, pending = [], []
        skipped = []
        for variant, folds in step["variants"].items():
            for model in models:
                for seed in seeds:
                    if sink.is_done(step["step_id"], variant, model, seed):
                        continue
                    if folds is None:
                        skipped.append((variant, model, seed))
                        continue
                    pending.append((variant, model, seed))
                    for fi, (tr, te) in enumerate(folds):
                        jobs.append(delayed(_job)(
                            step["step_id"], variant, model, seed, fi,
                            step["X"], step["y"], tr, te))
        if skipped:
            sink.write(_skipped_rows(step, models, seeds,
                                     "fewer than 2 distinct conditions"), [])
        if not jobs:
            continue

        results = Parallel(n_jobs=n_jobs, prefer="processes")(jobs)

        acc = {}
        for _sid, variant, model, seed, _fi, te, yhat in results:
            key = (variant, model, seed)
            if key not in acc:
                acc[key] = (np.full(len(step["y"]), np.nan),
                            np.zeros(len(step["y"]), bool))
            pred, cov = acc[key]
            pred[te] = yhat
            cov[te] = True

        mrows, prows = [], []
        for (variant, model, seed), (pred, cov) in acc.items():
            mrows += _metric_rows(step, variant, model, seed, pred, cov, data)
            prows += _pred_rows(step, variant, model, seed, pred, cov)
        sink.write(mrows, prows)

        head = [r for r in mrows if r["test_domain"] in ("all", "all 57")]
        for r in sorted(head, key=lambda r: (r["model"], r["seed"], r["variant"])):
            print(f"  [{part} {step['step_id']} {r['variant']:<1} "
                  f"{r['model']:<13} seed={r['seed']}] "
                  f"R2={r['R2']:.3f} RMSE={r['RMSE']:.3f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"[{part}] done in {time.time()-t0:.0f}s", flush=True)


def write_design(data, out_dir):
    rows = []
    for n_added in range(0, len(data.dopt_row_ids) + 1):
        row_ids = data.order57[:data.n_ofat + n_added]
        mac, cn, dv = design_metrics(data.X[row_ids])
        rows.append({"n_added": n_added, "n_total": len(row_ids),
                     "n_groups": len(np.unique(data.group_id[row_ids])),
                     "mean_abs_corr": mac, "cond_num": cn, "det_XtX": dv})
    pd.DataFrame(rows).to_csv(out_dir / "design_dopt.csv", index=False)

    rows = []
    for n in range(N_MIN, data.n_ofat + 1):
        row_ids = data.order57[:n]
        mac, cn, dv = design_metrics(data.X[row_ids])
        rows.append({"n": n, "n_total": n,
                     "n_groups": len(np.unique(data.group_id[row_ids])),
                     "mean_abs_corr": mac, "cond_num": cn, "det_XtX": dv})
    pd.DataFrame(rows).to_csv(out_dir / "design_ofatseq.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(42, 52)))
    ap.add_argument("--parts", nargs="+",
                    default=["dopt", "ofatseq", "sametest", "fivepos"],
                    choices=["dopt", "ofatseq", "sametest", "fivepos", "loocv_ref"])
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER,
                    choices=MODEL_ORDER)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--dopt-steps", type=int, nargs="+", default=None,
                    help="restrict the dopt part to these n_added values")
    ap.add_argument("--ofatseq-steps", type=int, nargs="+", default=None,
                    help="restrict the ofatseq part to these n values")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data = Data()
    print(f"data: {len(data.df)} wafers, {data.n_ofat} OFAT, "
          f"{data.group_id.max()+1} conditions; seeds={args.seeds}; "
          f"parts={args.parts}; n_jobs={args.n_jobs}", flush=True)

    if "dopt" in args.parts or "ofatseq" in args.parts:
        write_design(data, args.out)

    for part in args.parts:
        steps_filter = None
        if part == "dopt" and args.dopt_steps is not None:
            steps_filter = {f"n_added={v}" for v in args.dopt_steps}
        if part == "ofatseq" and args.ofatseq_steps is not None:
            steps_filter = {f"n={v}" for v in args.ofatseq_steps}
        run_part(part, data, args.seeds, args.models, args.out, args.n_jobs,
                 steps_filter)


if __name__ == "__main__":
    main()
