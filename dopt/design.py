"""Design-matrix quantities used by the selection routine.

Three quantities describe a design and none of them uses the measured response:

    rank      rank of the quadratic model matrix X (P = 1 + 2p + p(p-1)/2 terms).
              rank < P means at least one model term is an exact linear
              combination of the others and is not estimable at any sample size.
              Judged by numpy.linalg.matrix_rank at the standard SVD tolerance,
              so no threshold has to be chosen.
    logdet    log10 det(X'X) on the same quadratic basis. det(X'X) is zero
              while rank < P, so this carries the same threshold-free statement
              as rank and is the quantity the D-optimal criterion maximises.
    kappa     cond(X'X) on the standardized p factors. Requires a threshold.
    mean|r|   mean off-diagonal |Pearson r| over the standardized p factors.
              Requires a threshold.

The two bases differ and the difference matters: rank is taken on the P-term
quadratic basis, kappa and mean|r| on the p-factor basis, following Section 4.4
of the manuscript. Report which basis is used whenever these are quoted.

Numerical note: do not round coordinates before calling rank(). Rounding a
star-shaped OFAT design to a fixed number of decimals introduces ~1e-10 axis
departures that matrix_rank counts as real dimensions and inflates the rank.
"""
from __future__ import annotations
import itertools
import numpy as np

FEATURES = ["Plasma_Power_W", "Cycles", "Chuck_T_C", "O2_Flow_sccm", "Oxidant_Dose_s"]


def n_terms(p: int) -> int:
    """Number of terms in a full quadratic model in p factors."""
    return 1 + 2 * p + p * (p - 1) // 2


def model_matrix(X: np.ndarray) -> np.ndarray:
    """Quadratic model matrix: intercept, p linear, p square, p(p-1)/2 two-way."""
    X = np.asarray(X, float)
    n, p = X.shape
    cols = [np.ones(n)] + [X[:, i] for i in range(p)] + [X[:, i] ** 2 for i in range(p)]
    cols += [X[:, i] * X[:, j] for i, j in itertools.combinations(range(p), 2)]
    return np.column_stack(cols)


def standardize(X: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per column. Constant columns are left alone."""
    X = np.asarray(X, float)
    s = X.std(0)
    return (X - X.mean(0)) / np.where(s == 0, 1.0, s)


def rank(X: np.ndarray) -> int:
    """rank of the quadratic model matrix built from X."""
    return int(np.linalg.matrix_rank(model_matrix(X)))


def kappa(X: np.ndarray) -> float:
    """cond(X'X) on the standardized factors."""
    sv = np.linalg.svd(standardize(X), compute_uv=False)
    return float(sv[0] ** 2 / sv[-1] ** 2) if sv[-1] > 0 else float("inf")


def mean_abs_r(X: np.ndarray) -> float:
    """Mean off-diagonal |Pearson r| over the standardized factors."""
    R = np.corrcoef(standardize(X), rowvar=False)
    iu = np.triu_indices(R.shape[0], 1)
    return float(np.abs(R[iu]).mean())


def logdet(X: np.ndarray) -> float:
    """log10 det(X'X) on the quadratic model matrix.

    This is the quantity the D-optimal criterion maximises. While rank < P the
    information matrix is singular and det(X'X) is zero, so the logarithm is
    undefined and NaN is returned. The step from NaN to a finite value is the
    same event as rank reaching P, expressed in the currency of the selection
    rule rather than as a count.
    """
    Q = model_matrix(X)
    if np.linalg.matrix_rank(Q) < Q.shape[1]:
        return float("nan")
    return float(np.log10(np.linalg.svd(Q.T @ Q, compute_uv=False)).sum())


def metrics(X: np.ndarray) -> dict:
    """All four, on the bases documented above."""
    return {"rank": rank(X), "logdet": logdet(X), "kappa": kappa(X), "mean_r": mean_abs_r(X)}


def aliased_terms(X: np.ndarray, tol: float = 1e-8) -> list[str]:
    """Names of the model terms that are not identifiable from X.

    Greedy: add columns left to right, keep a column only if it raises the rank.
    The complement is one valid description of the deficit; it is not unique,
    because which of a set of collinear terms is called aliased is a choice.
    """
    p = np.asarray(X, float).shape[1]
    names = (["1"] + [f"x{i+1}" for i in range(p)] + [f"x{i+1}^2" for i in range(p)]
             + [f"x{i+1}x{j+1}" for i, j in itertools.combinations(range(p), 2)])
    Q = model_matrix(X)
    kept, out = [], []
    for k in range(Q.shape[1]):
        trial = kept + [k]
        if np.linalg.matrix_rank(Q[:, trial], tol=tol) > len(kept):
            kept = trial
        else:
            out.append(names[k])
    return out
