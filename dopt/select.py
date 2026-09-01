"""Propose the next experiments for an existing record, by sequential D-optimal selection.

The routine needs only the process conditions. It never reads a measured
response, so it runs on any record for which the conditions are known, including
designs published by other groups. Point it at a CSV of the experiments already
performed and a candidate range per parameter, both given in a config file.

    python -m dopt.select --config config/thiswork_hfo2_peald.toml --n-add 15
    python -m dopt.select --config config/yoon2023_pt_ald.toml --n-add 6

Config (TOML)

    name             label for the run
    conditions       CSV of experiments already performed, one row each.
                     Columns not listed under [[parameters]] are ignored, so a
                     file that also carries measured responses can be used as is.
    levels           grid levels per parameter
    round_to_integer round proposals to whole numbers
    [[parameters]]   one block per parameter: column, min, max

At each step the candidate maximising det(M + f(x)'f(x)) is chosen, which by the
matrix determinant lemma is the candidate maximising the leverage
h(x) = f(x)' M^-1 f(x). rank, log10 det(X'X), the condition number and mean |r|
are reported after every addition, so the user can decide where to stop.

The candidate set evaluated at each step is selectable:

    --n-candidate omitted   the full grid; deterministic
    --n-candidate N         N candidates drawn per step, for grids too large to
                            score exhaustively; --seed makes the draw repeatable

M is regularised with eps*I, as in the manuscript. --pinv substitutes the
Moore-Penrose leverage h(x) = x' M+ x, which is not equivalent: M+ maps the null
space to zero and so cannot raise the rank.
"""
import argparse, itertools, pathlib, sys, tomllib
import numpy as np, pandas as pd

from . import design as D

ROOT = pathlib.Path(__file__).resolve().parents[1]
EPS = 1e-8


def load_config(path):
    cfg = tomllib.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for key in ("conditions", "parameters"):
        if key not in cfg:
            raise SystemExit(f"{path}: '{key}' is required")
    cols = [p["column"] for p in cfg["parameters"]]
    bounds = np.array([[p["min"], p["max"]] for p in cfg["parameters"]], float)
    if (bounds[:, 0] >= bounds[:, 1]).any():
        raise SystemExit(f"{path}: every parameter needs min < max")
    csv = pathlib.Path(cfg["conditions"])
    if not csv.is_absolute():
        csv = ROOT / csv
    df = pd.read_csv(csv)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv}: missing column(s) {', '.join(missing)}")
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise SystemExit(f"{csv}: non-numeric or empty entries in {cols}")
    return {"name": cfg.get("name", pathlib.Path(path).stem),
            "columns": cols, "bounds": bounds, "X": X.to_numpy(float),
            "levels": int(cfg.get("levels", 7)),
            "round": bool(cfg.get("round_to_integer", False)),
            "source": csv}


def select(X_have, bounds, levels, n_add, n_candidate=None, seed=None,
           round_int=False, pinv=False):
    grid = np.array(list(itertools.product(
        *[np.linspace(bounds[i, 0], bounds[i, 1], levels) for i in range(bounds.shape[0])])))
    both = np.vstack([X_have, grid])
    mn, mx = both.min(0), both.max(0)
    span = np.where(mx - mn == 0, 1.0, mx - mn)
    nrm = lambda A: (A - mn) / span
    Gn, cur = nrm(grid), nrm(X_have).copy()
    rng = np.random.default_rng(seed)
    taken, out = set(), []
    for step in range(1, n_add + 1):
        Q = D.model_matrix(cur)
        M = Q.T @ Q
        Mi = np.linalg.pinv(M) if pinv else np.linalg.inv(M + EPS * np.eye(M.shape[0]))
        idx = (np.arange(len(Gn)) if n_candidate is None
               else rng.choice(len(Gn), size=min(n_candidate, len(Gn)), replace=False))
        idx = np.array([i for i in idx if i not in taken])
        if not len(idx):
            raise SystemExit("candidate grid exhausted; raise levels or n-candidate")
        G = D.model_matrix(Gn[idx])
        best = int(idx[int(np.argmax(np.einsum("ij,jk,ik->i", G, Mi, G)))])
        taken.add(best)
        cur = np.vstack([cur, Gn[best]])
        chosen = grid[sorted(taken)]
        if round_int:
            chosen = np.round(chosen)
        proposal = np.round(grid[best]) if round_int else grid[best]
        # metrics on the working coordinates; see the numerical note in design.py
        out.append({"step": step, **dict(zip(range(len(proposal)), proposal)),
                    **D.metrics(nrm(np.vstack([X_have, chosen])))})
    return pd.DataFrame(out), grid


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--n-add", type=int, default=15)
    ap.add_argument("--n-candidate", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-round", action="store_true", help="override round_to_integer")
    ap.add_argument("--pinv", action="store_true", help="Moore-Penrose leverage instead of eps*I")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "outputs" / "d_optimal_selection.csv")
    a = ap.parse_args()

    cfg = load_config(a.config)
    round_int = cfg["round"] and not a.no_round
    df, grid = select(cfg["X"], cfg["bounds"], cfg["levels"], a.n_add,
                      a.n_candidate, a.seed, round_int, a.pinv)
    df = df.rename(columns={i: c for i, c in enumerate(cfg["columns"])})

    p = len(cfg["columns"])
    mode = "full grid" if a.n_candidate is None else f"{a.n_candidate} candidates/step (seed={a.seed})"
    rule = "Moore-Penrose" if a.pinv else f"eps*I (eps={EPS:g})"
    print(f"# {cfg['name']}")
    print(f"# {len(cfg['X'])} experiments in {cfg['source'].name}, {p} parameters, "
          f"P = {D.n_terms(p)} quadratic terms")
    print(f"# grid {cfg['levels']}^{p} = {len(grid)} candidates · {mode} · {rule}\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
