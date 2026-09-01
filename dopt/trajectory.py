"""Design quantities over the D-optimal additions made in this work.

Uses the 42 measured OFAT conditions and the 15 measured D-optimal conditions in
the order in which they were selected. No synthetic data is used.

    python -m dopt.trajectory
"""
import pathlib
import numpy as np, pandas as pd

from . import design as D

ROOT = pathlib.Path(__file__).resolve().parents[1]
P_TERMS = D.n_terms(len(D.FEATURES))


def trajectory():
    """Return one row per addition, from the 42 OFAT conditions to all 57."""
    ofat = pd.read_csv(ROOT / "data" / "ofat_clean.csv")[D.FEATURES].to_numpy(float)
    doe = (pd.read_csv(ROOT / "data" / "doe_doptimal_order.csv")
             .sort_values("dopt_step")[D.FEATURES].to_numpy(float))
    if len(ofat) != 42 or len(doe) != 15:
        raise SystemExit(f"expected 42 OFAT and 15 D-optimal rows, found {len(ofat)} and {len(doe)}")

    # Normalisation uses the combined 57 conditions, as in the manuscript.
    lo, hi = np.vstack([ofat, doe]).min(0), np.vstack([ofat, doe]).max(0)
    nrm = lambda A: (A - lo) / (hi - lo)
    rows = []
    for s in range(len(doe) + 1):
        X = np.vstack([nrm(ofat), nrm(doe[:s])]) if s else nrm(ofat)
        rows.append({"added": s, "n": len(X), **D.metrics(X)})
    return pd.DataFrame(rows), nrm(ofat)


def main():
    traj, ofat_n = trajectory()
    print(f"P = {P_TERMS} quadratic terms in {len(D.FEATURES)} factors\n")
    print(traj.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nFirst attainment")
    hits = [("det(X'X) > 0", traj[traj.logdet.notna()], "none"),
            ("kappa <= 10", traj[traj.kappa <= 10], "convention"),
            ("mean|r| <= 0.30", traj[traj.mean_r <= 0.30], "stated target")]
    for label, h, kind in hits:
        where = (f"addition {int(h.iloc[0].added)} (n = {int(h.iloc[0].n)})"
                 if len(h) else "not attained")
        print(f"  {label:18s} {where:26s} threshold: {kind}")
    print(f"  det(X'X) is zero, and rank is below {P_TERMS}, at every earlier addition.")

    print("\nNon-monotone steps (a single crossing is not a stopping proof)")
    for col in ("kappa", "mean_r"):
        d = traj[col].diff()
        for i in traj.index[1:]:
            if d[i] > 0:
                print(f"  {col:8s} addition {int(traj.added[i-1])} -> {int(traj.added[i])}: "
                      f"{traj[col][i-1]:.3f} -> {traj[col][i]:.3f}")

    print(f"\nTerms not identifiable from the 42 OFAT conditions "
          f"({P_TERMS - traj['rank'][0]} of {P_TERMS}):")
    print("  " + ", ".join(D.aliased_terms(ofat_n)))

    out = ROOT / "outputs" / "design_trajectory.csv"
    out.parent.mkdir(exist_ok=True)
    traj.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
