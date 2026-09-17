"""Sequential D-optimal augmentation of an existing experimental record.

The design quantities are defined once in `design`. Two command-line entry
points use them:

    python -m dopt.select       propose the next experiments for a record
    python -m dopt.trajectory   the quantities over the additions made here
"""
from .design import (FEATURES, n_terms, model_matrix, standardize,
                     rank, logdet, kappa, mean_abs_r, metrics, aliased_terms)

__all__ = ["FEATURES", "n_terms", "model_matrix", "standardize",
           "rank", "logdet", "kappa", "mean_abs_r", "metrics", "aliased_terms"]
__version__ = "1.1.0"
