"""Cross-validation and uncertainty analysis for the HfO2 PEALD record.

Predictive performance is measured by leave-one-condition-out cross-validation:
every wafer that shares a process condition is held out together, so a repeated
condition can never be both in the training set and in the test set.  The four
regressors of the study (Ridge, Random Forest, XGBoost, a Gaussian process) are
tuned inside each training fold and scored on the held-out condition.

    python -m cv.pipeline      fit the folds and save metrics and predictions
    python -m cv.bootstrap     condition-level bootstrap confidence intervals
    python -m cv.tables        the summary tables in Markdown
    python -m cv.figure_data   reduce the results to the figure inputs
    python -m cv.figures       draw the figures

Every module is run from the repository root and resolves its paths from there.
"""
__all__ = []
