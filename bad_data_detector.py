"""
Takes inputs of "[PA systoic - PA diastolic] / PA mean" direclty from Merlin.net to create an ever expanding array.
Indentifies if a new value deviates sharply from baseline.
Requires a separate mechanism to review and exclude bad data.
"""

import numpy as np

def t_prediction_test(baseline, x_new, alpha=0.01):
    """
    Two-sided test for whether x_new deviates from a baseline (one future obs).
    Returns: dict with interval, p-value, and a boolean flag.
    """
    baseline = np.asarray(baseline, dtype=float)
    n = baseline.size
    if n < 2:
        raise ValueError("Need at least 2 baseline points.")
    xbar = baseline.mean()
    s = baseline.std(ddof=1)
    se_pred = s * np.sqrt(1 + 1/n)
    df = n - 1

    try:
        from scipy.stats import t
        tcrit = t.ppf(1 - alpha/2, df)
        tstat = (x_new - xbar) / se_pred
        p = 2 * (1 - t.cdf(abs(tstat), df))
    except Exception:
        # Fallback to normal approx if SciPy isn't available
        from math import erf, sqrt
        zcrit = 2.5758293035489004 if alpha == 0.01 else 1.959963984540054  # 99% or 95%
        tcrit = zcrit
        z = (x_new - xbar) / se_pred
        p = 2 * (1 - 0.5 * (1 + erf(abs(z)/sqrt(2))))

    lo = xbar - tcrit * se_pred
    hi = xbar + tcrit * se_pred
    is_outlier = (x_new < lo) or (x_new > hi)
    return {"lower": lo, "upper": hi, "p_value": p, "outlier": is_outlier}
