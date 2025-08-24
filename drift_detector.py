from typing import Dict, List
import numpy as np

def detect_drift_ewcusum(
    x: np.ndarray,
    alpha_baseline: float = 0.01,   # slow EWMA baseline (tracks very slow changes)
    alpha_var: float = 0.05,        # EWMA variance estimator for standardization
    delta: float = 0.25,            # target shift size in σ units (0.25–0.5 for subtle drift)
    h: float = 5.0,                 # decision threshold; larger -> fewer false alarms
    warmup: int = 100,              # burn-in for stable stats
    clip_z: float = 6.0             # winsorize to reduce heavy-tail spikes
) -> Dict[str, np.ndarray]:
    """
    Online subtle-drift detector using EWMA baseline + one-sided CUSUM on standardized residuals.

    Returns
    -------
    dict with arrays over time:
      'alarms'  : indices where drift flagged
      'mu'      : slow EWMA baseline
      'sigma'   : EWMA std estimate
      'S_plus'  : upward CUSUM path
      'S_minus' : downward CUSUM path
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return {"alarms": np.array([]), "mu": np.array([]), "sigma": np.array([]),
                "S_plus": np.array([]), "S_minus": np.array([])}

    mu = np.zeros(n)
    sig = np.zeros(n)
    S_plus = np.zeros(n)
    S_minus = np.zeros(n)

    # initialize with first value
    mu_t = x[0]
    var_t = 1e-6
    k = delta / 2.0  # standard CUSUM reference value for best sensitivity to shift δ

    alarms: List[int] = []

    for t in range(n):
        # update slow baseline (tracks very slow background changes)
        mu_t = (1 - alpha_baseline) * mu_t + alpha_baseline * x[t]
        mu[t] = mu_t

        # standardized residual using EWMA variance
        r = x[t] - mu_t
        var_t = (1 - alpha_var) * var_t + alpha_var * (r * r)
        sigma_t = float(np.sqrt(max(var_t, 1e-12)))
        sig[t] = sigma_t

        z = r / (sigma_t + 1e-12)
        # winsorize to protect against single spikes
        if clip_z is not None:
            z = float(np.clip(z, -clip_z, clip_z))

        # one-sided CUSUM updates (in z-units)
        S_plus_t = max(0.0, (S_plus[t-1] if t else 0.0) + z - k)
        S_minus_t = max(0.0, (S_minus[t-1] if t else 0.0) - z - k)
        S_plus[t] = S_plus_t
        S_minus[t] = S_minus_t

        if t >= warmup and (S_plus_t > h or S_minus_t > h):
            alarms.append(t)
            # reset statistics for quick re-arm
            S_plus[t] = 0.0
            S_minus[t] = 0.0

    return {
        "alarms": np.array(alarms, dtype=int),
        "mu": mu,
        "sigma": sig,
        "S_plus": S_plus,
        "S_minus": S_minus,
    }
