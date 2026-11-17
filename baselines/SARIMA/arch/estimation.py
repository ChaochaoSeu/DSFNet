"""Train-only conditional least squares for one seasonal ARIMA series."""

import numpy as np
from scipy.optimize import minimize

from .innovations import whiten


def fit_series(target, exog, season, initial, maxiter, steps_per_day=None):
    """Profile out beta while estimating phi/theta/Theta in the minus convention.

    exog is already standardized on the training split. Whiten target and
    regressors together so each candidate has its own conditional GLS beta.
    Scaling the objective by target variance improves numerical conditioning;
    the returned beta and all predictions remain in original target units.
    """
    target = np.asarray(target, dtype=np.float64)
    scale = max(float(target.std()), 1e-8)
    values = np.column_stack((target/scale, exog))

    def profile(parameters):
        filtered = whiten(values, season, *parameters, steps_per_day=steps_per_day)[season:]
        y, x = filtered[:, 0], filtered[:, 1:]
        beta = np.linalg.lstsq(x, y, rcond=1e-8)[0]
        innovations = y - x @ beta
        return float(np.mean(innovations**2)), beta*scale

    initial = np.clip(np.asarray(initial, dtype=np.float64), -0.995, 0.995)
    initial_loss, initial_beta = profile(initial)
    result = minimize(lambda parameters: profile(parameters)[0], initial,
                      method='L-BFGS-B', bounds=[(-0.995, 0.995)]*3,
                      options={'maxiter': maxiter, 'ftol': 1e-9})
    if not np.isfinite(result.fun) or result.fun > initial_loss:
        return initial, initial_beta, False
    _, beta = profile(result.x)
    return result.x, beta, bool(result.success)
