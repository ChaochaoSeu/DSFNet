"""Conditional innovations for (1,0,1)(0,1,1)[s], without dense state matrices."""

import numpy as np
from scipy.signal import lfilter


def whiten(values, season, phi, theta, seasonal_theta, steps_per_day=None):
    """Apply (1-phi B)(1-B**s)/((1-theta B)(1-Theta B**s)).

    The first season supplies initial levels; its differences and innovations
    are conditioned to zero. Time is axis 0. This is a causal linear filter.
    """
    values = np.asarray(values, dtype=np.float64)
    differences = np.zeros_like(values)
    differences[season:] = values[season:] - values[:-season]
    if steps_per_day is None:
        innovations = lfilter([1.0, -phi], [1.0, -theta], differences, axis=0)
    else:
        # MetroFlow's operating-time model has no short-term AR/MA transition
        # from closing time to next day's opening. Weekly memory is retained.
        days = (len(values)+steps_per_day-1)//steps_per_day
        padded = np.zeros((days*steps_per_day, *values.shape[1:]), dtype=np.float64)
        padded[:len(values)] = differences
        by_day = padded.reshape(days, steps_per_day, *values.shape[1:])
        innovations = lfilter([1.0, -phi], [1.0, -theta], by_day, axis=1)
        innovations = innovations.reshape(padded.shape)[:len(values)]
    # Invert the seasonal MA polynomial in blocks, in O(T), not O(T * s).
    for start in range(season, len(values), season):
        stop = min(start + season, len(values))
        innovations[start:stop] += seasonal_theta * innovations[start-season:stop-season]
    return innovations


class SeasonalFilter:
    """Keep only s+1 past residuals/innovations for independent N x K series.

    (1-phi B)(1-B**s) r_t = (1-theta B)(1-Theta B**s) e_t.
    The multiplicative MA cross term +theta*Theta is retained. Coefficients
    may be scalars or arrays broadcastable to shape for per-series estimates.
    Forecasts condition all future innovations to zero and never update state.
    """

    def __init__(self, season, shape, phi=0.98, theta=0.7, seasonal_theta=0.88,
                 steps_per_day=None):
        if steps_per_day is not None and (steps_per_day < 2 or season % steps_per_day):
            raise ValueError('Season must be a multiple of steps_per_day.')
        self.season = season
        self.steps_per_day = steps_per_day
        self.phi = phi
        self.theta = theta
        self.seasonal_theta = seasonal_theta
        self.residuals = np.zeros((season + 1, *shape), dtype=np.float64)
        self.innovations = np.zeros_like(self.residuals)
        self.count = 0
        self.previous_difference = np.zeros(shape, dtype=np.float64)

    def update(self, residual):
        t, s = self.count, self.season
        if t < s:
            difference = np.zeros_like(residual)
            innovation = np.zeros_like(residual)
        else:
            difference = residual - self.residuals[(t-s) % (s+1)]
            opening = self.steps_per_day is not None and t % self.steps_per_day == 0
            innovation = (
                difference - (0.0 if opening else self.phi * self.previous_difference)
                + (0.0 if opening else self.theta * self.innovations[(t-1) % (s+1)])
                + self.seasonal_theta * self.innovations[(t-s) % (s+1)]
                - (0.0 if opening else self.theta * self.seasonal_theta
                   * self.innovations[(t-s-1) % (s+1)])
            )
        self.residuals[t % (s+1)] = residual
        self.innovations[t % (s+1)] = innovation
        self.previous_difference = difference
        self.count += 1

    def forecast(self, horizon):
        if self.count <= self.season:
            raise ValueError('SARIMA requires at least season + 1 historical observations.')
        result = np.empty((horizon, *self.previous_difference.shape), dtype=np.float64)
        difference = self.previous_difference.copy()
        s, origin = self.season, self.count
        for h in range(horizon):
            t = origin + h
            opening = self.steps_per_day is not None and t % self.steps_per_day == 0
            e1 = self.innovations[(t-1) % (s+1)] if h == 0 else 0.0
            es = self.innovations[(t-s) % (s+1)] if t-s < origin else 0.0
            es1 = self.innovations[(t-s-1) % (s+1)] if t-s-1 < origin else 0.0
            difference = ((0.0 if opening else self.phi * difference - self.theta * e1)
                          - self.seasonal_theta * es
                          + (0.0 if opening else self.theta * self.seasonal_theta * es1))
            seasonal_level = (self.residuals[(t-s) % (s+1)] if t-s < origin
                              else result[t-s-origin])
            result[h] = seasonal_level + difference
        return result
