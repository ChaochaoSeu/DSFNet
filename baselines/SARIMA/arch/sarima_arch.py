"""BasicTS adapter for fixed or train-estimated SARIMA, using minus MA terms."""

import json
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn

from .innovations import SeasonalFilter, whiten
from .estimation import fit_series


class SARIMA(nn.Module):
    """One independent model per node and target channel of a T,N,D array.

    ``forecast_origin`` indexes the full, raw data.dat timeline. The companion
    dataset/runner supply it, so the usual 12-step input does not discard weekly
    history. Only observations strictly before each origin update the filter.
    Use on CPU; coefficients are estimated conditionally or fixed, then frozen.
    The MA polynomial is (1-theta B)(1-Theta B**s).
    """

    def __init__(self, dataset_name, target_features, output_len, seasonal_period,
                 train_val_test_ratio=(0.6, 0.2, 0.2), order=(1, 0, 1),
                 seasonal_order=(0, 1, 1), phi=0.98, theta=0.7, seasonal_theta=0.88,
                 exog_features=(), known_future_features=(), data_root='datasets',
                 parameter_mode='fixed', fit_maxiter=100, ma_convention='minus',
                 steps_per_day=None):
        super().__init__()
        if parameter_mode not in ('fixed', 'learned'):
            raise ValueError("parameter_mode must be 'fixed' or 'learned'.")
        if ma_convention != 'minus':
            raise ValueError("This implementation uses ma_convention='minus'.")
        if int(fit_maxiter) != fit_maxiter or fit_maxiter < 1:
            raise ValueError('fit_maxiter must be a positive integer.')
        self.parameter_mode = parameter_mode
        self.fit_maxiter = int(fit_maxiter)
        if tuple(order) != (1, 0, 1) or tuple(seasonal_order) != (0, 1, 1):
            raise ValueError('This fixed baseline implements only (1,0,1)(0,1,1)[s].')
        if int(seasonal_period) != seasonal_period or seasonal_period < 2:
            raise ValueError('seasonal_period must be an integer >= 2.')
        if steps_per_day is not None:
            if int(steps_per_day) != steps_per_day or steps_per_day < 2 or seasonal_period % steps_per_day:
                raise ValueError('steps_per_day must be an integer dividing seasonal_period.')
            steps_per_day = int(steps_per_day)
        self.steps_per_day = steps_per_day
        if not all(abs(v) < 1 for v in (phi, theta, seasonal_theta)):
            raise ValueError('AR and MA coefficients must have absolute value < 1.')
        if output_len < 1:
            raise ValueError('output_len must be positive.')
        ratios = tuple(train_val_test_ratio)
        if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1.0):
            raise ValueError('Supply three positive split ratios summing to 1.')

        folder = Path(data_root) / dataset_name
        with (folder / 'desc.json').open(encoding='utf-8') as stream:
            self.description = json.load(stream)
        self.shape = tuple(self.description['shape'])
        self.data_path = folder / 'data.dat'
        self.target_features = list(target_features)
        self.exog_features = list(exog_features)
        self.known_future_features = list(known_future_features)
        used = self.target_features + self.exog_features
        if (not self.target_features or len(set(used)) != len(used)
                or min(used) < 0 or max(used) >= self.shape[-1]):
            raise ValueError('Target/exogenous channels must be distinct valid raw feature indices.')
        if not set(self.known_future_features).issubset(self.exog_features):
            raise ValueError('known_future_features must be a subset of exog_features.')
        self.output_len = int(output_len)
        self.seasonal_period = int(seasonal_period)
        self.train_end = self.shape[0] - int(self.shape[0]*ratios[1]) - int(self.shape[0]*ratios[2])
        if self.train_end <= self.seasonal_period + 1:
            raise ValueError('Training split must contain more than one seasonal period + 1 step.')

        n, e, k = self.shape[1], len(self.exog_features), len(self.target_features)
        # Fixed mode shares three constants; learned mode estimates three
        # coefficients independently for every node and target channel.
        coefficients = torch.tensor([phi, theta, seasonal_theta], dtype=torch.float64)
        if parameter_mode == 'learned':
            coefficients = coefficients.expand(n, k, 3).clone()
        self.coefficients = nn.Parameter(coefficients, requires_grad=False)
        self.beta = nn.Parameter(torch.zeros(n, e, k, dtype=torch.float64), requires_grad=False)
        self.register_buffer('exog_mean', torch.zeros(n, e, dtype=torch.float64))
        self.register_buffer('exog_std', torch.ones(n, e, dtype=torch.float64))
        self.register_buffer('fitted', torch.tensor(False))
        self.register_buffer('ma_sign', torch.tensor(-1, dtype=torch.int8))
        self.register_buffer('parameter_mode_code', torch.tensor(
            int(parameter_mode == 'learned'), dtype=torch.int8))
        if steps_per_day is not None:
            self.register_buffer('operating_steps_per_day', torch.tensor(steps_per_day))
        self._data = None
        self._filter = None

    @property
    def data(self):
        if self._data is None:
            self._data = np.memmap(self.data_path, dtype='float32', mode='r', shape=self.shape)
        return self._data

    @torch.no_grad()
    def fit(self):
        """Fix or estimate ARIMA coefficients, using the training split only.

        Whiten y and X with the SAME candidate seasonal ARIMA filter, then
        solve least squares per series. No intercept: seasonal differencing removes
        constants. Static/weekly repeating and collinear columns are handled
        by the minimum-norm least-squares solution.
        """
        if self.coefficients.device.type != 'cpu':
            raise ValueError('SARIMA/SARIMAX are CPU baselines; set GPU_NUM=0.')
        if bool(self.fitted):
            return self
        unconverged = 0
        if self.exog_features or self.parameter_mode == 'learned':
            for node in range(self.shape[1]):
                train = np.asarray(self.data[:self.train_end, node, :], dtype=np.float64)
                x = train[:, self.exog_features]
                y = train[:, self.target_features]
                if not (np.isfinite(x).all() and np.isfinite(y).all()):
                    raise ValueError('Training targets/exogenous variables must be finite.')
                mean, std = x.mean(axis=0), x.std(axis=0)
                std[std < 1e-8] = 1.0
                x = (x-mean)/std
                if self.parameter_mode == 'learned':
                    for target in range(len(self.target_features)):
                        coefficients, beta, success = fit_series(
                            y[:, target], x, self.seasonal_period,
                            self.coefficients[node, target].numpy(), self.fit_maxiter,
                            steps_per_day=self.steps_per_day)
                        self.coefficients[node, target].copy_(torch.from_numpy(coefficients))
                        self.beta[node, :, target].copy_(torch.from_numpy(beta))
                        unconverged += not success
                else:
                    phi, theta, seasonal_theta = self.coefficients.tolist()
                    xw = whiten(x, self.seasonal_period, phi, theta, seasonal_theta,
                                steps_per_day=self.steps_per_day)
                    yw = whiten(y, self.seasonal_period, phi, theta, seasonal_theta,
                                steps_per_day=self.steps_per_day)
                    beta = np.linalg.lstsq(xw[self.seasonal_period:], yw[self.seasonal_period:], rcond=1e-8)[0]
                    self.beta[node].copy_(torch.from_numpy(beta))
                self.exog_mean[node].copy_(torch.from_numpy(mean))
                self.exog_std[node].copy_(torch.from_numpy(std))
        if unconverged:
            warnings.warn(f'{unconverged} SARIMA series did not converge within the optimizer limits; '
                          'using their best finite estimates. Consider increasing fit_maxiter.',
                          RuntimeWarning, stacklevel=2)
        self.fitted.fill_(True)
        self._filter = None
        return self

    def _regression(self, values):
        x = np.asarray(values[..., self.exog_features], dtype=np.float64)
        x = (x-self.exog_mean.numpy())/self.exog_std.numpy()
        return np.einsum('...ne,nek->...nk', x, self.beta.numpy())

    def _advance(self, origin):
        if self._filter is None or self._filter.count > origin:
            self._filter = SeasonalFilter(self.seasonal_period,
                                          (self.shape[1], len(self.target_features)),
                                          *np.moveaxis(self.coefficients.numpy(), -1, 0),
                                          steps_per_day=self.steps_per_day)
        # Bound the temporary allocation even for long NJRC timelines.
        for start in range(self._filter.count, origin, 4096):
            block = self.data[start:min(start+4096, origin)]
            residuals = np.asarray(block[..., self.target_features], dtype=np.float64)
            if self.exog_features:
                residuals -= self._regression(block)
            if not np.isfinite(residuals).all():
                raise ValueError('Observed targets/exogenous variables must be finite.')
            for residual in residuals:
                self._filter.update(residual)

    def _future_regression(self, origin):
        # Unknown future covariates use the last observation. Only explicitly
        # declared, genuinely known future channels can read the future array.
        future = np.repeat(np.asarray(self.data[origin-1:origin]), self.output_len, axis=0)
        for feature in self.known_future_features:
            future[..., feature] = self.data[origin:origin+self.output_len, :, feature]
        return self._regression(future)

    @torch.no_grad()
    def forward(self, history_data=None, future_data=None, forecast_origin=None, **kwargs):
        """Return B,H,N,K predictions; future target values are never used.

        history_data/future_data follow BasicTS's interface. Raw long history
        is read by forecast_origin instead of re-filtering overlapping windows.
        """
        if forecast_origin is None:
            raise ValueError('Use SeasonalForecastingDataset and SeasonalARIMARunner to supply forecast_origin.')
        self.fit()
        origins = np.asarray(torch.as_tensor(forecast_origin).cpu(), dtype=np.int64).reshape(-1)
        if (not len(origins) or origins.min() <= self.seasonal_period
                or origins.max()+self.output_len > self.shape[0]):
            raise ValueError('Forecast origins need season + 1 past steps and a complete forecast horizon.')
        predictions = np.empty((len(origins), self.output_len, self.shape[1], len(self.target_features)), dtype=np.float32)
        for i in np.argsort(origins, kind='stable'):
            origin = int(origins[i])
            self._advance(origin)
            result = self._filter.forecast(self.output_len)
            if self.exog_features:
                result += self._future_regression(origin)
            if not np.isfinite(result).all():
                raise ValueError('Forecast or future exogenous variables contain non-finite values.')
            predictions[i] = result
        return torch.from_numpy(predictions)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name, expected in [('ma_sign', -1),
                               ('parameter_mode_code', int(self.parameter_mode == 'learned'))]:
            value = state_dict.get(prefix+name)
            if value is None or value.numel() != 1 or value.item() != expected:
                error_msgs.append('Incompatible SARIMA checkpoint: MA convention or parameter mode '
                                  'does not match. Train again with the current configuration.')
                break
        if self.steps_per_day is not None:
            value = state_dict.get(prefix+'operating_steps_per_day')
            if value is None or value.numel() != 1 or value.item() != self.steps_per_day:
                error_msgs.append('Incompatible operating-hours checkpoint; retrain with the MetroFlow configuration.')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        # Filter state is reconstructed causally using the loaded coefficients.
        self._filter = None
