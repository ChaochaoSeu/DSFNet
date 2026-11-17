# SARIMA / SARIMAX baselines

Inputs use the existing float32 `[T,N,D]` format in `datasets/<DATA_NAME>/data.dat`, together with `desc.json`. A separate univariate seasonal model is built for each node and target channel, producing `[B,H,N,K]` outputs. Configurations are located in `baselines/SARIMA/` and `baselines/SARIMAX/`.

## Parameter modes and seasonal periods

The default model is `SARIMA(1,0,1)(0,1,1)[s]` with `phi=0.98`, `theta=0.7`, and `seasonal_theta=0.88`, using the **minus-sign MA convention**. Both baselines support the following modes:

```python
# baselines/SARIMA/config.py: shared defaults for all ten dataset configurations
PARAMETER_MODE = 'fixed'  # 'fixed' or 'learned'
FIT_MAXITER = 100        # Maximum optimization iterations per univariate series
```

| Mode | phi / theta / seasonal_theta | SARIMAX exogenous coefficients beta |
| --- | --- | --- |
| `'fixed'` | Supplied values; defaults are `0.98 / 0.7 / 0.88` | Estimated on the training set only |
| `'learned'` | Estimated separately for each node and target channel on the training set; supplied values initialize the optimizer | Estimated jointly with the ARIMA coefficients on the training set |

Each dataset configuration also defines `PARAMETER_MODE`: `None` inherits the shared default; `'fixed'` or `'learned'` overrides it. Model orders and seasonal periods are unchanged across modes. Parameters are listed in `CFG.MODEL.PARAM` and may also be overridden through the corresponding `make_config` arguments.

For continuously sampled datasets, the weekly seasonal period `s` is computed from `frequency (minutes)` in the description:

| Sampling interval | Steps per day | Steps per week s |
| --- | --- | --- |
| 3 minutes | 480 | 3360 |
| 5 minutes | 288 | 2016 |
| 10 minutes | 144 | 1008 |
| 15 minutes | 96 | 672 |
| 30 minutes | 48 | 336 |

This assumes a continuous time axis at the specified interval. If nighttime or other periods were removed, restore a regular time axis or redefine the seasonal lag using the actual time index.

**MetroFlow uses a daily period on concatenated operating-hour slots.** Both configurations read the existing `datasets/MetroFlow` directly with `SEASONAL_PERIOD=102`, without filling data or rebuilding the time axis. All original evaluation windows are retained, and daily state resets are disabled. SARIMAX uses only weekday and workday/rest-day flags `[12,13]` as exogenous inputs. See [MetroFlow details](METROFLOW.md) for the configuration and evaluation protocol.

## Estimation and forecasting

SARIMA recurses over the original target variables. SARIMAX uses regression with SARIMA errors:

```text
y_t = X_t beta + r_t
(1 - phi B)(1 - B^s) r_t = (1 - theta B)(1 - Theta B^s) e_t
```

`theta=0.7` and `Theta=0.88` are coefficients in the minus-sign polynomials; do not negate them in the configuration. The expanded MA polynomial is `1 - theta B - Theta B^s + theta*Theta B^(s+1)`, with a positive multiplicative cross term.

The implementation uses conditional innovation initialization: the first week supplies initial levels, its differences and innovations are set to zero, and recursion follows. Future innovations are set to zero during multi-step forecasts, including the `+theta * Theta` multiplicative MA cross term. This conditional recursion may differ initially from Kalman maximum likelihood with exact diffuse initialization. It does not depend on statsmodels or search model orders.

SARIMAX estimates exogenous means and standard deviations on the training set only. It applies the same fixed ARIMA whitening filter to `y` and `X`, then solves conditional least squares for each node. There is no intercept because seasonal differencing removes constants. Static or exactly weekly repeating covariates are also removed by seasonal differencing; collinear columns use a minimum-norm solution, so coefficients need not be identifiable.

In `'learned'` mode, conditional least-squares optimization runs separately for each node and target channel. L-BFGS-B estimates the three ARIMA coefficients within `[-0.995, 0.995]`. For SARIMAX, each candidate ARIMA parameter set is paired with a least-squares fit of the whitened exogenous variables to minimize the training innovation sum of squares. Target variance scales the optimization objective only; predictions remain in original units. This is not exact Kalman maximum likelihood. Learned mode takes longer than fixed mode. If optimization fails to converge, warnings are summarized and the best finite estimate is retained; increase `FIT_MAXITER` if needed. Parameters remain frozen during validation and testing; only the filtering state is updated with observed history.

For ordinary datasets, `INPUT_LEN`, `OUTPUT_LEN`, and split ratios follow the description. Validation and test origins match the existing default `overlap=False` reader. The specialized dataset supplies an absolute `forecast_origin`, allowing the model to access all observed history before that origin, so a 12-step input does not discard weekly seasonality. Previously observed validation or test values may update the state, as in rolling-origin forecasting. Future targets are never read during a multi-step forecast from a given origin.

The model retains only `s+1` residual and innovation steps. Sequential forecasts reuse state; returning to an earlier origin restarts filtering. Computation runs on the CPU, and metrics use original units without an external scaler. All target and exogenous inputs must be finite. Zeros are retained, and metrics follow `NULL_VAL` in the dataset description.

## Data channels

All indices refer to the original feature axis `D`. Target channels match the corresponding existing baselines.

| Dataset configuration | Target channels | SARIMAX exogenous channels | Known future channels |
| --- | --- | --- | --- |
| PEMS03 | `[0]` | `[1,2,3]` | `[1,2,3]` |
| MetroFlow | `0-7` | `[12,13]` | `[12,13]` |
| NYC | `0-5` | `[6,7,8]` | `[6,7,8]` |
| NJRC | `[0,2,3,4,5,6,9,10,11,12]` | `13-27` | `[25,26,27]` |
| SZparking | `0-9` | `12-24` | `[22,23,24]` |

SARIMA does not use exogenous variables. By default, only calendar variables such as time slot, weekday, and workday/holiday flags are known in the future for SARIMAX. Unknown future covariates, including weather and events, use the last observation before the forecast origin. Add original feature indices to `KNOWN_FUTURE_FEATURES` only when the corresponding exogenous forecasts are available at prediction time. Do not supply retrospectively observed future weather, events, or target values as known inputs.

NJRC and SZparking default to `sampling_granularity=10`. Select another locally available resolution as needed; the weekly period is always read from the selected dataset description.

## Running

Use the project environment with PyTorch, NumPy, SciPy, EasyDict, EasyTorch, and the other runtime dependencies:

```powershell
python experiments/train.py -c baselines/SARIMA/PEMS03.py
python experiments/train.py -c baselines/SARIMAX/PEMS03.py
python experiments/train.py -c baselines/SARIMAX/MetroFlow.py
```

The current MetroFlow configuration reads the existing data directly and requires no preparation or filling commands. Earlier corrected variants and tools are archived and are not used by these configurations.

Training runs once: parameters are supplied or estimated on the complete training set according to the selected mode, and a window at the end of training records training metrics. Full validation, testing, and checkpoint saving follow. SGD with `lr=0` serves only the framework's checkpoint interface; all model parameters are frozen, and the runner disables backpropagation.

```powershell
python experiments/evaluate.py -cfg baselines/SARIMAX/PEMS03.py -ckpt PATH_TO_CHECKPOINT -d cpu
python -m unittest tests.test_seasonal_arima tests.test_metroflow_sarima -v
```

Checkpoints include ARIMA and regression coefficients, exogenous scaling statistics, and markers for the MA convention and parameter mode. Loading reconstructs the historical filtering state without refitting. Fixed, learned, and minus-sign variants use distinct save directories, such as `checkpoints/SARIMA/PEMS03_fixed_ma_minus_s2016_12_12/`. Checkpoints from the earlier plus-sign convention are incompatible and require retraining; incompatible loads are rejected. Metrics and predictions use the BasicTS files `test_metrics.json` and `test_results.npz`.

Formula and interface references: [statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) and [SciPy lfilter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html).
