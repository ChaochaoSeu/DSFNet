# MetroFlow: daily SARIMA / SARIMAX over operating-hour slots

Both configurations model a concatenated sequence of operating-hour slots and read the existing `datasets/MetroFlow/data.dat` directly. They do not fill data, replace anomalous days, insert nighttime zeros, rebuild the time axis, or read the `MetroFlow_SARIMA` variant.

| Model | Time handling | Exogenous variables | Role |
| --- | --- | --- | --- |
| SARIMA | Concatenated operating-hour slots; daily period `s=102` | None | Univariate seasonal baseline |
| SARIMAX | Concatenated operating-hour slots; daily period `s=102` | Weekday and workday/rest-day flags | Calendar-augmented baseline |

## Configuration

`baselines/SARIMA/MetroFlow.py`:

```python
DATA_NAME = 'MetroFlow'
SEASONAL_PERIOD = 102
TARGET_FEATURES = list(range(8))
PARAMETER_MODE = 'fixed'  # 'learned': fit on training data only; None: inherit shared defaults
```

`baselines/SARIMAX/MetroFlow.py`:

```python
DATA_NAME = 'MetroFlow'
SEASONAL_PERIOD = 102
TARGET_FEATURES = list(range(8))
EXOG_FEATURES = [12, 13]
KNOWN_FUTURE_FEATURES = [12, 13]
PARAMETER_MODE = None  # Alternatively, select 'fixed' or 'learned'.
```

Each day retains 17 operating hours with 6 steps per hour, giving a daily period of `17*6=102`. Both models retain the order `SARIMA(1,0,1)(0,1,1)[102]` and the minus-sign MA convention. Fixed mode uses `phi=0.98`, `theta=0.7`, and `seasonal_theta=0.88`. Learned mode estimates coefficients separately for each node and target on the training set only. SARIMAX estimates exogenous regression coefficients on the training set in both modes.

Original channel 12 is `day of week`, and channel 13 is `is workday`. The latter describes workdays versus rest days; it is not a separate statutory-holiday label, and no additional label is fabricated. Weather channels `8..10` and time-slot channel `11` are excluded from the current SARIMAX regression. Weekday and workday flags are treated as known calendar inputs at forecast time.

## Evaluation aligned with the original data

- The data shape remains `[7044,302,14]`, with eight target channels `0..7`.
- Split ratios remain `60%/20%/20%`: 4228 training steps, 1408 validation steps, and 1408 test steps.
- Input and forecast lengths are both 12; validation and testing each contain 1385 sliding windows.
- Validation forecast origins are `4240..5624`, and test origins are `5648..7032`, inclusive.
- The ordinary `SeasonalForecastingDataset` retains the original cross-day windows without daily short-term state resets.
- Recursion continues over the operating-slot index. A 12-step forecast covers the next 12 retained operating slots; a window crossing nighttime does not represent two continuous clock hours.
- MAE, RMSE, and MAPE use original units with `NULL_VAL=0` and the existing BasicTS masking and aggregation functions.
- Filtering state may use observed history before the forecast origin, but not future targets for that forecast.

This operating-slot configuration preserves the original files and calendar information. It does not repair the previously identified timestamp-mapping issues in the source data. Internally, `make_metroflow_config` uses its `legacy` path to reuse the original sliding windows. The two runnable configurations no longer expose a switch to corrected data, avoiding accidental evaluation on a different dataset.

## Running

No data preparation commands are required:

```powershell
python experiments/train.py -c baselines/SARIMA/MetroFlow.py
python experiments/train.py -c baselines/SARIMAX/MetroFlow.py
python -m unittest tests.test_seasonal_arima tests.test_metroflow_sarima -v
```

Current fixed-mode checkpoint paths include the daily period `s102`:

```text
checkpoints/SARIMA/MetroFlow_legacy_fixed_ma_minus_s102_12_12/
checkpoints/SARIMAX/MetroFlow_legacy_fixed_ma_minus_s102_12_12/
```

The framework also creates a configuration-hash subdirectory. Rerun training after changing `PARAMETER_MODE`, the period, or exogenous channels. Do not interchange checkpoints trained with different periods or exogenous channel sets.

Earlier `MetroFlow_SARIMA` data, filling utilities, and operating-hour expansion code are archived and are not used by these configurations. This baseline setup does not modify, overwrite, or delete data files, or alter configurations for other models or datasets.
