from baselines.SARIMA.metroflow_support import make_metroflow_config

from .arch import SARIMAX

DATA_NAME = 'MetroFlow'  # Use the original concatenated operating-hour data without filling or rebuilding the time axis.
SEASONAL_PERIOD = 102  # Daily period: 102 retained operating slots per day.
TARGET_FEATURES = list(range(8))
EXOG_FEATURES = [12, 13]  # Weekday and workday flags; the original data have no separate statutory-holiday label.
KNOWN_FUTURE_FEATURES = [12, 13]
PARAMETER_MODE = None  # None: inherit SARIMA/config.py; alternatively, use 'fixed' or 'learned'.
CFG = make_metroflow_config(SARIMAX, TARGET_FEATURES, EXOG_FEATURES, KNOWN_FUTURE_FEATURES,
                            parameter_mode=PARAMETER_MODE, dataset_name=DATA_NAME,
                            evaluation_mode='legacy', legacy_seasonal_period=SEASONAL_PERIOD)
CFG.DESCRIPTION += '; concatenated operating-time daily SARIMAX, s=102; calendar covariates; no data completion'
MODEL_ARCH, MODEL_PARAM = SARIMAX, CFG.MODEL.PARAM
