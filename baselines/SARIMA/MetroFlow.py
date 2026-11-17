from .arch import SARIMA
from .metroflow_support import make_metroflow_config

DATA_NAME = 'MetroFlow'  # Use the original concatenated operating-hour data without filling or rebuilding the time axis.
SEASONAL_PERIOD = 102  # Daily period: 17 operating hours times 6 steps per hour = 102.
TARGET_FEATURES = list(range(8))
PARAMETER_MODE = "fixed"  # None: inherit config.py; alternatively, use 'fixed' or 'learned'.
CFG = make_metroflow_config(SARIMA, TARGET_FEATURES, parameter_mode=PARAMETER_MODE,
                            dataset_name=DATA_NAME, evaluation_mode='legacy',
                            legacy_seasonal_period=SEASONAL_PERIOD)
CFG.DESCRIPTION += '; concatenated operating-time daily SARIMA, s=102; no data completion'
MODEL_ARCH, MODEL_PARAM = SARIMA, CFG.MODEL.PARAM
