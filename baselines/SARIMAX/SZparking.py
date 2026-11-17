from baselines.SARIMA.config import make_config

from .arch import SARIMAX

Dataset = 'SZparking'
sampling_granularity = 10  # Available: 5, 10, 15 minutes.
DATA_NAME = f'{Dataset}_{sampling_granularity}min'
TARGET_FEATURES = list(range(10))
EXOG_FEATURES = list(range(12, 25))  # Weather, parking attributes, calendar.
KNOWN_FUTURE_FEATURES = [22, 23, 24]
PARAMETER_MODE = None  # None: inherit SARIMA/config.py; alternatively, use 'fixed' or 'learned'.
CFG = make_config(SARIMAX, DATA_NAME, TARGET_FEATURES, EXOG_FEATURES, KNOWN_FUTURE_FEATURES,
                  parameter_mode=PARAMETER_MODE)
MODEL_ARCH, MODEL_PARAM = SARIMAX, CFG.MODEL.PARAM
