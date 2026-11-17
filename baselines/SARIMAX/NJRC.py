from baselines.SARIMA.config import make_config

from .arch import SARIMAX

Dataset = 'NJRC'
sampling_granularity = 10  # Available: 3, 5, 10, 15 minutes.
DATA_NAME = f'{Dataset}_{sampling_granularity}min'
TARGET_FEATURES = [0, 2, 3, 4, 5, 6, 9, 10, 11, 12]
EXOG_FEATURES = list(range(13, 28))  # Weather, events, road attributes, calendar.
KNOWN_FUTURE_FEATURES = [25, 26, 27]
PARAMETER_MODE = None  # None: inherit SARIMA/config.py; alternatively, use 'fixed' or 'learned'.
CFG = make_config(SARIMAX, DATA_NAME, TARGET_FEATURES, EXOG_FEATURES, KNOWN_FUTURE_FEATURES,
                  parameter_mode=PARAMETER_MODE)
MODEL_ARCH, MODEL_PARAM = SARIMAX, CFG.MODEL.PARAM
