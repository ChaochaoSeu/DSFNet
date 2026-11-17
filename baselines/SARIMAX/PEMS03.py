from baselines.SARIMA.config import make_config

from .arch import SARIMAX

DATA_NAME = 'PEMS03'
TARGET_FEATURES = [0]
EXOG_FEATURES = [1, 2, 3]  # Time of day, day of week, workday.
KNOWN_FUTURE_FEATURES = [1, 2, 3]
PARAMETER_MODE = None  # None: inherit SARIMA/config.py; alternatively, use 'fixed' or 'learned'.
CFG = make_config(SARIMAX, DATA_NAME, TARGET_FEATURES, EXOG_FEATURES, KNOWN_FUTURE_FEATURES,
                  parameter_mode=PARAMETER_MODE)
MODEL_ARCH, MODEL_PARAM = SARIMAX, CFG.MODEL.PARAM
