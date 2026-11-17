from baselines.SARIMA.config import make_config

from .arch import SARIMAX

DATA_NAME = 'NYC'
TARGET_FEATURES = list(range(6))
EXOG_FEATURES = [6, 7, 8]
KNOWN_FUTURE_FEATURES = [6, 7, 8]
PARAMETER_MODE = None  # None: inherit SARIMA/config.py; alternatively, use 'fixed' or 'learned'.
CFG = make_config(SARIMAX, DATA_NAME, TARGET_FEATURES, EXOG_FEATURES, KNOWN_FUTURE_FEATURES,
                  parameter_mode=PARAMETER_MODE)
MODEL_ARCH, MODEL_PARAM = SARIMAX, CFG.MODEL.PARAM
