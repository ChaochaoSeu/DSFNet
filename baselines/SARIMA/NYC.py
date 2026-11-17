from .arch import SARIMA
from .config import make_config

DATA_NAME = 'NYC'
TARGET_FEATURES = list(range(6))
PARAMETER_MODE = None  # None: inherit config.py; alternatively, use 'fixed' or 'learned'.
CFG = make_config(SARIMA, DATA_NAME, TARGET_FEATURES, parameter_mode=PARAMETER_MODE)
MODEL_ARCH, MODEL_PARAM = SARIMA, CFG.MODEL.PARAM
