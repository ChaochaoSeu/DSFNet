from .arch import SARIMA
from .config import make_config

Dataset = 'NJRC'
sampling_granularity = 15  # Available: 3, 5, 10, 15 minutes.
DATA_NAME = f'{Dataset}_{sampling_granularity}min'
TARGET_FEATURES = [0, 2, 3, 4, 5, 6, 9, 10, 11, 12]
PARAMETER_MODE = None  # None: inherit config.py; alternatively, use 'fixed' or 'learned'.
CFG = make_config(SARIMA, DATA_NAME, TARGET_FEATURES, parameter_mode=PARAMETER_MODE)
MODEL_ARCH, MODEL_PARAM = SARIMA, CFG.MODEL.PARAM
