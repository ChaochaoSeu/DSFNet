from .arch import SARIMA
from .config import make_config

Dataset = 'SZparking'
sampling_granularity = 15  # Available: 5, 10, 15 minutes.
DATA_NAME = f'{Dataset}_{sampling_granularity}min'
TARGET_FEATURES = list(range(10))
PARAMETER_MODE = None  # None: inherit config.py; alternatively, use 'fixed' or 'learned'.
CFG = make_config(SARIMA, DATA_NAME, TARGET_FEATURES, parameter_mode=PARAMETER_MODE)
MODEL_ARCH, MODEL_PARAM = SARIMA, CFG.MODEL.PARAM
