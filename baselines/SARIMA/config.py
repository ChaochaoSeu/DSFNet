"""Shared BasicTS configuration; dataset-specific channel choices stay explicit."""

import json
import os
from pathlib import Path

from easydict import EasyDict

from basicts.metrics import masked_mae, masked_mape, masked_rmse

from .dataset import SeasonalForecastingDataset
from .runner import SeasonalARIMARunner


# Shared defaults; dataset configurations may override PARAMETER_MODE.
PARAMETER_MODE = 'fixed'  # 'fixed': use supplied coefficients; 'learned': estimate on the training set only.
FIT_MAXITER = 100  # Maximum optimization iterations per node and target channel in learned mode.


def weekly_period(frequency_minutes):
    """Number of uniformly sampled steps in seven 24-hour days."""
    frequency_minutes = float(frequency_minutes)
    if frequency_minutes <= 0:
        raise ValueError('Sampling frequency must be positive.')
    steps = 7*24*60/frequency_minutes
    if abs(steps-round(steps)) > 1e-8:
        raise ValueError('Sampling frequency must divide a calendar week into whole steps.')
    return int(round(steps))


def make_config(model_arch, dataset_name, target_features, exog_features=(),
                known_future_features=(), phi=0.98, theta=0.7,
                seasonal_theta=0.88, data_root='datasets', parameter_mode=None,
                fit_maxiter=None):
    parameter_mode = PARAMETER_MODE if parameter_mode is None else parameter_mode
    fit_maxiter = FIT_MAXITER if fit_maxiter is None else fit_maxiter
    if parameter_mode not in ('fixed', 'learned'):
        raise ValueError("parameter_mode must be 'fixed' or 'learned'.")
    with (Path(data_root) / dataset_name / 'desc.json').open(encoding='utf-8') as stream:
        description = json.load(stream)
    regular = description['regular_settings']
    season = weekly_period(description['frequency (minutes)'])
    input_len, output_len = regular['INPUT_LEN'], regular['OUTPUT_LEN']
    selected = list(target_features) + list(exog_features)

    cfg = EasyDict()
    cfg.DESCRIPTION = (f'SARIMA(1,0,1)(0,1,1)[s]; {parameter_mode} coefficients; '
                       'minus MA convention; train-only conditional estimation')
    cfg.GPU_NUM = 0
    cfg.RUNNER = SeasonalARIMARunner
    cfg.ENV = EasyDict(SEED=0)
    cfg.DATASET = EasyDict(NAME=dataset_name, TYPE=SeasonalForecastingDataset,
                           PARAM=EasyDict(dataset_name=dataset_name,
                                          train_val_test_ratio=regular['TRAIN_VAL_TEST_RATIO'],
                                          input_len=input_len, output_len=output_len,
                                          data_root=data_root))
    # Work in raw units. The model standardizes exogenous variables internally
    # using training data only; metric outputs already have original units.
    cfg.MODEL = EasyDict(NAME=model_arch.__name__, ARCH=model_arch,
                         PARAM=dict(dataset_name=dataset_name,
                                    target_features=list(target_features),
                                    exog_features=list(exog_features),
                                    known_future_features=list(known_future_features),
                                    output_len=output_len, seasonal_period=season,
                                    order=(1, 0, 1), seasonal_order=(0, 1, 1),
                                    phi=phi, theta=theta, seasonal_theta=seasonal_theta,
                                    ma_convention='minus', parameter_mode=parameter_mode,
                                    fit_maxiter=fit_maxiter,
                                    train_val_test_ratio=regular['TRAIN_VAL_TEST_RATIO'],
                                    data_root=data_root),
                         FORWARD_FEATURES=selected, TARGET_FEATURES=list(target_features),
                         EXOG_FEATURES=list(exog_features),
                         KNOWN_FUTURE_FEATURES=list(known_future_features))
    cfg.METRICS = EasyDict(FUNCS=EasyDict(MAE=masked_mae, RMSE=masked_rmse, MAPE=masked_mape),
                           TARGET='MAE', NULL_VAL=regular['NULL_VAL'])
    cfg.TRAIN = EasyDict(NUM_EPOCHS=1, LOSS=masked_mae,
                         CKPT_SAVE_DIR=os.path.join('checkpoints', model_arch.__name__,
                                                   f'{dataset_name}_{parameter_mode}_ma_minus_s{season}_{input_len}_{output_len}'),
                         # BasicTS's checkpoint lifecycle requires an optimizer.
                         # All model parameters are frozen and backward is disabled.
                         OPTIM=EasyDict(TYPE='SGD', PARAM=dict(lr=0.0)),
                         DATA=EasyDict(BATCH_SIZE=1, SHUFFLE=False, NUM_WORKERS=0))
    cfg.VAL = EasyDict(INTERVAL=1, DATA=EasyDict(BATCH_SIZE=64, SHUFFLE=False, NUM_WORKERS=0))
    cfg.TEST = EasyDict(INTERVAL=1, DATA=EasyDict(BATCH_SIZE=64, SHUFFLE=False, NUM_WORKERS=0))
    cfg.EVAL = EasyDict(HORIZONS=list(range(1, output_len+1)), USE_GPU=False, SAVE_RESULTS=True)
    return cfg
