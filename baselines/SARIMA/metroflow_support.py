"""Legacy-compatible and corrected evaluation for the two MetroFlow baselines."""

import json
from pathlib import Path

import numpy as np

from .config import make_config
from .dataset import SeasonalForecastingDataset


class MetroFlowForecastingDataset(SeasonalForecastingDataset):
    """Keep both history and target windows inside one real operating day."""

    def __init__(self, dataset_name, train_val_test_ratio, mode, input_len,
                 output_len, data_root='datasets', logger=None):
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len,
                         output_len, data_root=data_root, logger=logger)
        metadata = self.description['operating_hours']
        steps = metadata['steps_per_day']
        timestamps = np.load(Path(data_root)/dataset_name/metadata['timestamps_file'])
        if len(timestamps) != len(self.data) or len(timestamps) % steps:
            raise ValueError('MetroFlow timestamps must cover complete operating days.')
        dates = timestamps.astype('datetime64[D]')
        within = dates.reshape(-1, steps)
        deltas = np.diff(timestamps.reshape(-1, steps), axis=1)
        if not np.all(within == within[:, :1]) or not np.all(deltas == np.timedelta64(10, 'm')):
            raise ValueError('MetroFlow timestamps are not regular ten-minute operating days.')
        if not np.all(np.diff(within[:, 0]) == np.timedelta64(1, 'D')):
            raise ValueError('MetroFlow operating dates must be consecutive for a fixed weekly lag.')
        valid = int(len(self.data)*train_val_test_ratio[1])
        test = int(len(self.data)*train_val_test_ratio[2])
        train = len(self.data)-valid-test
        start, stop = {'train': (0, train), 'valid': (train, train+valid),
                       'test': (train+valid, len(self.data))}[mode]
        origins = np.arange(start+input_len, stop-output_len+1, dtype=np.int64)
        origins = origins[dates[origins-input_len] == dates[origins+output_len-1]]
        if not len(origins):
            raise ValueError('No complete within-day MetroFlow windows exist in this split.')
        self.origins = origins[-1:] if mode == 'train' else origins

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        origin = int(self.origins[index])
        return {'inputs': self.data[origin-self.input_len:origin].copy(),
                'target': self.data[origin:origin+self.output_len].copy(),
                'forecast_origin': origin}


def make_metroflow_config(model_arch, target_features, exog_features=(),
                          known_future_features=(), parameter_mode=None,
                          dataset_name=None, data_root='datasets', evaluation_mode='legacy',
                          legacy_seasonal_period=None, **kwargs):
    """Choose evaluation data/windows independently of ARIMA parameter fitting.

    legacy: original data, all BasicTS windows, no daily resets. Its default
    lag follows the original description (1008 at ten-minute frequency).
    corrected: isolated real-time data, within-day windows and daily resets.
    """
    if evaluation_mode not in ('legacy', 'corrected'):
        raise ValueError("evaluation_mode must be 'legacy' or 'corrected'.")
    if dataset_name is None:
        dataset_name = 'MetroFlow' if evaluation_mode == 'legacy' else 'MetroFlow_SARIMA'
    folder = Path(data_root)/dataset_name
    if not (folder/'desc.json').is_file():
        if evaluation_mode == 'legacy':
            raise FileNotFoundError(f'Original MetroFlow data description not found: {folder/"desc.json"}')
        raise FileNotFoundError('Prepare the isolated MetroFlow data first: '
                                'python -m baselines.SARIMA.prepare_metroflow')
    with (folder/'desc.json').open(encoding='utf-8') as stream:
        desc = json.load(stream)
    if evaluation_mode == 'legacy' and 'operating_hours' in desc:
        raise ValueError('legacy evaluation requires the original data, not the corrected operating-hours copy.')
    if evaluation_mode == 'corrected' and 'operating_hours' not in desc:
        raise ValueError('corrected evaluation requires prepared data with real operating-hours metadata.')
    if evaluation_mode == 'legacy' and legacy_seasonal_period is not None:
        if int(legacy_seasonal_period) != legacy_seasonal_period or legacy_seasonal_period < 2:
            raise ValueError('legacy_seasonal_period must be an integer >= 2.')
    cfg = make_config(model_arch, dataset_name, target_features, exog_features,
                       known_future_features, data_root=data_root,
                       parameter_mode=parameter_mode, **kwargs)
    cfg.EVAL.MODE = evaluation_mode
    if evaluation_mode == 'legacy':
        if legacy_seasonal_period is not None:
            cfg.MODEL.PARAM['seasonal_period'] = int(legacy_seasonal_period)
        cfg.DESCRIPTION += '; MetroFlow legacy data; all original windows; no daily reset'
        reset_suffix = ''
    else:
        metadata = desc['operating_hours']
        steps, season = metadata['steps_per_day'], metadata['weekly_period']
        if season != 7*steps:
            raise ValueError('MetroFlow weekly period must be seven complete operating days.')
        cfg.DATASET.TYPE = MetroFlowForecastingDataset
        cfg.MODEL.PARAM.update(seasonal_period=season, steps_per_day=steps)
        cfg.DESCRIPTION += '; MetroFlow corrected data; within-day windows; daily short-term reset'
        reset_suffix = '_daily_reset'
    season = cfg.MODEL.PARAM['seasonal_period']
    cfg.TRAIN.CKPT_SAVE_DIR = str(Path('checkpoints')/model_arch.__name__/
                                 f'{dataset_name}_{evaluation_mode}_{cfg.MODEL.PARAM["parameter_mode"]}'
                                 f'_ma_minus_s{season}{reset_suffix}_'
                                 f'{cfg.DATASET.PARAM.input_len}_{cfg.DATASET.PARAM.output_len}')
    return cfg
