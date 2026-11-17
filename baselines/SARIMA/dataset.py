"""Keep BasicTS evaluation windows while exposing absolute forecast origins."""

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class SeasonalForecastingDataset(Dataset):
    """Raw T,N,D memmap with the same non-overlapping windows as BasicTS.

    Validation/test origins match TimeSeriesForecastingDataset(overlap=False).
    Earlier observed splits remain accessible to the model's seasonal filter.
    Training has one bookkeeping window: parameters are set once on the
    complete training split, rather than optimized over shuffled mini-batches.
    """

    def __init__(self, dataset_name, train_val_test_ratio, mode, input_len,
                 output_len, data_root='datasets', logger=None):
        if mode not in ('train', 'valid', 'test'):
            raise ValueError(f'Unknown mode: {mode}')
        folder = Path(data_root) / dataset_name
        with (folder / 'desc.json').open(encoding='utf-8') as stream:
            self.description = json.load(stream)
        shape = tuple(self.description['shape'])
        valid_len = int(shape[0]*train_val_test_ratio[1])
        test_len = int(shape[0]*train_val_test_ratio[2])
        train_len = shape[0] - valid_len - test_len
        start, stop = {'train': (0, train_len),
                       'valid': (train_len, train_len+valid_len),
                       'test': (train_len+valid_len, shape[0])}[mode]
        if stop-start < input_len+output_len:
            raise ValueError('Split is shorter than input_len + output_len; choose longer splits.')
        self.data = np.memmap(folder / 'data.dat', dtype='float32', mode='r', shape=shape)
        self.input_len = input_len
        self.output_len = output_len
        self.first_origin = stop-output_len if mode == 'train' else start+input_len
        self.length = 1 if mode == 'train' else stop-start-input_len-output_len+1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        origin = self.first_origin + index
        return {'inputs': self.data[origin-self.input_len:origin].copy(),
                'target': self.data[origin:origin+self.output_len].copy(),
                'forecast_origin': origin}
