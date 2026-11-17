# Adapted from BasicTS; unused analysis metrics removed.
from .mae import masked_mae
from .mape import masked_mape
from .mse import masked_mse
from .rmse import masked_rmse
from .wape import masked_wape
from .metric_meter import AvgMeter, RMSEMeter

ALL_METRICS = {'MAE': masked_mae, 'MAPE': masked_mape, 'MSE': masked_mse,
               'RMSE': masked_rmse, 'WAPE': masked_wape}
METRIC_METER = {'RMSE': RMSEMeter, 'default': AvgMeter}
__all__ = ['masked_mae', 'masked_mape', 'masked_mse', 'masked_rmse', 'masked_wape',
           'AvgMeter', 'RMSEMeter', 'ALL_METRICS', 'METRIC_METER']
