import torch

from basicts.runners import NoBPRunner


class SeasonalARIMARunner(NoBPRunner):
    """Use BasicTS metrics/checkpoints with CPU conditional estimation."""

    def __init__(self, cfg):
        if cfg.get('GPU_NUM', 0) != 0 or 'SCALER' in cfg:
            raise ValueError('SeasonalARIMARunner requires CPU mode and raw, unscaled data.')
        super().__init__(cfg)

    def build_model(self, cfg):
        # Seasonal recursion and conditional GLS use NumPy/SciPy on CPU,
        # including when evaluation's command-line device is left at default.
        return self.define_model(cfg).cpu()

    @torch.no_grad()
    def forward(self, data, epoch=None, iter_num=None, train=True, **kwargs):
        prediction = self.model(forecast_origin=data['forecast_origin'])
        return {'prediction': prediction,
                'target': self.select_target_features(data['target']).cpu(),
                'inputs': self.select_input_features(data['inputs']).cpu()}
