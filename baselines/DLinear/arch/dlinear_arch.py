# import torch
# import torch.nn as nn
#
#
# class moving_avg(nn.Module):
#     """Moving average block to highlight the trend of time series"""
#
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size,
#                                 stride=stride, padding=0)
#
#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x
#
#
# class series_decomp(nn.Module):
#     """Series decomposition block"""
#
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)
#
#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean
#
#
# class DLinear(nn.Module):
#     """
#         Paper: Are Transformers Effective for Time Series Forecasting?
#         Link: https://arxiv.org/abs/2205.13504
#         Official Code: https://github.com/cure-lab/DLinear
#         Venue: AAAI 2023
#         Task: Long-term Time Series Forecasting
#     """
#     def __init__(self, **model_args):
#         super(DLinear, self).__init__()
#         self.seq_len = model_args["seq_len"]
#         self.pred_len = model_args["pred_len"]
#
#         # Decompsition Kernel Size
#         kernel_size = 25
#         self.decompsition = series_decomp(kernel_size)
#         self.individual = model_args["individual"]
#         self.channels = model_args["enc_in"]
#
#         if self.individual:
#             self.Linear_Seasonal = nn.ModuleList()
#             self.Linear_Trend = nn.ModuleList()
#
#             for i in range(self.channels):
#                 self.Linear_Seasonal.append(
#                     nn.Linear(self.seq_len, self.pred_len))
#                 self.Linear_Trend.append(
#                     nn.Linear(self.seq_len, self.pred_len))
#
#         else:
#             self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
#             self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
#
#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         """Feed forward of DLinear.
#
#         Args:
#             history_data (torch.Tensor): history data with shape [B, L, N, C]
#
#         Returns:
#             torch.Tensor: prediction with shape [B, L, N, C]
#         """
#
#         assert history_data.shape[-1] == 1      # only use the target feature
#         x = history_data[..., 0]     # B, L, N
#         seasonal_init, trend_init = self.decompsition(x)
#         seasonal_init, trend_init = seasonal_init.permute(
#             0, 2, 1), trend_init.permute(0, 2, 1)
#         if self.individual:
#             seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(
#                 1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
#             trend_output = torch.zeros([trend_init.size(0), trend_init.size(
#                 1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
#             for i in range(self.channels):
#                 seasonal_output[:, i, :] = self.Linear_Seasonal[i](
#                     seasonal_init[:, i, :])
#                 trend_output[:, i, :] = self.Linear_Trend[i](
#                     trend_init[:, i, :])
#         else:
#             seasonal_output = self.Linear_Seasonal(seasonal_init)
#             trend_output = self.Linear_Trend(trend_init)
#
#         prediction = seasonal_output + trend_output
#         return prediction.permute(0, 2, 1).unsqueeze(-1)  # [B, L, N, 1]

import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,
                                stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block"""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
        Paper: Are Transformers Effective for Time Series Forecasting?
        Link: https://arxiv.org/abs/2205.13504
        Official Code: https://github.com/cure-lab/DLinear
        Venue: AAAI 2023
        Task: Long-term Time Series Forecasting
    """

    def __init__(self, **model_args):
        super(DLinear, self).__init__()
        self.seq_len = model_args["seq_len"]
        self.pred_len = model_args["pred_len"]

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = model_args["individual"]
        self.channels = model_args["enc_in"]  # Number of nodes N
        self.num_features = model_args.get("num_features", 1)  # Number of features C

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            # Create a separate linear layer for each feature at each node.
            for i in range(self.channels):
                seasonal_layers = nn.ModuleList()
                trend_layers = nn.ModuleList()
                for j in range(self.num_features):
                    seasonal_layers.append(nn.Linear(self.seq_len, self.pred_len))
                    trend_layers.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal.append(seasonal_layers)
                self.Linear_Trend.append(trend_layers)

        else:
            # Share the same linear layers across all nodes and features.
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """Feed forward of DLinear.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        x = history_data  # [B, L, N, C]
        B, L, N, C = x.shape

        # Reshape to [B*N*C, L] for decomposition.
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * N * C, L)  # [B*N*C, L]

        # Time-series decomposition
        seasonal_init, trend_init = self.decompsition(x_reshaped.unsqueeze(-1))  # [B*N*C, L, 1]
        seasonal_init = seasonal_init.squeeze(-1)  # [B*N*C, L]
        trend_init = trend_init.squeeze(-1)  # [B*N*C, L]

        if self.individual:
            # Individual mode: process each feature at each node separately.
            seasonal_init = seasonal_init.reshape(B, N, C, L)  # [B, N, C, L]
            trend_init = trend_init.reshape(B, N, C, L)  # [B, N, C, L]

            seasonal_output = torch.zeros([B, N, C, self.pred_len],
                                          dtype=seasonal_init.dtype,
                                          device=seasonal_init.device)
            trend_output = torch.zeros([B, N, C, self.pred_len],
                                       dtype=trend_init.dtype,
                                       device=trend_init.device)

            for i in range(N):
                for j in range(C):
                    seasonal_output[:, i, j, :] = self.Linear_Seasonal[i][j](
                        seasonal_init[:, i, j, :])
                    trend_output[:, i, j, :] = self.Linear_Trend[i][j](
                        trend_init[:, i, j, :])

            prediction = seasonal_output + trend_output  # [B, N, C, pred_len]
            prediction = prediction.permute(0, 3, 1, 2)  # [B, pred_len, N, C]

        else:
            # Shared mode: use the same linear layers for all nodes and features.
            seasonal_output = self.Linear_Seasonal(seasonal_init)  # [B*N*C, pred_len]
            trend_output = self.Linear_Trend(trend_init)  # [B*N*C, pred_len]

            prediction = seasonal_output + trend_output  # [B*N*C, pred_len]
            prediction = prediction.reshape(B, N, C, self.pred_len)  # [B, N, C, pred_len]
            prediction = prediction.permute(0, 3, 1, 2)  # [B, pred_len, N, C]

        return prediction  # [B, L, N, C]
