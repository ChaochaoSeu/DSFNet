# from math import ceil
#
# import torch
# import torch.nn as nn
# from einops import rearrange, repeat
#
# from .cross_encoder import Encoder
# from .cross_decoder import Decoder
# from .cross_embed import DSW_embedding
#
#
# class Crossformer(nn.Module):
#     """
#         Paper: Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
#         Link: https://openreview.net/forum?id=vSVLM2j9eie
#         Official Code: https://github.com/Thinklab-SJTU/Crossformer
#         Venue: ICLR 2023
#         Task: Long-term Time Series Forecasting
#     """
#     def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
#                 factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3,
#                 dropout=0.0, baseline = False, num_channels=1):
#         super(Crossformer, self).__init__()
#         self.data_dim = data_dim
#         self.in_len = in_len
#         self.out_len = out_len
#         self.seg_len = seg_len
#         self.merge_win = win_size
#         self.num_channels = num_channels  # Store the channel count.
#
#         self.baseline = baseline
#
#         # The padding operation to handle invisible sgemnet length
#         self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
#         self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
#         self.in_len_add = self.pad_in_len - self.in_len
#
#         # Embedding
#         # self.enc_value_embedding = DSW_embedding(seg_len, d_model)
#         self.enc_value_embedding = DSW_embedding(seg_len, d_model)
#         self.enc_pos_embedding = nn.Parameter(
#             torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model)
#         )
#         self.pre_norm = nn.LayerNorm(d_model)
#
#         # Add a channel fusion layer.
#         if num_channels > 1:
#             self.channel_fusion = nn.Linear(num_channels, 1)
#         else:
#             self.channel_fusion = None
#
#         # Encoder
#         self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
#                                     dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)
#
#         # Decoder
#         self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
#         self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
#                                     out_seg_num = (self.pad_out_len // seg_len), factor = factor)
#
#       # Project single-channel outputs back to multiple channels.
#         if num_channels > 1:
#             self.output_projection = nn.Linear(1, num_channels)
#         else:
#             self.output_projection = None
#
#
#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
#         x_seq = history_data  # [B, L, N, C]
#         batch_size = x_seq.shape[0]
#         # Process the baseline component.
#         if (self.baseline):
#             base = x_seq.mean(dim = 1, keepdim = True)
#         else:
#             base = 0
#         # Padding
#         if (self.in_len_add != 0):
#             x_seq = torch.cat((x_seq[:, :1, :, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)
#
#         # Channel fusion: [B, L, N, C] -> [B, L, N]
#         if self.channel_fusion is not None:
#             # [B, L, N, C] -> [B, L, N, 1] -> [B, L, N]
#             x_seq = self.channel_fusion(x_seq).squeeze(-1)
#         else:
#             # Squeeze the channel axis when there is only one channel.
#             x_seq = x_seq.squeeze(-1)
#
#         # Embedding: [B, L, N] -> [B, N, num_seg, d_model]
#         x_seq = self.enc_value_embedding(x_seq)
#         x_seq += self.enc_pos_embedding
#         x_seq = self.pre_norm(x_seq)
#
#         enc_out = self.encoder(x_seq)
#
#         dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
#         predict_y = self.decoder(dec_in, enc_out)  # [B, out_len, N]
#
#         # Output projection: [B, out_len, N] -> [B, out_len, N, C]
#         if self.output_projection is not None:
#             predict_y = predict_y.unsqueeze(-1)  # [B, out_len, N, 1]
#             predict_y = self.output_projection(predict_y)  # [B, out_len, N, C]
#         else:
#             predict_y = predict_y.unsqueeze(-1)  # [B, out_len, N, 1]
#
#         pred = base + predict_y[:, :self.out_len, :, :]  # [B, out_len, N, C]
#
#         return pred

from math import ceil

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .cross_encoder import Encoder
from .cross_decoder import Decoder
from .cross_embed import DSW_embedding


class Crossformer(nn.Module):
    """
        Paper: Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
        Link: https://openreview.net/forum?id=vSVLM2j9eie
        Official Code: https://github.com/Thinklab-SJTU/Crossformer
        Venue: ICLR 2023
        Task: Long-term Time Series Forecasting
    """
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3,
                dropout=0.0, baseline = False, num_channels=1):
        super(Crossformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.num_channels = num_channels  # Store the channel count.

        self.baseline = baseline

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        # self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_value_embedding = nn.ModuleList([
            DSW_embedding(seg_len, d_model) for _ in range(num_channels)
        ])
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model)
        )
        self.pre_norm = nn.LayerNorm(d_model)

        self.fusion_layer = nn.Linear(d_model * num_channels, d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)

        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        x_seq = history_data  # [B, L, N, C]
        batch_size = x_seq.shape[0]
        # Process the baseline component.
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        # Padding
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        # Embed each channel separately.
        channel_embeddings = []
        for c in range(self.num_channels):
            x_seq_c = x_seq[:, :, :, c]  # [B, L, N]
            x_emb_c = self.enc_value_embedding[c](x_seq_c)  # [B, N, num_seg, d_model]
            channel_embeddings.append(x_emb_c)

        x_seq_emb = torch.cat(channel_embeddings, dim=-1)  # [B, N, num_seg, d_model*C]
        x_seq_emb = self.fusion_layer(x_seq_emb)
        # x_seq_emb = torch.stack(channel_embeddings, dim=0).mean(dim=0)  # [B, N, num_seg, d_model]
        x_seq_emb += self.enc_pos_embedding
        x_seq_emb = self.pre_norm(x_seq_emb)

        enc_out = self.encoder(x_seq_emb)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)  # [B, out_len, N]

        # Expand the output to multiple channels.
        # predict_y: [B, out_len, N] -> [B, out_len, N, C]
        predict_y = predict_y.unsqueeze(-1).expand(-1, -1, -1, self.num_channels)

        pred = base + predict_y[:, :self.out_len, :, :]  # [B, out_len, N, C]
        return pred
