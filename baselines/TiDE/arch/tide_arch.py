# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from basicts.utils import data_transformation_4_xformer
#
# class LayerNorm(nn.Module):
#     """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
#
#     def __init__(self, ndim, bias):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(ndim))
#         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
#
#     def forward(self, input):
#         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
#
#
# class ResBlock(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
#         super().__init__()
#
#         self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
#         self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
#         self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
#         self.ln = LayerNorm(output_dim, bias=bias)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.dropout(out)
#         out = out + self.fc3(x)
#         out = self.ln(out)
#         return out
#
#
# # TiDE
# class TiDE(nn.Module):
#     """
#     Paper: Long-term Forecasting with TiDE: Time-series Dense Encoder
#     Official Code: https://github.com/lich99/TiDE
#     Link: https://arxiv.org/abs/2304.08424
#     Venue: TMLR 2023
#     Task: Long-term Time Series Forecasting
#     """
#
#     def __init__(self,  **model_args):
#         super(TiDE, self).__init__()
#
#         self.seq_len = model_args['seq_len']  # L
#         self.label_len = int(model_args['label_len'])
#         self.pred_len = model_args['pred_len']  # H
#         self.hidden_dim = model_args['d_model']
#         self.res_hidden = model_args['d_model']
#         self.encoder_num = model_args['e_layers']
#         self.decoder_num =  model_args['d_layers']
#         self.freq = model_args["freq"]
#         self.bias = model_args["bias"]
#         self.feature_encode_dim = model_args["feature_encode_dim"]
#         self.decode_dim = model_args['c_out']
#         self.temporalDecoderHidden = model_args['d_ff']
#         dropout = model_args["dropout"]
#         self.target_dims = model_args['target_dims']
#         self.mark_dims = model_args['mark_dims']
#
#         freq_map = {'h': 4, 't': 5, 's': 6,
#                     'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
#
#         self.feature_dim = freq_map[self.freq]
#
#         flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim
#
#         self.feature_encoder = ResBlock(self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, self.bias)
#         self.encoders = nn.Sequential(ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, self.bias), *(
#                     [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, self.bias)] * (self.encoder_num - 1)))
#
#         self.decoders = nn.Sequential(*(
#                         [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, self.bias)] * (
#                             self.decoder_num - 1)),
#                                           ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len,
#                                                    dropout, self.bias))
#         self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1,
#                                             dropout, self.bias)
#         self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=self.bias)
#
#
#     def forward_xformer(self, x_enc, x_mark_enc, x_dec, batch_y_mark) -> torch.Tensor:
#         # Normalization
#         means = x_enc.mean(1, keepdim=True).detach()
#         x_enc = x_enc - means
#         stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         x_enc /= stdev
#
#         feature = self.feature_encoder(batch_y_mark)
#         hidden = self.encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
#         decoded = self.decoders(hidden).reshape(hidden.shape[0], self.pred_len, self.decode_dim)
#         dec_out = self.temporalDecoder(torch.cat([feature[:, self.seq_len:], decoded], dim=-1)).squeeze(
#             -1) + self.residual_proj(x_enc)
#
#         # De-Normalization
#         dec_out = dec_out * (stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len))
#         dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len))
#         return dec_out
#
#
#
#     def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
#                 **kwargs) -> torch.Tensor:
#         '''x_mark_enc is the exogenous dynamic feature described in the original paper'''
#         x_enc, x_mark_enc, x_dec, batch_y_mark = data_transformation_4_xformer(history_data=history_data,
#                                                                              future_data=future_data,
#                                                                              start_token_len=0,
#                                                                              target_dims=self.target_dims,
#                                                                              mark_dims=self.mark_dims)
#         print(x_enc.shape, x_mark_enc.shape, x_dec.shape, batch_y_mark.shape)
#         batch_y_mark = torch.concat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]], dim=1)
#
#         batch_y_mark = torch.concat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]], dim=1)
#         dec_out = torch.stack([self.forward_xformer(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark) for feature in
#                                range(x_enc.shape[-1])], dim=-1)
#         return dec_out.unsqueeze(-1)  # [B, L, D]
#
#
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicts.utils import data_transformation_4_xformer


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


# TiDE
class TiDE(nn.Module):
    """
    Paper: Long-term Forecasting with TiDE: Time-series Dense Encoder
    Official Code: https://github.com/lich99/TiDE
    Link: https://arxiv.org/abs/2304.08424
    Venue: TMLR 2023
    Task: Long-term Time Series Forecasting
    """

    def __init__(self, **model_args):
        super(TiDE, self).__init__()

        self.seq_len = model_args['seq_len']  # L
        self.label_len = int(model_args['label_len'])
        self.pred_len = model_args['pred_len']  # H
        self.hidden_dim = model_args['d_model']
        self.res_hidden = model_args['d_model']
        self.encoder_num = model_args['e_layers']
        self.decoder_num = model_args['d_layers']
        self.freq = model_args["freq"]
        self.bias = model_args["bias"]
        self.feature_encode_dim = model_args["feature_encode_dim"]
        self.decode_dim = model_args['c_out']
        self.temporalDecoderHidden = model_args['d_ff']
        dropout = model_args["dropout"]
        self.target_dims = model_args['target_dims']
        self.mark_dims = model_args['mark_dims']

        # Use the actual temporal feature dimension rather than a frequency lookup.
        # freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        # self.feature_dim = freq_map[self.freq]

        # Set the temporal feature dimension from the length of mark_dims.
        self.feature_dim = len(self.mark_dims)  # Expected temporal feature count: 3

        # Count the target features.
        self.target_feature_num = len(self.target_dims)  # Expected target feature count: 10

        # Read the number of spatial nodes from the configuration when available.
        # The [B, L, N, T] input layout requires the node count N.
        # Infer the node count from c_out, which typically represents N*T or N.
        self.num_nodes = model_args.get('enc_in')

        # Include the spatial dimension in flatten_dim.
        # A flattened single-feature input has shape [B, L*N].
        flatten_dim = self.seq_len * self.num_nodes + (self.seq_len + self.pred_len) * self.feature_encode_dim

        self.feature_encoder = ResBlock(self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, self.bias)
        self.encoders = nn.Sequential(ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, self.bias), *(
                [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, self.bias)] * (
                    self.encoder_num - 1)))

        self.decoders = nn.Sequential(*(
                [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, self.bias)] * (
                self.decoder_num - 1)),
                                      ResBlock(self.hidden_dim, self.res_hidden, self.num_nodes * self.pred_len,
                                               dropout, self.bias))
        self.temporalDecoder = ResBlock(self.num_nodes + self.feature_encode_dim, self.temporalDecoderHidden,
                                        self.num_nodes,
                                        dropout, self.bias)
        self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=self.bias)

    def forward_xformer(self, x_enc, batch_y_mark) -> torch.Tensor:
        """
        x_enc: [B, L, N] - Historical values of one target feature across all nodes
        batch_y_mark: [B, L+H, F] - Historical and future temporal features
        Returns: [B, H, N] - Predictions for one feature across all nodes
        """
        # Normalize over the spatial dimension.
        means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, N]
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Encode temporal features.
        batch_size = batch_y_mark.shape[0]
        seq_total_len = batch_y_mark.shape[1]  # L+H

        batch_y_mark_flat = batch_y_mark.reshape(-1, self.feature_dim)  # [B*(L+H), F]
        feature = self.feature_encoder(batch_y_mark_flat)  # [B*(L+H), feature_encode_dim]
        feature = feature.reshape(batch_size, seq_total_len, self.feature_encode_dim)  # [B, L+H, feature_encode_dim]

        # Prepare the encoder input.
        # x_enc: [B, L, N] -> flatten to [B, L*N]
        x_enc_flat = x_enc.reshape(batch_size, -1)  # [B, L*N]

        # feature: [B, L+H, feature_encode_dim] -> flatten to [B, (L+H)*feature_encode_dim]
        feature_flat = feature.reshape(batch_size, -1)  # [B, (L+H)*feature_encode_dim]

        # Concatenate the inputs.
        encoder_input = torch.cat([x_enc_flat, feature_flat], dim=-1)  # [B, L*N + (L+H)*feature_encode_dim]

        hidden = self.encoders(encoder_input)  # [B, hidden_dim]

        # Decode the representation.
        decoded = self.decoders(hidden)  # [B, N * pred_len]
        decoded = decoded.reshape(batch_size, self.pred_len, self.num_nodes)  # [B, H, N]

        # Temporal decoder
        future_feature = feature[:, self.seq_len:, :]  # [B, H, feature_encode_dim]

        # Concatenate the decoder output and future temporal features.
        temporal_input = torch.cat([decoded, future_feature], dim=-1)  # [B, H, N + feature_encode_dim]

        # Temporal decoder output
        dec_out = self.temporalDecoder(temporal_input)  # [B, H, N]

        # Residual connection
        residual = self.residual_proj(x_enc.transpose(1, 2))  # [B, N, L] -> [B, N, H]
        residual = residual.transpose(1, 2)  # [B, H, N]

        dec_out = dec_out + residual

        # De-Normalization

        dec_out = dec_out * stdev
        dec_out = dec_out + means

        return dec_out  # [B, H, N]

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        '''
        history_data: [B, L, D] where D includes target and temporal features
        x_enc: [B, L, N, T] - N is the number of spatial nodes and T is the number of target features
        Expected output: [B, H, T, 1], where T is the number of target features
        '''
        x_enc, x_mark_enc, x_dec, batch_y_mark = data_transformation_4_xformer(
            history_data=history_data,
            future_data=future_data,
            start_token_len=0,
            target_dims=self.target_dims,
            mark_dims=self.mark_dims
        )

        # Concatenate historical and future temporal features.
        # x_mark_enc: [B, L, F] - Historical temporal features
        # batch_y_mark: [B, H, F] - Future temporal features
        batch_y_mark_full = torch.cat([x_mark_enc, batch_y_mark], dim=1)  # [B, L+H, F]

        # Iterate over the feature axis of x_enc, which has shape [B, L, N, T].
        # x_enc[:, :, :, t] has shape [B, L, N] for feature t across all nodes.
        results = []
        for feature_idx in range(self.target_feature_num):
            # Extract one target feature across all nodes.
            x_enc_single = x_enc[:, :, :, feature_idx]  # [B, L, N]

            # Run the single-feature forward pass.
            result = self.forward_xformer(x_enc_single, batch_y_mark_full)  # [B, H, N]
            results.append(result)

        # Stack outputs across features.
        # results: T tensors, each with shape [B, H, N]
        dec_out = torch.stack(results, dim=2)  # [B, H, T, N]

        # Reduce the spatial dimension by summation or averaging to obtain [B, H, T].
        # This task averages predictions across nodes.
        # dec_out = dec_out.mean(dim=-1, keepdim=True)  # [B, H, T, 1]
        dec_out = dec_out.transpose(2, 3)

        return dec_out  # [B, H, T, 1], following [B, L, T, C] with L=pred_len and C=1
