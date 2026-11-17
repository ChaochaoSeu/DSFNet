import torch
import torch.nn as nn

from basicts.utils import data_transformation_4_xformer

from .embed import DataEmbedding_wo_pos, DataEmbedding
from .auto_correlation import AutoCorrelation, AutoCorrelationLayer
from .enc_dec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(nn.Module):
    """
    Paper: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
    Link: https://arxiv.org/abs/2106.13008
    Ref Official Code: https://github.com/thuml/Autoformer
    Venue: NeurIPS 2021
    Task: Long-term Time Series Forecasting
    """

    def __init__(self, **model_args):
        super(Autoformer, self).__init__()
        self.seq_len = int(model_args["seq_len"])
        self.label_len = int(model_args["label_len"])
        self.pred_len = int(model_args["pred_len"])
        self.output_attention = model_args["output_attention"]

        self.time_of_day_size = model_args.get("time_of_day_size", None)
        self.day_of_week_size = model_args.get("day_of_week_size", None)
        self.day_of_month_size = model_args.get("day_of_month_size", None)
        self.day_of_year_size = model_args.get("day_of_year_size", None)
        self.embed = model_args["embed"]

        self.num_channels = model_args.get("num_channels", 1)
        self.target_dims = model_args["target_dims"]
        self.mark_dims = model_args["mark_dims"]

        # Multiply enc_in and dec_in by the number of channels.
        model_args["enc_in"] = model_args["enc_in"] * self.num_channels
        model_args["dec_in"] = model_args["dec_in"] * self.num_channels

        # Decomp
        kernel_size = model_args["moving_avg"]
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(
                                                    model_args["enc_in"],
                                                    model_args["d_model"],
                                                    self.time_of_day_size,
                                                    self.day_of_week_size,
                                                    self.day_of_month_size,
                                                    self.day_of_year_size,
                                                    model_args["embed"],
                                                    model_args["num_time_features"],
                                                    model_args["dropout"])
        self.dec_embedding = DataEmbedding_wo_pos(
                                                    model_args["dec_in"],
                                                    model_args["d_model"],
                                                    self.time_of_day_size,
                                                    self.day_of_week_size,
                                                    self.day_of_month_size,
                                                    self.day_of_year_size,
                                                    model_args["embed"],
                                                    model_args["num_time_features"],
                                                    model_args["dropout"])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, model_args["factor"], attention_dropout=model_args["dropout"],
                                        output_attention=model_args["output_attention"]),
                        model_args["d_model"], model_args["n_heads"]),
                    model_args["d_model"],
                    model_args["d_ff"],
                    moving_avg=model_args["moving_avg"],
                    dropout=model_args["dropout"],
                    activation=model_args["activation"]
                ) for l in range(model_args["e_layers"])
            ],
            norm_layer=my_Layernorm(model_args["d_model"])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, model_args["factor"], attention_dropout=model_args["dropout"],
                                        output_attention=False),
                        model_args["d_model"], model_args["n_heads"]),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, model_args["factor"], attention_dropout=model_args["dropout"],
                                        output_attention=False),
                        model_args["d_model"], model_args["n_heads"]),
                    model_args["d_model"],
                    model_args["c_out"]*model_args["num_channels"],
                    model_args["d_ff"],
                    moving_avg=model_args["moving_avg"],
                    dropout=model_args["dropout"],
                    activation=model_args["activation"],
                )
                for l in range(model_args["d_layers"])
            ],
            norm_layer=my_Layernorm(model_args["d_model"]),
            projection=nn.Linear(model_args["d_model"], model_args["c_out"]*model_args["num_channels"], bias=True)
        )

    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: torch.Tensor=None, dec_self_mask: torch.Tensor=None, dec_enc_mask: torch.Tensor=None) -> torch.Tensor:

        B, L, N, C = x_enc.shape
        x_enc = x_enc.reshape(B, L, N * C)

        B_dec, L_dec, N_dec, C_dec = x_dec.shape
        x_dec = x_dec.reshape(B_dec, L_dec, N_dec * C_dec)

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)

        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        dec_out = dec_out[:, -self.pred_len:, :]
        dec_out = dec_out.reshape(B, self.pred_len, N, C)

        return dec_out  # [B, L, N, C]

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """
        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
                                                                             future_data=future_data,
                                                                             start_token_len=self.label_len,
                                                                             target_dims=self.target_dims,
                                                                             mark_dims=self.mark_dims)
        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
        return prediction
