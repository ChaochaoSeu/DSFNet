import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding_inverted
import numpy as np
from basicts.utils import data_transformation_4_xformer



class iTransformer(nn.Module):
    """
    Paper: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    Official Code: https://github.com/thuml/iTransformer
    Link: https://arxiv.org/abs/2310.06625
    Venue: ICLR 2024
    Task: Long-term Time Series Forecasting
    """
    def __init__(self, **model_args):
        super(iTransformer, self).__init__()
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.output_attention = model_args['output_attention']
        self.enc_in = model_args['enc_in']
        self.dec_in = model_args['dec_in']
        self.c_out = model_args['c_out']
        self.factor = model_args["factor"]
        self.d_model = model_args['d_model']
        self.n_heads = model_args['n_heads']
        self.d_ff = model_args['d_ff']
        self.embed = model_args['embed']
        self.freq = model_args["freq"]
        self.dropout = model_args["dropout"]
        self.activation = model_args['activation']
        self.e_layers = model_args['e_layers']
        self.d_layers = model_args['d_layers']
        self.use_norm = model_args['use_norm']
        self.target_dims = model_args['target_dims']
        self.mark_dims = model_args['mark_dims']

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq,
                                                    self.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)

    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor,
                        x_mark_dec: torch.Tensor,
                        enc_self_mask: torch.Tensor = None, dec_self_mask: torch.Tensor = None,
                        dec_enc_mask: torch.Tensor = None) -> torch.Tensor:

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach() # shape: [B, 1, N, Num_Targets]
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, L, N, C = x_enc.shape
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        # C: number of target features
        # Process channels independently.

        # 1. Permute: [B, L, N, C] -> [B, C, L, N]
        x_enc = x_enc.permute(0, 3, 1, 2)
        # 2. Reshape: [B, C, L, N] -> [B*C, L, N]
        x_enc = x_enc.reshape(B * C, L, N)
        if x_mark_enc is not None:
            x_mark_enc = x_mark_enc.repeat_interleave(C, dim=0)

        # Embedding
        # x_enc shape: [B*C, L, N]
        # x_mark_enc shape: [B*C, L, M]
        # Inside DataEmbedding_inverted:
        # 1. x_enc.permute(0, 2, 1) -> [B*C, N, L]
        # 2. x_mark_enc.permute(0, 2, 1) -> [B*C, M, L]
        # 3. torch.cat -> [B*C, N + M, L]
        # 4. self.value_embedding (Linear(L, E)) -> [B*C, N + M, E]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # enc_out shape: [B*C, N + M, E]

        # Encoder
        # [B*C, N + M, E] -> [B*C, N + M, E]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projector
        # [B*C, N + M, E] -> [B*C, N + M, S]
        dec_out = self.projector(enc_out)

        # [B*C, N + M, S] -> [B*C, S, N + M]
        dec_out = dec_out.permute(0, 2, 1)

        # Keep predictions for the N target variables and discard the M covariates.
        # [B*C, S, N + M] -> [B*C, S, N]
        dec_out = dec_out[:, :, :N]

        # Restore the original axis layout.
        # 1. Reshape: [B*C, S, N] -> [B, C, S, N]
        dec_out = dec_out.reshape(B, C, self.pred_len, N)
        # 2. Permute: [B, C, S, N] -> [B, S, N, C]
        dec_out = dec_out.permute(0, 2, 3, 1)

        if self.use_norm:
            # De-Normalization
            # dec_out [B, S, N, C]
            # stdev/means [B, 1, N, C]
            # Broadcast across the remaining dimensions.
            dec_out = dec_out * stdev
            dec_out = dec_out + means

        return dec_out

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, C]
        """
        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
                                                                             future_data=future_data,
                                                                             start_token_len=0,
                                                                             target_dims=self.target_dims,
                                                                             mark_dims=self.mark_dims
                                                                             )

        # x_enc shape: [B, L, N, C_targets]
        # x_mark_enc shape: [B, L, C_marks]

        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)

        # prediction already has shape [B, S, N, C]; no unsqueeze(-1) is needed.
        return prediction