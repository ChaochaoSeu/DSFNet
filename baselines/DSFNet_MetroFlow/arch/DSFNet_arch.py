import torch
from torch import nn
import torch.nn.functional as F

class ExVarEmbedding(nn.Module):

    def __init__(self, time_of_day_size):
        super(ExVarEmbedding, self).__init__()

        # MeteorologicalEmbedding
        self.embed_temp = nn.Embedding(4, 3)
        self.embed_rain = nn.Embedding(3, 3)
        self.embed_wind = nn.Embedding(4, 3)

        # TimeEmbedding
        # a day contains time_of_day_size slots
        self.embed_slot = nn.Embedding(time_of_day_size, 3)  # Time of Day
        self.embed_day = nn.Embedding(7, 3)  # Day of the Week

    def forward(self, x):
        """Embed exogenous inputs of shape [B, T, N, K].

        Input columns: temperature class, rainfall class, wind class, time slot, weekday, workday flag.
        Returns a tensor of shape [B, T, N, 16].
        """

        x_temp = self.embed_temp(x[..., 0])
        x_rain = self.embed_rain(x[..., 1])
        x_wind = self.embed_wind(x[..., 2])

        x_slot = self.embed_slot(x[..., 3])
        x_day = self.embed_day(x[..., 4])


        out = torch.cat((x_temp, x_rain, x_wind , x_slot, x_day, x[..., -1:]), -1)

        # out
        # [0:9]   Weather features
        # [9:16] Temporal features
        return out  # [b, t, n, 16]

class PreNorm(nn.Module):
    """
    Pre-normalization module that applies layer normalization before forwarding the input to the given function.

    Args:
        dim (int): The input dimension.
        fn (callable): The function to be applied after layer normalization.

    Attributes:
        norm (nn.LayerNorm): The layer normalization module.
        fn (callable): The function to be applied after layer normalization.

    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Forward pass of the PreNorm module.

        Args:
            x (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            torch.Tensor: The output tensor after applying layer normalization and the given function.

        """
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """
    A feedforward neural network module.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The dimension of the hidden layer.
        dropout (float, optional): The dropout probability. Default is 0.0.

    Attributes:
        net (nn.Sequential): The sequential network architecture.

    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass of the feedforward neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        return self.net(x)

class DDI_layer(nn.Module):

    def __init__(self, num_nodes, hidden_size, num_blocks, num_blocks_f, GCO_Thre=0.5, FFO_Thre=0.5, hidden_size_factor=2):
        super().__init__()
        assert hidden_size % num_blocks == 0
        self.num_blocks_f = num_blocks_f if num_blocks_f is not None else num_blocks

        # Compute the number of feature modes per block.
        total_modes_f = hidden_size // 2 + 1
        kept_modes_f = int(total_modes_f * FFO_Thre)
        assert kept_modes_f % self.num_blocks_f == 0
        self.block_size_f = kept_modes_f // self.num_blocks_f

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.GCO_Thre = GCO_Thre
        self.FFO_Thre = FFO_Thre
        self.hidden_size_factor = hidden_size_factor

        self.scale_l = 0.1
        self.scale_r = 0.5

        self.pos_embedding = nn.Parameter(torch.randn(num_nodes, hidden_size))

        # Spatial filtering parameters
        self.w1 = nn.Parameter(
            self.scale_l * torch.randn(num_blocks, self.block_size, self.block_size * hidden_size_factor))
        self.b1 = nn.Parameter(self.scale_l * torch.randn(num_blocks, self.block_size * hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale_l * torch.randn(num_blocks, self.block_size * hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale_l * torch.randn(num_blocks, self.block_size))

        # Feature filtering parameters
        self.ffo_w1 = nn.Parameter(
            self.scale_r * torch.randn(self.num_blocks_f, self.block_size_f, self.block_size_f * hidden_size_factor))
        self.ffo_b1 = nn.Parameter(
            self.scale_r * torch.randn(self.num_blocks_f, self.block_size_f * hidden_size_factor))
        self.ffo_w2 = nn.Parameter(
            self.scale_r * torch.randn(self.num_blocks_f, self.block_size_f * hidden_size_factor, self.block_size_f))
        self.ffo_b2 = nn.Parameter(self.scale_r * torch.randn(self.num_blocks_f, self.block_size_f))

    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape


        # Step 1: Filter along the latent feature axis.
        x_f = torch.fft.rfft(x, dim=2, norm="ortho")
        total_modes_f = C // 2 + 1
        kept_modes_f = int(total_modes_f * self.FFO_Thre)
        x_f = x_f[..., :kept_modes_f]
        x_f = x_f.reshape(B, N, self.num_blocks_f, kept_modes_f // self.num_blocks_f)

        # Apply grouped linear maps with bias.
        real_1f = F.relu(
            torch.einsum("bnfc,fcd->bnfd", x_f.real, self.ffo_w1) +
            self.ffo_b1[None, None, :, :]
        )
        real_2f = (
                torch.einsum("bnfc,fcd->bnfd", real_1f, self.ffo_w2) +
                self.ffo_b2[None, None, :, :]
        )

        imag_f = torch.zeros_like(real_2f)
        x_f = torch.view_as_complex(torch.stack([real_2f, imag_f], dim=-1))
        x_f = x_f.reshape(B, N, kept_modes_f)
        x = torch.fft.irfft(x_f, n=C, dim=2, norm="ortho")

        # ===================================================
        # Step 2: Apply Fourier filtering along the node axis.
        # ===================================================
        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)

        real_1 = torch.zeros(
            [B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device,
        )
        real_2 = torch.zeros(x.shape, device=x.device)
        imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.GCO_Thre)

        real_1[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1) + self.b1
        )

        real_2[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", real_1[:, :kept_modes], self.w2) + self.b2
        )

        x = torch.stack([real_2, imag], dim=-1)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")

        x = x.type(dtype)

        return x + bias

class DDImodule(nn.Module):
    def __init__(
        self,
        num_nodes,
        dim,
        depth,
        mlp_dim,
        GCO_Thre,
        FFO_Thre,
        dropout=0.0,
    ):
        super().__init__()
        self.GCO_Thre = GCO_Thre
        self.FFO_Thre = FFO_Thre
        self.hidden_size_factor = 2
        self.dim = dim
        self.hidden_dim = dim

        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        DDI_layer(
                            num_nodes = num_nodes,
                            hidden_size=self.hidden_dim,
                            num_blocks=8,
                            num_blocks_f=2,
                            GCO_Thre=self.GCO_Thre,
                            FFO_Thre=self.FFO_Thre,
                            hidden_size_factor=self.hidden_size_factor
                        ),
                        PreNorm(self.hidden_dim, FeedForward(self.hidden_dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        """
        Forward pass of the dual-domain interaction module.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, N, T].

        """
        b, c, n, t = x.shape

        x = x.permute(0, 3, 2, 1).reshape(b * t, n, c)

        for dfo, ff in self.layers:
            x = dfo(x)
            x = ff(x) + x

        x = x.reshape(b, t, n, self.hidden_dim).permute(0, 3, 2, 1)  # b d n t

        return x


class DSFNet(nn.Module):

    def __init__(self, num_nodes, dropout=0.3, GCO_Thre = 0.5, FFO_Thre = 0.5, hidden_size_factor=2,
                    in_dim=2, pre_len=12, feature_dim = 6, residual_channels=32,
                    dilation_channels=32, skip_channels=256, end_channels=512,
                    kernel_size=2, blocks=4, layers=2, time_of_day_size = 288):
        super(DSFNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.in_dim = in_dim
        self.pre_len = pre_len
        self.feature_dim = feature_dim

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dfos = nn.ModuleList()
        receptive_field = 1
        self.exog_dim = 16

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.start_exog_conv = nn.Conv2d(in_channels=self.exog_dim,
                                    out_channels=dilation_channels,
                                    kernel_size=(1, 1))

        self.embedding = ExVarEmbedding(time_of_day_size)

        for i in range(layers):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for b in range(blocks):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.skip_gate_convs.append(nn.Conv2d(in_channels=dilation_channels*2,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))


                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

                self.dfos.append(DDImodule(num_nodes, dilation_channels, 1, 512, GCO_Thre, FFO_Thre, dropout))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=pre_len * feature_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Forward pass of DSFNet.

        Args:
            history_data (torch.Tensor): shape [B, L, N, C]

        Returns:
            torch.Tensor: [B, L, N, D]
        """
        B, L, N, C = history_data.shape

        selected =  history_data[..., -6:]
        x_embed = self.embedding(selected.long())

        X = history_data[..., 0:self.in_dim]

        weather = x_embed[:, :, :, :9] # [b, t, n, 9] Weather features
        covars = x_embed[:, :, :, 9:16]  # [b, t, n, 7] Temporal features

        exog_features = torch.cat([weather, covars], dim=-1)   # [b, t, n, 16]

        input = X.transpose(1, 3).contiguous()
        exog_features = exog_features.transpose(1, 3).contiguous()  # [b, 16, n, t]

        # input [B, C, N, L]
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field-in_len, 0, 0, 0))
            exog_features = nn.functional.pad(
                exog_features,
                (self.receptive_field - in_len, 0, 0, 0),
                mode='replicate'
            )
        else:
            x = input

        x = self.start_conv(x)

        exog_features = self.start_exog_conv(exog_features)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)

            x = filter
            s = x
            s = self.skip_convs[i](s)

            select_exog_features = exog_features[...,-s.size(3):]
            features = torch.cat([x, select_exog_features], dim=1)

            gate_s = self.skip_gate_convs[i](features)
            s = torch.tanh(s)
            gate_s = torch.sigmoid(gate_s)
            s = s * gate_s

            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0

            skip = s + skip

            x = self.dfos[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)

        x = F.relu(self.end_conv_1(x))

        x = self.end_conv_2(x)

        x = x.view(B, self.pre_len, self.feature_dim, N, 1)

        x = x.squeeze(-1)  # (B, pre_len, feature_dim, N)

        x = x.permute(0, 1, 3, 2)  # (B, pre_len, N, feature_dim)

        return x
