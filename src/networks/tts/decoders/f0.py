import torch
from torch import nn

from src.networks.misc.cln import ConditionalLayerNormalization
from src.networks.misc.convrelunorm import ConvReLUNorm


class F0Decoder(nn.Module):
    """
    F0 decoder.
    """

    def __init__(
        self,
        conv_in_channels: int,
        conv_out_channels: int,
        conv_kernel_size: int,
        conv_stride: int,
        style_dim: int,
        dropout: float,
        n_convs: int,
        gru_hidden_size: int,
        gru_num_layers: int,
        conv_middle_channels: int,
    ):
        """Initializer.
        Args:
            ConvReLUNorm layers params,
            ConditionalLayerNormalization layers params,
            GRU params,
            Linear layer params.
        """
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "conv": ConvReLUNorm(
                            conv_in_channels if i == 0 else conv_middle_channels,
                            conv_out_channels
                            if i == n_convs - 1
                            else conv_middle_channels,
                            conv_kernel_size,
                            conv_stride,
                            dropout,
                        ),
                        "cln": ConditionalLayerNormalization(
                            style_dim,
                            conv_out_channels
                            if i == n_convs - 1
                            else conv_middle_channels,
                        ),
                    }
                )
                for i in range(n_convs)
            ]
        )
        self.gru = nn.GRU(
            conv_out_channels,
            gru_hidden_size,
            gru_num_layers,
        )
        self.linear = nn.Linear(gru_hidden_size, 1)

    def forward(
        self,
        phoneme_features: torch.Tensor,
        amplitude_hiddens: torch.Tensor,
        style_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            phoneme_features: [torch.float32; [B, 128, N]], upsampled phoneme features.
            amplitude_hiddens: [torch.float32; [B, 128, N]].
        Returns:
            f0_contour: [torch.float32; [B, 1, N]] predicted fundamental frequency.
        """
        loop_variable = phoneme_features + amplitude_hiddens
        for conv in self.convs:
            loop_variable = conv["conv"](loop_variable)
            loop_variable = conv["cln"](loop_variable, style_embedding)
        gru_out, _ = self.gru(loop_variable.transpose(1, 2))
        f0_contour: torch.Tensor = self.linear(gru_out).transpose(1, 2)
        return f0_contour.sigmoid()
