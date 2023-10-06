from typing import Tuple

import torch
from torch import nn

from src.networks.misc.cln import ConditionalLayerNormalization
from src.networks.misc.convrelunorm import ConvReLUNorm


class AmplitudeDecoder(nn.Module):
    """Amplitude decoder."""

    def __init__(
        self,
        conv_in_channels: int,
        conv_out_channels: int,
        conv_kernel_size: int,
        conv_stride: int,
        style_dim: int,
        dropout: float,
        n_convs: int,
        conv_middle_channels: int,
    ):
        """Initializer.
        Args:
            ConvReLUNorm layers params,
            ConditionalLayerNormalization params,
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
        self.linear = nn.Linear(conv_out_channels, 2)

    def forward(
        self, phoneme_features: torch.Tensor, style_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            phoneme_features: [torch.float32, [B, 128, N]], upsampled phoneme features.
            style_embedding: [torch.float32, [B, 128, N]].
        Returns:
            p_amp: [torch.float32; [B, 1, N]], periodic amplitude.
            ap_amp: [torch.float32; [B, 1, N]], aperiodic amplitude,
            hiddens: [torch.float32; [B, 128, N]], amplitude hidden features.
        """
        hiddens: torch.Tensor = phoneme_features
        for conv in self.convs:
            hiddens = conv["conv"](hiddens)
            hiddens = conv["cln"](hiddens, style_embedding)
        linear_out: torch.Tensor = self.linear(hiddens.transpose(1, 2)).transpose(1, 2)
        p_amp, ap_amp = torch.split(linear_out, [1, 1], dim=1)
        return (
            p_amp.sigmoid().squeeze(dim=-1),
            ap_amp.sigmoid().squeeze(dim=-1),
            hiddens,
        )
