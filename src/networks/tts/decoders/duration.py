import torch
from torch import nn

from src.networks.misc.cln import ConditionalLayerNormalization
from src.networks.misc.convrelunorm import ConvReLUNorm


class DurationPredictor(nn.Module):
    """Duration predictor."""

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
            ConditionalLayerNormalization layers params,
            Conv1d params.
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
                            # normalized_shape,
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
        # k=1, s=1
        out_dim = 1
        self.conv1d = nn.Conv1d(conv_out_channels, out_dim, 1, 1)

    def forward(
        self, phoneme_features: torch.Tensor, style_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            phoneme_features: [torch.float32, [B, 128, Ntext]], upsampled phoneme features.
            style_embedding: [torch.float32, [B, 128, N]]
        Returns:
            duration: [torch.float32, [B, 1, Ntext]]
        """
        loop_variable = phoneme_features
        for conv in self.convs:
            loop_variable = conv["conv"](loop_variable)
            loop_variable = conv["cln"](loop_variable, style_embedding)
        conv_out: torch.Tensor = self.conv1d(loop_variable)
        return conv_out.relu()
