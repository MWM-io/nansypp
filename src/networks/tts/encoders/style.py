import torch
from torch import nn

from src.networks.backbone.encoders.timbre import TTSAttentiveStatisticsPooling
from src.networks.backbone.encoders.wav2vec2 import Wav2Vec2Wrapper
from src.networks.misc.convrelunorm import ConvReLUNorm


class StyleEncoder(nn.Module):
    """Style encoder."""

    def __init__(
        self,
        conv_in_channels: int,
        conv_middle_channels: int,
        conv_out_channels: int,
        conv_kernel_size: int,
        conv_stride: int,
        dropout: float,
        n_convs: int,
        bottleneck: int,
        wav2vec2: Wav2Vec2Wrapper,
        out_linear: int,
    ):
        """Initializer.
        Args:
            ConvReLUNorm layers params,
            AttentiveStatisticsPooling params,
            Linear layer params.
        """
        super().__init__()
        # Wav2Vec2 first 3 layers
        self.w2v_3rd = wav2vec2
        # ConvReLUNorm
        self.convs = nn.ModuleList(
            [
                ConvReLUNorm(
                    conv_in_channels if i == 0 else conv_middle_channels,
                    conv_out_channels if i == n_convs - 1 else conv_middle_channels,
                    conv_kernel_size,
                    conv_stride,
                    # normalized_shape,
                    dropout,
                )
                for i in range(n_convs)
            ]
        )
        # LeakyRelu
        self.asp = TTSAttentiveStatisticsPooling(conv_out_channels, bottleneck)
        # Dropout
        self.linear = nn.Linear(
            in_features=2 * conv_out_channels, out_features=out_linear
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [torch.float32; [B, N]]
        Returns:
            style_embedding: [torch.float32; [B, 128, 1]]
        """
        w2v_features = self.w2v_3rd(waveform).transpose(1, 2)
        convs_hidden = w2v_features
        for conv in self.convs:
            convs_hidden = conv(convs_hidden)
        asp_hidden = self.asp(convs_hidden)
        linear_out: torch.Tensor = self.linear(asp_hidden).unsqueeze(2)
        return linear_out
