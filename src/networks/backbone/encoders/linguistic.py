from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from src.networks.backbone.encoders.wav2vec2 import Wav2Vec2Wrapper


class ConvGLU(nn.Module):
    """Dropout - Conv1d - GLU - residual connection."""

    def __init__(self, in_channels: int, kernels: int, dropout: float):
        """Initializer.
        Args:
            in_channels: size of the input channels.
            kernels: size of the convolutional kernels.
            dropout: dropout rate.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(in_channels, in_channels * 2, kernels, padding=kernels // 2),
            nn.GLU(dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, N]], input 1D tensor.
        Returns:
            [torch.float32; [B, C, N]], transformed.
        """
        return inputs + self.conv(inputs)


class LinguisticEncoder(nn.Module):
    """Additional linguistic information encoder."""

    in_channels: int
    """The number of channels in the input and intermediate layers."""

    hidden_channels: int
    """The number of hidden channels."""

    preconv_blocks: int
    """The number of pre-convolution blocks."""

    convglu_kernel_sizes: List[int]
    """The sizes of the ConvGLU kernels."""

    leak: float
    """The negative slope of leaky ReLUs."""

    dropout_rate: float
    """The dropout rate."""

    wav2vec2: Wav2Vec2Wrapper
    """A wav2vec2 model for extracting linguistic features."""

    def __init__(
        self,
        wav2vec2: Wav2Vec2Wrapper,
        hidden_channels: int,
        preconv_blocks: int,
        convglu_kernel_sizes: List[int],
        leak: float,
        dropout_rate: float,
    ):
        """Initializer.
        Args:
            wav2vec2: a Wav2Vec2 model for extracting linguistic features.
            hidden_channels: The number of hidden channels.
            preconv_blocks: The number of pre-convolution blocks.
            convglu_kernel_sizes: The sizes of the ConvGLU kernels.
            leak: The negative slope of leaky ReLUs.
            dropout_rate: The dropout rate.
        """
        super().__init__()

        self.hidden_channels = hidden_channels
        self.preconv_blocks = preconv_blocks
        self.convglu_kernel_sizes = convglu_kernel_sizes
        self.leak = leak
        self.dropout_rate = dropout_rate

        self.wav2vec2 = wav2vec2
        self.in_channels = wav2vec2.channels

        # in_channels=1024, hidden_channels=128, preconv=2
        # unknown `leak`, `dropout`
        self.preconv = nn.Sequential(
            nn.Conv1d(self.in_channels, hidden_channels, 1),
            nn.LeakyReLU(leak),
            nn.Dropout(dropout_rate),
            *[
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, 1),
                    nn.LeakyReLU(leak),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(preconv_blocks - 1)
            ]
        )
        # kernels=[3] * 8 + [1] * 2
        self.convglu = nn.Sequential(
            *[
                ConvGLU(hidden_channels, kernel_size, dropout_rate)
                for kernel_size in convglu_kernel_sizes
            ]
        )

        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Filter the linguistic information.
        Args:
            inputs: [torch.float32; [B, in_channels, N]], input features.
        Returns:
            [torch.float32; [B, hidden_channels, N]], linguistic information.
        """
        # [B, S, w2v2_channels]
        w2v2 = self.wav2vec2(inputs)
        # [B, hidden_channels, N]
        x = self.preconv(w2v2.transpose(1, 2))
        # [B, hidden_channels, N]
        x = self.convglu(x)
        # [B, hidden_channels, N]
        return F.normalize(self.proj(x), dim=1)
