from typing import List

import torch
from torch import nn


class ConvReLUNorm(nn.Module):
    """
    Convolutional block with ReLU activations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        # normalized_shape: List[int],
        dropout: float,
        **kwargs,
    ):
        """Initializer.
        Args:
            in_channels:
            out_channels:
            kernel_size:
            normalized_shape
            dropout:
        """
        super().__init__()
        # Conv1D(k, d, s)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )
        # LeakyRelu
        self.relu = nn.ReLU()
        # LayerNorm
        self.layer_norm = nn.LayerNorm([out_channels])
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: [B, C, N]
        Returns:
            output: [B, C, N/S]
        """
        conv_features = self.conv1d(embedding)
        relu = self.relu(conv_features)
        layer_normed = self.layer_norm(relu.transpose(1, 2)).transpose(1, 2)
        dropped_out: torch.Tensor = self.dropout(layer_normed)
        return dropped_out
