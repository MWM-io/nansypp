from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    """Residual block,"""

    def __init__(self, in_channels: int, out_channels: int, kernels: int):
        """Initializer.
        Args:
            in_channels: size of the input channels.
            out_channels: size of the output channels.
            kernels: size of the convolutional kernels.
        """
        super().__init__()
        self.branch = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels, out_channels, (kernels, 1), padding=(kernels // 2, 0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels, out_channels, (kernels, 1), padding=(kernels // 2, 0)
            ),
        )

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, in_channels, F, N]], input channels.
        Returns:
            [torch.float32; [B, out_channels, F // 2, N]], output channels.
        """
        # [B, out_channels, F, N]
        outputs = self.branch(inputs)
        # [B, out_channels, F, N]
        shortcut = self.shortcut(inputs)
        # [B, out_channels, F // 2, N]
        return F.avg_pool2d(outputs + shortcut, (2, 1))


def exponential_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Exponential sigmoid.
    Args:
        x: [torch.float32; [...]], input tensors.
    Returns:
        sigmoid outputs.
    """
    return 2.0 * torch.sigmoid(x) ** np.log(10) + 1e-7


class PitchEncoder(nn.Module):
    """Pitch-encoder."""

    freq: int
    """Number of frequency bins."""
    min_pitch: float
    """The minimum predicted pitch."""
    max_pitch: float
    """The maximum predicted pitch."""
    prekernels: int
    """Size of the first convolutional kernels."""
    kernels: int
    """Size of the frequency-convolution kernels."""
    channels: int
    """Size of the channels."""
    blocks: int
    """Number of residual blocks."""
    gru_dim: int
    """Size of the GRU hidden states."""
    hidden_channels: int
    """Size of the hidden channels."""
    f0_bins: int
    """Size of the output f0-bins."""
    f0_activation: str
    """F0 activation function."""

    def __init__(
        self,
        freq: int,
        min_pitch: float,
        max_pitch: float,
        prekernels: int,
        kernels: int,
        channels: int,
        blocks: int,
        gru_dim: int,
        hidden_channels: int,
        f0_bins: int,
        f0_activation: str,
    ):
        """Initializer.
        Args:
            freq: Number of frequency bins.
            min_pitch: The minimum predicted pitch.
            max_pitch: The maximum predicted pitch.
            prekernels: Size of the first convolutional kernels.
            kernels: Size of the frequency-convolution kernels.
            channels: Size of the channels.
            blocks: Number of residual blocks.
            gru_dim: Size of the GRU hidden states.
            hidden_channels: Size of the hidden channels.
            f0_bins: Size of the output f0-bins.
            f0_activation: f0 activation function.
        """
        super().__init__()

        self.freq = freq
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.prekernels = prekernels
        self.kernels = kernels
        self.channels = channels
        self.blocks = blocks
        self.gru_dim = gru_dim
        self.hidden_channels = hidden_channels
        self.f0_bins = f0_bins
        self.f0_activation = f0_activation

        assert f0_activation in ["softmax", "sigmoid", "exp_sigmoid"]

        # prekernels=7
        self.preconv = nn.Conv2d(
            1, channels, (prekernels, 1), padding=(prekernels // 2, 0)
        )
        # channels=128, kernels=3, blocks=2
        self.resblock = nn.Sequential(
            *[ResBlock(channels, channels, kernels) for _ in range(blocks)]
        )
        # unknown `gru`
        self.gru = nn.GRU(
            freq * channels // (2 * blocks),
            gru_dim,
            batch_first=True,
            bidirectional=True,
        )
        # unknown `hidden_channels`
        # f0_bins=64
        self.proj = nn.Sequential(
            nn.Linear(gru_dim * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, f0_bins + 2),
        )

        self.register_buffer(
            "pitch_bins",
            # linear space in log-scale
            torch.linspace(
                np.log(self.min_pitch),
                np.log(self.max_pitch),
                self.f0_bins,
            ).exp(),
        )

    def forward(
        self, cqt_slice: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the pitch from inputs.
        Args:
            cqt_slice: [torch.float32; [B, F, N]]
                The input tensor. A frequency-axis slice of a Constant-Q Transform.
        Returns:
            pitch: [torch.float32; [B, N]], predicted pitch, based on frequency bins.
            f0_logits: [torch.float32; [B, N, f0_bins]], predicted f0 activation weights,
                based on the pitch bins.
            p_amp, ap_amp: [torch.float32; [B, N]], amplitude values.
        """
        # B, _, N
        batch_size, _, timesteps = cqt_slice.shape
        # [B, C, F, N]
        x = self.preconv(cqt_slice[:, None])
        # [B, C F // 4, N]
        x = self.resblock(x)
        # [B, N, C x F // 4]
        x = x.permute(0, 3, 1, 2).reshape(batch_size, timesteps, -1)
        # [B, N, G x 2]
        x, _ = self.gru(x)
        # [B, N, f0_bins], [B, N, 1], [B, N, 1]
        f0_weights, p_amp, ap_amp = torch.split(
            self.proj(x), [self.f0_bins, 1, 1], dim=-1
        )

        # potentially apply activation function
        if self.f0_activation == "softmax":
            f0_weights = torch.softmax(f0_weights, dim=-1)
        elif self.f0_activation == "sigmoid":
            f0_weights = torch.sigmoid(f0_weights) / torch.sigmoid(f0_weights).sum(
                dim=-1, keepdim=True
            )
        elif self.f0_activation == "exp_sigmoid":
            f0_weights = exponential_sigmoid(f0_weights) / exponential_sigmoid(
                f0_weights
            ).sum(dim=-1, keepdim=True)

        # [B, N]
        pitch = (f0_weights * self.pitch_bins).sum(dim=-1)

        return (
            pitch,
            f0_weights,
            exponential_sigmoid(p_amp).squeeze(dim=-1),
            exponential_sigmoid(ap_amp).squeeze(dim=-1),
        )
