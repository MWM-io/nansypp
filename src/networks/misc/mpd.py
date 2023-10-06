from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.utilities.profiling import DO_PROFILING, cuda_synchronized_timer
from src.utilities.types import copy_docstring_and_signature


class PeriodDiscriminator(nn.Module):
    """Period-aware discriminator."""

    channels: List[int]
    """List of the channel sizes."""
    period: int
    """Size of the unit period."""
    kernels: int
    """Size of the convolutional kernels."""
    stride: int
    """Stride of the convolutions."""
    postkernels: int
    """Size of the postnet convolutional kernels."""
    leak: float
    """Negative slope of leaky ReLUs."""

    def __init__(
        self,
        channels: List[int],
        period: int,
        kernels: int,
        stride: int,
        postkernels: int,
        leak: float,
    ):
        """Initializer.
        Args:
            channels: list of the channel sizes.
            period: size of the unit period.
            kernels: size of the convolutional kernels.
            stride: stride of the convolutions.
            postkernels: size of the postnet convolutional kernels.
            leak: negative slope of leaky ReLU.
        """
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv2d(
                            inc,
                            outc,
                            (kernels, 1),
                            (stride, 1),
                            padding=(kernels // 2, 0),
                        )
                    ),
                    nn.LeakyReLU(leak),
                )
                for inc, outc in zip([1] + channels, channels)
            ]
        )

        lastc = channels[-1]
        self.convs.append(
            nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv2d(lastc, lastc, (kernels, 1), padding=(kernels // 2, 0))
                ),
                nn.LeakyReLU(leak),
            )
        )

        self.postconv = nn.utils.weight_norm(
            nn.Conv2d(lastc, 1, (postkernels, 1), padding=(postkernels // 2, 0))
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Discriminate the inputs in multiple periods.
        Args:
            x: [torch.float32; [B, T]], input audio signal.
        Returns:
            outputs: [torch.float32; [B, S]], logits.
            feature_maps: [torch.float32; [B, C, F, P]], list of feature maps.
        """
        # B, T
        bsize, timestep = inputs.shape
        if timestep % self.period != 0:
            # padding for foldability
            padsize = self.period - timestep % self.period
            # [B, T + R]
            inputs = F.pad(inputs[:, None], (0, padsize), "reflect").squeeze(1)
            # T + R
            timestep = timestep + padsize
        # [B, 1, F(=T // P), P]
        x = inputs.view(bsize, 1, timestep // self.period, self.period)

        # period-aware discriminator
        feature_maps: List[torch.Tensor] = []
        for conv in self.convs:
            x = conv(x)
            feature_maps.append(x)
        # [B, 1, S', P]
        x = self.postconv(x)
        feature_maps.append(x)

        # [B, S]
        return x.view(bsize, -1), feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """MPD: Multi-period discriminator."""

    channels: List[int]
    """List of the channel sizes. (Common to all individual discriminators.)"""
    periods: List[int]
    """Sizes of the respective unit period for each individual discriminator."""
    kernels: int
    """Size of the convolutional kernel. (Common to all individual discriminators.)"""
    stride: int
    """Stride of the convolution. (Common to all individual discriminators.)"""
    postkernels: int
    """Size of the postnet convolutional kernel. (Common to all individual discriminators.)"""
    leak: float
    """Negative slope of leaky ReLUs. (Common to all individual discriminators.)"""

    def __init__(
        self,
        periods: List[int],
        channels: List[int],
        kernels: int,
        stride: int,
        postkernels: int,
        leak: float,
    ):
        """Initializer."""
        super().__init__()
        self.periods = periods
        self.channels = channels
        self.kernels = kernels
        self.stride = stride
        self.postkernels = postkernels
        self.leak = leak

        self.discriminators = nn.ModuleList(
            [
                PeriodDiscriminator(
                    channels=list(channels),
                    period=period,
                    kernels=kernels,
                    stride=stride,
                    postkernels=postkernels,
                    leak=leak,
                )
                for period in list(periods)
            ]
        )

    @cuda_synchronized_timer(DO_PROFILING, prefix="MultiPeriodDiscriminator")
    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Discriminate the samples from real or fake.
        Args:
            x: [B, T], audio sample.
        Returns:
            multiple discriminating results and feature maps.
        """
        results: List[torch.Tensor] = []
        features_maps_per_period: List[List[torch.Tensor]] = []

        for discriminator in self.discriminators:
            result, features_maps = discriminator(inputs)
            results.append(result)
            features_maps_per_period.append(features_maps)

        return results, features_maps_per_period

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)
