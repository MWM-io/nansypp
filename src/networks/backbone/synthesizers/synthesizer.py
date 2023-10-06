from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.networks.backbone.synthesizers.wavenet import WaveNet
from src.utilities.profiling import DO_PROFILING, cuda_synchronized_timer
from src.utilities.types import copy_docstring_and_signature


class SignalGenerator(nn.Module):
    """Additive sinusoidal, subtractive filtered noise signal generator."""

    def __init__(self, scale: int, input_sample_rate: int, output_sample_rate: int):
        """Initializer.
        Args:
            scale: upscaling factor.
            sample_rate: sampling rate.
        """
        super().__init__()
        self.output_sample_rate = output_sample_rate
        self.upsampler = nn.Upsample(
            scale_factor=scale * (output_sample_rate / input_sample_rate), mode="linear"
        )

    @cuda_synchronized_timer(DO_PROFILING, prefix="SignalGenerator")
    def forward(
        self,
        pitch: torch.Tensor,
        p_amp: torch.Tensor,
        ap_amp: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate the signal.
        Args:
            pitch: [torch.float32; [B, N]], frame-level pitch sequence.
            p_amp: [torch.float32; [B, N]], periodic amplitude.
            ap_amp: [torch.float32; [B, N]], aperiodic amplitude.
            noise: [torch.float32; [B, T]], predefined noise, if provided.
        Returns:
            [torch.float32; [B, T(=N x scale)]], base signal.
        """
        # [B, T]
        pitch = self.upsampler(pitch[:, None]).squeeze(dim=1)
        p_amp = self.upsampler(p_amp[:, None]).squeeze(dim=1)
        # [B, T]
        phase = torch.cumsum(2 * torch.pi * pitch / self.output_sample_rate, dim=-1)
        # [B, T]
        x = p_amp * torch.sin(phase)
        # [B, T]
        ap_amp = self.upsampler(ap_amp[:, None]).squeeze(dim=1)
        if noise is None:
            # [B, T], U[-1, 1] sampled
            noise = torch.rand_like(x) * 2.0 - 1.0
        # [B, T]
        y = ap_amp * noise
        return x + y

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)


class Synthesizer(nn.Module):
    """Signal-level synthesizer."""

    def __init__(
        self,
        scale: int,
        input_sample_rate: int,
        output_sample_rate: int,
        channels: int,
        aux: int,
        kernels: int,
        dilation_rate: int,
        layers: int,
        cycles: int,
    ):
        """Initializer.
        Args:
            scale: upscaling factor.
            sample_rate: sampling rate.
            channels: size of the hidden channels.
            aux: size of the auxiliary input channels.
            kernels: size of the convolutional kernels.
            dilation_rate: dilation rate.
            layers: the number of the wavenet blocks in single cycle.
            cycles: the number of the cycles.
        """
        super().__init__()
        self.excitation_generator = SignalGenerator(
            scale, input_sample_rate, output_sample_rate
        )

        self.wavenet = WaveNet(channels, aux, kernels, dilation_rate, layers, cycles)

    @cuda_synchronized_timer(DO_PROFILING, prefix="Synthesizer")
    def forward(
        self,
        pitch: torch.Tensor,
        p_amp: torch.Tensor,
        ap_amp: torch.Tensor,
        frame: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the signal.
        Args:
            pitch: [torch.float32; [B, N]], frame-level pitch sequence.
            p_amp, ap_amp: [torch.float32; [B, N]], periodical and aperiodical amplitudes.
            frame: [torch.float32; [B, aux, N']], frame-level feature map.
            noise: [torch.float32; [B, T]], predefined noise for excitation signal, if provided.
        Returns:
            [torch.float32; [B, T]], excitation signal and generated signal.
        """
        # [B, T]
        excitation = self.excitation_generator(pitch, p_amp, ap_amp, noise=noise)
        # [B, aux, T]
        interp = torch.nn.functional.interpolate(
            frame, size=excitation.shape[-1], mode="linear"
        )
        # [B, T]
        signal = self.wavenet(excitation, interp)
        return excitation, signal

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)
