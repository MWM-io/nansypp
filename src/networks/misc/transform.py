from typing import Optional

import torch
import torchaudio
from torch import nn

from src.networks.misc.rough_cqt import CQT2010v2
from src.utilities.profiling import DO_PROFILING, cuda_synchronized_timer
from src.utilities.types import copy_docstring_and_signature


class ConstantQTransform(nn.Module):
    """Constant Q-Transform."""

    hop_length: int
    """The number of samples between adjacent frame."""

    fmin: float
    """The minimum frequency."""

    bins: int
    """The number of output bins."""

    bins_per_octave: int
    """The number of frequency bins per octave."""

    sample_rate: int
    """The sampling rate."""

    def __init__(
        self,
        hop_length: int,
        fmin: float,
        bins: int,
        bins_per_octave: int,
        sample_rate: int,
    ):
        """Initializer.
        Args:
            hop_length: The number of samples between adjacent frame.
            fmin: The minimum frequency.
            bins: The number of output bins.
            bins_per_octave: The number of frequency bins per octave.
            sample_rate: The sampling rate.
        """
        super().__init__()
        # unknown `hop_length`
        # , since linguistic information is 50fps, hop_length could be 441
        self.hop_length = hop_length
        # fmin=32.7(C0)
        self.fmin = fmin
        # bins=191, bins_per_octave=24
        # , fmax = 2 ** (bins / bins_per_octave) * fmin
        #        = 2 ** (191 / 24) * 32.7
        #        = 8132.89

        self.bins = bins
        self.bins_per_octave = bins_per_octave
        self.sample_rate = sample_rate

        self.cqt = CQT2010v2(
            sample_rate,
            hop_length,
            fmin,
            n_bins=bins,
            bins_per_octave=bins_per_octave,
            trainable=False,
            output_format="Magnitude",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply CQT on inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns:
            [torch.float32; [B, bins, T / hop_length]], CQT magnitudes.
        """
        return self.cqt(inputs[:, None])

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)


class MelSpectrogram(nn.Module):
    """log-Mel scale spectrogram."""

    hann: torch.Tensor
    melfilter: torch.Tensor

    def __init__(
        self,
        hop_length: int,
        window_length: int,
        mel: int,
        fmin: float,
        sample_rate: int,
        fmax: Optional[float] = None,
    ):
        """Initializer.
        Args:
            hop_length: Hop length, the number of frames between adjacent windows.
            window_length: The length of the window.
            mel: The size of the mel filterbanks.
            fmin: The minimum frequency.
            fmax: The maximum frequency,
                if None, uses half of the sample rate as default.
            sample_rate: The sample rate.
        """
        super().__init__()
        self.hop_length = hop_length
        self.window_length = window_length

        if fmax is None:
            fmax = sample_rate / 2
        # [mel, window_length // 2 + 1]
        # use slaney-scale mel filterbank for `librosa.filters.mel` compatibility.
        melfilter = torchaudio.functional.melscale_fbanks(
            window_length // 2 + 1,
            fmin,
            fmax,
            mel,
            sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).T

        self.register_buffer(
            "melfilter",
            melfilter,
            persistent=False,
        )
        # [window_length], use hann window
        self.register_buffer("hann", torch.hann_window(window_length), persistent=False)

    @cuda_synchronized_timer(do_profiling=DO_PROFILING, prefix="MelSpectrogram")
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Generate the log-mel scale spectrogram.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
        Returns:
            [torch.float32; [B, mel, T / hop_length]], log-mel spectrogram
        """
        # [B, window_length // 2 + 1, T / hop_length, 2]
        fft = torch.view_as_real(
            torch.stft(
                audio,
                self.window_length,
                self.hop_length,
                window=self.hann,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
        )
        # [B, window_length // 2 + 1, T / hop_length]
        mag = torch.sqrt(fft.square().sum(dim=-1) + 1e-7)
        # [B, mel, T / hop_length]
        return torch.log(torch.matmul(self.melfilter, mag) + 1e-7)

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)
