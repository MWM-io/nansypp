from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from src.data.preprocessing.augmentation import RandomBackgroundNoise
from src.networks.misc.peq import ParametricEqualizer
from src.networks.misc.praat import PraatAugment
from src.utilities.profiling import DO_PROFILING, cuda_synchronized_timer
from src.utilities.types import copy_docstring_and_signature


class InformationPerturbator(nn.Module):
    """Waveform information perturbator.

    To be applied before linguistic features extraction in
    the training stage of the backbone architecture.
    """

    formant_shift: float
    pitch_shift: float
    pitch_range: float
    cutoff_lowpass: float
    cutoff_highpass: float
    q_min: float
    q_max: float
    num_peaks: int
    gain_range: float
    window: torch.Tensor
    peak_centers: torch.Tensor

    def __init__(
        self,
        formant_shift: float,
        pitch_shift: float,
        pitch_range: float,
        cutoff_lowpass: float,
        cutoff_highpass: float,
        q_min: float,
        q_max: float,
        num_peaks: int,
        gain_range: float,
        stft_window_length: int,
        stft_hop_length: int,
        praat_augment: PraatAugment,
        parametric_equalizer: ParametricEqualizer,
        additive_noise: Optional[RandomBackgroundNoise] = None,
    ):
        super().__init__()

        self.num_peaks = num_peaks
        self.formant_shift = formant_shift
        self.pitch_shift = pitch_shift
        self.pitch_range = pitch_range
        self.cutoff_lowpass = cutoff_lowpass
        self.cutoff_highpass = cutoff_highpass
        self.q_min = q_min
        self.q_max = q_max
        self.gain_range = gain_range
        self.stft_window_length = stft_window_length
        self.stft_hop_length = stft_hop_length

        self.praat_augment = praat_augment
        self.parametric_equalizer = parametric_equalizer
        self.additive_noise = additive_noise

        self.register_buffer(
            "window",
            torch.hann_window(self.stft_window_length),
            persistent=False,
        )
        f_min, f_max, peaks = (
            self.cutoff_lowpass,
            self.cutoff_highpass,
            self.num_peaks,
        )
        # peaks except frequency min and max
        self.register_buffer(
            "peak_centers",
            f_min * (f_max / f_min) ** (torch.arange(peaks + 2)[1:-1] / (peaks + 1)),
            persistent=False,
        )

    def augment(
        self,
        wavs: torch.Tensor,
        pitch_shift: Optional[torch.Tensor] = None,
        pitch_range: Optional[torch.Tensor] = None,
        formant_shift: Optional[torch.Tensor] = None,
        quality_power: Optional[torch.Tensor] = None,
        gain: Optional[torch.Tensor] = None,
        noises: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Augment the audio signal, random pitch, formant shift and PEQ.

        Augmentations are symmetric around the identity, ensuring that the unperturbed
        data remains within the model's training distribution.

        Args:
            wavs: [torch.float32; [B, T]], audio signal.
            pitch_shift: [torch.float32; [B]], pitch shifts.
            pitch_range: [torch.float32; [B]], pitch ranges.
            formant_shift: [torch.float32; [B]], formant shifts.
            quality_power: [torch.float32; [B, num_peaks + 2]],
                exponents of quality factor, for PEQ.
            gain: [torch.float32; [B, num_peaks + 2]], gain in decibel.
        Returns:
            [torch.float32; [B, T]], augmented.
        """
        # B
        bsize, _ = wavs.shape
        # [B, F, T / S], complex64
        fft = torch.stft(
            wavs,
            self.stft_window_length,
            self.stft_hop_length,
            self.stft_window_length,
            self.window,
            return_complex=True,
        )

        # PEQ
        if quality_power is not None:
            # [B, num_peaks + 2]
            q = self.q_min * (self.q_max / self.q_min) ** quality_power
            if gain is None:
                # [B, num_peaks]
                gain = torch.zeros_like(q[:, :-2])
            # [B, num_peaks]
            center = self.peak_centers[None].repeat(bsize, 1)
            # [B, F]
            peaks = torch.prod(
                self.parametric_equalizer.peaking_equalizer(
                    center, gain[:, :-2], q[:, :-2]
                ),
                dim=1,
            )
            # [B, F]
            lowpass = self.parametric_equalizer.low_shelving(
                self.cutoff_lowpass, gain[:, -2], q[:, -2]
            )
            highpass = self.parametric_equalizer.high_shelving(
                self.cutoff_highpass, gain[:, -1], q[:, -1]
            )
            # [B, F]
            filters = peaks * highpass * lowpass
            # [B, F, T / S]
            fft = fft * filters[..., None]

        # [B, T]
        out = torch.istft(
            fft,
            self.stft_window_length,
            self.stft_hop_length,
            self.stft_window_length,
            self.window,
        ).clamp(-1.0, 1.0)
        # max value normalization
        out = out / out.abs().amax(dim=-1, keepdim=True).clamp_min(1e-7)

        if (
            formant_shift is None
            and pitch_shift is None
            and pitch_range is None
            and noises is None
        ):
            return out

        # praat-based augmentation
        if formant_shift is None:
            formant_shift = torch.ones(bsize)
        if pitch_shift is None:
            pitch_shift = torch.ones(bsize)
        if pitch_range is None:
            pitch_range = torch.ones(bsize)
        out = torch.tensor(
            np.stack(
                [
                    self.praat_augment.augment(o, fs.item(), ps.item(), pr.item())
                    for o, fs, ps, pr in zip(
                        out.cpu().numpy(),
                        formant_shift.cpu().numpy(),
                        pitch_shift.cpu().numpy(),
                        pitch_range.cpu().numpy(),
                    )
                ],
                axis=0,
            ),
            device=out.device,
            dtype=torch.float32,
        )
        if self.additive_noise is not None:
            out = self.additive_noise(out, noises=noises)
        return out

    def sample_like(
        self, signal: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample augmentation parameters.
        Args:
            signal: [torch.float32; [B, T]], speech signal.
        Returns:
            augmentation parameters.
        """
        # [B]
        batch_size, audio_length = signal.shape

        def sampler(ratio: float) -> torch.Tensor:
            shifts = torch.rand(batch_size, device=signal.device) * (ratio - 1.0) + 1.0
            # flip
            flip = torch.rand(batch_size) < 0.5
            shifts[flip] = shifts[flip] ** -1
            return shifts

        # sample shifts
        formant_shifts = sampler(self.formant_shift)
        pitch_shifts = sampler(self.pitch_shift)
        pitch_ranges = sampler(self.pitch_range)
        # parametric equalizer
        peaks = self.num_peaks
        # quality factor
        power = torch.rand(batch_size, peaks + 2, device=signal.device)
        # gains
        g_min, g_max = -self.gain_range, self.gain_range
        gain = (
            torch.rand(batch_size, peaks + 2, device=signal.device) * (g_max - g_min)
            + g_min
        )
        # additive noise
        noise = None
        if self.additive_noise is not None:
            noise = self.additive_noise.sample(batch_size, audio_length).to(
                signal.device
            )
        return formant_shifts, pitch_shifts, pitch_ranges, power, gain, noise

    @cuda_synchronized_timer(DO_PROFILING, prefix="Synthesizer")
    @torch.no_grad()
    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        """Augment the speech.
        Args:
            wavs: [torch.float32; [B, T]], segmented speech.
        Returns:
            [torch.float32; [B, T]], speech signal.
        """
        # B
        bsize, _ = wavs.shape
        saves = None
        while saves is None or len(saves) < bsize:
            # [B] x 4
            (
                formant_shifts,
                pitch_shifts,
                pitch_ranges,
                power,
                gain,
                noises,
            ) = self.sample_like(wavs)
            # [B, T]
            out = self.augment(
                wavs, pitch_shifts, pitch_ranges, formant_shifts, power, gain, noises
            )

            # handle unexpected NaNs
            nan = out.isnan().any(dim=-1)
            if not nan.all():
                # save the outputs for not-nan inputs
                if saves is None:
                    saves = out[~nan]
                else:
                    saves = torch.cat([saves, out[~nan]], dim=0)

        # [B, T]
        return saves[:bsize]

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)
