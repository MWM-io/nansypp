from typing import Optional, Tuple, TypedDict, Union

import torch
import torch.nn.functional as F
from torch import nn

from src.networks.backbone.encoders.linguistic import LinguisticEncoder
from src.networks.backbone.encoders.pitch import PitchEncoder
from src.networks.backbone.encoders.timbre import TimbreEncoder
from src.networks.backbone.synthesizers.framelevel import FrameLevelSynthesizer
from src.networks.backbone.synthesizers.synthesizer import Synthesizer
from src.networks.misc.perturbator import InformationPerturbator
from src.networks.misc.transform import ConstantQTransform, MelSpectrogram
from src.utilities.profiling import DO_PROFILING, cuda_synchronized_timer
from src.utilities.types import copy_docstring_and_signature


class AnalysisFeatures(TypedDict):
    cqt: torch.Tensor

    pitch: torch.Tensor
    ap_amp: torch.Tensor
    p_amp: torch.Tensor

    linguistic: torch.Tensor

    timbre_global: Optional[torch.Tensor]
    timbre_bank: Optional[torch.Tensor]


class SynthesisFeatures(TypedDict):
    excitation: torch.Tensor


class Generator(nn.Module):
    input_sample_rate: int
    output_sample_rate: int
    leak: float
    dropout_rate: float
    pitch_start: float
    pitch_end: float

    def __init__(
        self,
        input_sample_rate: int,
        output_sample_rate: int,
        leak: float,
        dropout_rate: float,
        cqt_center: int,
        linguistic_encoder: LinguisticEncoder,
        cqt: ConstantQTransform,
        pitch_encoder: PitchEncoder,
        mel_spectrogram_transform: MelSpectrogram,
        synthesizer: Synthesizer,
        frame_level_synthesizer: FrameLevelSynthesizer,
        information_perturbator: InformationPerturbator,
        timbre_encoder: Optional[TimbreEncoder],
    ):
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.leak = leak
        self.dropout_rate = dropout_rate

        # dependency-injection all the way
        self.linguistic_encoder = linguistic_encoder
        self.cqt = cqt
        self.pitch_encoder = pitch_encoder
        self.mel_spectrogram_transform = mel_spectrogram_transform
        self.timbre_encoder = timbre_encoder
        self.frame_level_synthesizer = frame_level_synthesizer
        self.information_perturbator = information_perturbator
        self.synthesizer = synthesizer

        self.cqt_center = cqt_center

    @cuda_synchronized_timer(DO_PROFILING, prefix="Generator")
    def analyze_pitch(
        self,
        inputs: torch.Tensor,
        index: Optional[int] = None,
        precomputed_cqt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate the pitch and periodical, aperiodical amplitudes.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
            index: CQT start index. If None, use `cqt_center`.
        Returns:
            [torch.float32; [B, cqt_bins, N]], CQT features.
            [torch.float2; [B, N]], frame-level pitch and amplitude sequence.
        """
        # TODO(@revsic): use log-scaled CQT or not?
        if precomputed_cqt is None:
            # [B, cqt_bins, N(=T / cqt_hop)]
            cqt = self.cqt(inputs)
        else:
            cqt = precomputed_cqt

        # alias
        freq = self.pitch_encoder.freq
        if index is None:
            index = self.cqt_center

        # [B, N], [B, N, f0_bins], [B, N], [B, N]
        pitch, _, p_amp, ap_amp = self.pitch_encoder(cqt[:, index : index + freq])

        # [B, cqt_bins, N], [B, N]
        return cqt, pitch, p_amp, ap_amp

    @cuda_synchronized_timer(DO_PROFILING, prefix="Generator")
    def analyze_linguistic(
        self,
        inputs: torch.Tensor,
        enable_information_perturbator: bool = False,
        perturbed_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Analyze the linguistic informations from inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
            enable_information_perturbator: bool
                Whether to apply linguistic-features-preserving data augmentations.
                Used during training only, do not enable at inference time.
        Returns:
            [torch.float32; [B, linguistic_hidden_channels, S]], linguistic informations.
        """
        if enable_information_perturbator and perturbed_inputs is None:
            inputs = self.information_perturbator(inputs)
        elif perturbed_inputs is not None:
            inputs = perturbed_inputs
        # [B, linguistic_hidden_channels, S]
        return self.linguistic_encoder(inputs)

    @cuda_synchronized_timer(DO_PROFILING, prefix="Generator")
    def analyze_timbre(
        self, inputs: torch.Tensor, precomputed_mel: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:
        """Analyze the timbre informations from inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns:
            [torch.float32; [B, timb_global]], global timbre emebddings.
            [torch.float32; [B, timb_timbre, timb_tokens]], timbre token bank.
        """
        if self.timbre_encoder is not None:
            # [B, timb_global], [B, timb_timbre, timb_tokens]
            return self.timbre_encoder(inputs, precomputed_mel=precomputed_mel)
        else:
            return None, None

    @cuda_synchronized_timer(DO_PROFILING, prefix="Generator")
    def analyze(
        self,
        inputs: torch.Tensor,
        enable_information_perturbator: bool,
        perturbed_inputs: Optional[torch.Tensor] = None,
        precomputed_cqt: Optional[torch.Tensor] = None,
        precomputed_mel: Optional[torch.Tensor] = None,
    ) -> AnalysisFeatures:
        """Analyze the input signal.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
            enable_information_perturbator: bool
                Whether to apply linguistic-features-preserving data augmentations.
                Used during training only, do not enable at inference time.
        Returns;
            Analysis features:
                cqt: [torch.float32; []], CQT features.
                pitch, p_amp, ap_amp: [torch.float2; [B, N]],
                    frame-level pitch and amplitude sequence.
                linguistic: [torch.float32; [B, linguistic_hidden_channels, S]], linguistic informations.
                timbre_global: [torch.float32; [B, timb_global]], global timbre emebddings.
                timbre_bank: [torch.float32; [B, timb_timbre, timb_tokens]], timbre token bank.
        """
        # [], [B, N]
        cqt, pitch, p_amp, ap_amp = self.analyze_pitch(
            inputs, precomputed_cqt=precomputed_cqt
        )
        # [B, linguistic_hidden_channels, S]
        linguistic_features = self.analyze_linguistic(
            inputs, enable_information_perturbator, perturbed_inputs
        )
        # [B, timb_global], [B, timb_timbre, timb_tokens]
        timbre_global, timbre_bank = self.analyze_timbre(
            inputs, precomputed_mel=precomputed_mel
        )

        return {
            "cqt": cqt,
            "pitch": pitch,
            "p_amp": p_amp,
            "ap_amp": ap_amp,
            "linguistic": linguistic_features,
            "timbre_global": timbre_global,
            "timbre_bank": timbre_bank,
        }

    @cuda_synchronized_timer(DO_PROFILING, prefix="Generator")
    def synthesize(
        self,
        pitch: torch.Tensor,
        p_amp: torch.Tensor,
        ap_amp: torch.Tensor,
        linguistic: torch.Tensor,
        timbre_global: Optional[torch.Tensor],
        timbre_bank: Optional[torch.Tensor],
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize the signal.
        Args:
            pitch, p_amp, ap_amp: [torch.float32; [B, N]], frame-level pitch, amplitude sequence.
            linguistic: [torch.float32; [B, linguistic_hidden_channels, S]], linguistic features.
            timbre_global: [torch.float32; [B, timb_global]], global timbre.
            timbre_bank: [torch.float32; [B, timb_timbre, timb_tokens]], timbre token bank.
            noise: [torch.float32; [B, T]], predefined noise for excitation signal, if provided.
        Returns:
            [torch.float32; [B, T]], excitation and synthesized speech signal.
        """
        # S
        ling_len = linguistic.shape[-1]
        # [B, 3, S]
        pitch_rel = F.interpolate(
            torch.stack([pitch, p_amp, ap_amp], dim=1), size=ling_len
        )
        if self.timbre_encoder is not None:
            assert (
                timbre_global is not None and timbre_bank is not None
            ), "timbre_global and timbre_bank should be provided if a non-null timbre_encoder is used"

            # [B, 3 + linguistic_hidden_channels + timb_global, S]
            contents = torch.cat(
                [
                    pitch_rel,
                    linguistic,
                    timbre_global[..., None].repeat(1, 1, ling_len),
                ],
                dim=1,
            )
            # [B, timbre_global, S]
            timbre_sampled = self.timbre_encoder.sample_timbre(
                contents, timbre_global, timbre_bank
            )
        else:
            timbre_sampled = None

        # [B, linguistic_hidden_channels, S]
        frame = self.frame_level_synthesizer(linguistic, timbre_sampled)
        # [B, T], [B, T]
        return self.synthesizer(pitch, p_amp, ap_amp, frame, noise)

    @cuda_synchronized_timer(DO_PROFILING, prefix="Generator")
    def forward(
        self,
        inputs: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        enable_information_perturbator: bool = True,
        perturbed_inputs: Optional[torch.Tensor] = None,
        precomputed_cqt: Optional[torch.Tensor] = None,
        precomputed_mel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, AnalysisFeatures, SynthesisFeatures]:
        """Reconstruct input audio.
        Args:
            inputs: [torch.float32; [B, T]], input signal.
            noise: [torch.float32; [B, T]], predefined noise for excitation, if provided.
            enable_information_perturbator: bool
                Whether to apply linguistic-features-preserving data augmentations.
                Used during training only, do not enable at inference time.
        Returns:
            [torch.float32; [B, T]], reconstructed.
            auxiliary outputs, reference `Nansypp.analyze`.
        """
        try:
            _, timesteps = inputs.shape
        except AttributeError:
            inputs, perturbed_inputs, _ = inputs
            _, timesteps = inputs.shape

        analysis_features = self.analyze(
            inputs,
            enable_information_perturbator,
            perturbed_inputs=perturbed_inputs,
            precomputed_cqt=precomputed_cqt,
        )

        # [B, T]
        excitation, synth = self.synthesize(
            analysis_features["pitch"],
            analysis_features["p_amp"],
            analysis_features["ap_amp"],
            analysis_features["linguistic"],
            analysis_features["timbre_global"],
            analysis_features["timbre_bank"],
            noise=noise,
        )
        # truncate
        synth = synth[
            :, : int(timesteps / self.input_sample_rate * self.output_sample_rate)
        ]
        # update
        synthesis_features = SynthesisFeatures(excitation=excitation)
        return synth, analysis_features, synthesis_features

    @torch.no_grad()
    def voice_conversion(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
        device: torch.device,
        f0_statistics: bool = True,
        enable_information_perturbator: bool = False,
    ) -> torch.Tensor:
        """Apply target audio voice timbre to source audio content.
        Args:
            source_audio: [torch.Tensor; [1, Ns]] tensor with mono audio content at sampling rate defined by model config.
            target_audio: [torch.Tensor; [1, Nt]] tensor with mono audio content at sampling rate defined by model config.
            device: [torch.device] device, CPU or GPU, on which computation are expected to run.
            f0_statistics: [boolean], whether to replace statistics of estimated f0 of the source (mean, std) with those of the target.
            enable_information_perturbator: bool
                Whether to apply linguistic-features-preserving data augmentations.
                Used during training only, do not enable at inference time.
        Return:
            f0_corrected_synth: [torch.Tensor; [1, Ns]] tensor with mono audio content at sampling rate defined by model config
        """
        source_features = self.analyze(
            source_audio.to(device),
            enable_information_perturbator=enable_information_perturbator,
        )
        target_features = self.analyze(
            target_audio.to(device),
            enable_information_perturbator=enable_information_perturbator,
        )

        # Transform the F0 statistics of the source speaker into the target speakers'
        source_pitch = source_features["pitch"]
        if f0_statistics:
            standardized_source_pitch = (
                source_pitch - source_pitch.mean()
            ) / source_pitch.std()
            source_pitch = (
                standardized_source_pitch * target_features["pitch"].std()
                + target_features["pitch"].mean()
            )

        noise = None
        _, f0_corrected_synth = self.synthesize(
            source_pitch,
            source_features["p_amp"],
            source_features["ap_amp"],
            source_features["linguistic"],
            target_features.get("timbre_global", None),
            target_features.get("timbre_bank", None),
            noise=noise,
        )
        return f0_corrected_synth

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)
