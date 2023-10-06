import itertools
import os
from abc import abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pyrootutils
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from matplotlib.figure import Figure
from torch import nn
from torch.optim import Optimizer

from src.networks.backbone.encoders.pitch import PitchEncoder
from src.networks.backbone.generator import Generator
from src.networks.misc.mpd import MultiPeriodDiscriminator
from src.networks.misc.transform import MelSpectrogram
from src.utilities.crepe import crepe_pitch_estimation
from src.utilities.profiling import DO_PROFILING, cuda_synchronized_timer
from src.utilities.types import copy_docstring_and_signature

LOG_IDX = 0
LOG_MAXLEN = 1.5
LOG_AUDIO = 3
LOG_PLOTS = 2

GeneratorLossMetrics = Dict[str, Union[float, torch.Tensor]]
GeneratorLossImages = Dict[str, np.ndarray]
GeneratorLossAudios = Dict[str, np.ndarray]
GeneratorLossPlots = Dict[str, Figure]

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)


class ContrastiveMetrics(TypedDict):
    cont_loss: torch.Tensor
    metric_pos: torch.Tensor
    metric_neg: torch.Tensor


class AdversarialDiscriminativeLosses(TypedDict):
    d_fake: torch.Tensor
    features_matching_loss: torch.Tensor


class ContrastiveLoss(nn.Module):
    """Contrastive loss.

    Introduced in Kaizhi Qian et al., _Contentvec: An improved self-supervised speech representation by disentangling speakers_
    """

    def __init__(
        self,
        num_candidates: int,
        negative_samples_minimum_distance_to_positive: int,
        temperature: float,
    ) -> None:
        super().__init__()

        self.negative_samples_minimum_distance_to_positive = (
            negative_samples_minimum_distance_to_positive
        )
        self.temperature = temperature
        self.num_candidates = num_candidates

    def make_negative_sampling_mask(
        self, num_items: int, device: torch.device
    ) -> torch.Tensor:
        upper_triu_mask = torch.triu(
            torch.ones(num_items, num_items),
            self.negative_samples_minimum_distance_to_positive + 1,
        )
        all_mask = upper_triu_mask.T + upper_triu_mask
        random_values = all_mask * torch.rand(num_items, num_items)
        k_th_quant = torch.topk(random_values, min(self.num_candidates, num_items))[0][
            :, -1:
        ]
        random_mask = (random_values >= k_th_quant) * all_mask + torch.eye(num_items)
        return random_mask.to(device)

    @cuda_synchronized_timer(DO_PROFILING, prefix="ContrastiveLoss")
    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [2, B, linguistic_hidden_channels, S], normalize for cosine similarity.
        cosine_similarity = F.normalize(
            torch.stack(
                [x, y],
                dim=0,
            ),
            p=2,
            dim=2,
        )

        # N
        num_tokens = cosine_similarity.shape[-1]
        # [B, N]
        positive = cosine_similarity.prod(dim=0).sum(dim=1) / self.temperature
        # [2, B, N, N]
        confusion_matrix = (
            torch.matmul(cosine_similarity.transpose(2, 3), cosine_similarity)
            / self.temperature
        )

        # [N, N]
        negative_sampling_mask = self.make_negative_sampling_mask(num_tokens, x.device)

        # [2, B, N, N(sum = candidates)], negative case
        masked_confusion_matrix = confusion_matrix.masked_fill(
            ~negative_sampling_mask.to(torch.bool), -np.inf
        )
        # [2, B, N], negative case
        negative = masked_confusion_matrix.exp().sum(dim=-1)
        # []
        contrastive_loss = (
            -torch.sum(positive / negative, dim=2).log().sum(dim=0).mean()
        )
        mean_positive = positive.mean() * self.temperature
        mean_negative = (
            (confusion_matrix * negative_sampling_mask).sum(dim=-1)
            / self.num_candidates
        ).mean() * self.temperature

        return contrastive_loss, mean_positive, mean_negative

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)


Features = Dict[str, torch.Tensor]


class ReconstructionLossWithFeatures(nn.Module):
    @abstractmethod
    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Features]:
        ...

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)


class MelL1Loss(ReconstructionLossWithFeatures):
    """Mel spectrogram loss.

    As described in Kong et al., 2020., _Neural source-filter waveform models for sta- tistical parametric speech synthesis._
    """

    def __init__(
        self,
        mel_spectrogram_transform: MelSpectrogram,
    ) -> None:
        super().__init__()
        self.mel_spectrogram_transform = mel_spectrogram_transform

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Features]:
        mel_reconstructed = self.mel_spectrogram_transform(input)
        mel_original = self.mel_spectrogram_transform(target)

        mel_features = {
            "mel_reconstructed": mel_reconstructed.detach(),
            "mel_original": mel_original.detach(),
        }
        return F.l1_loss(mel_reconstructed, mel_original), mel_features


class MultiScaleFFTLoss(ReconstructionLossWithFeatures):
    """Multi-scale spectrogram loss (MSS).

    As described in Wang et al., 2019, _Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis._
    """

    windows: Iterable[nn.Parameter]

    def __init__(
        self,
        hop_sizes: List[int],
        window_sizes: List[int],
    ) -> None:
        super().__init__()
        self.hop_sizes = hop_sizes
        self.window_sizes = window_sizes
        self.windows = nn.ParameterList(
            [
                nn.Parameter(torch.hann_window(window_size), requires_grad=False)
                for window_size in self.window_sizes
            ]
        )

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Features]:
        # [B x 2, T]
        mss_loss = input.new_zeros(1)

        # batch-process input and target together
        input_and_target = torch.cat([input, target], dim=0)
        for hop_size, window_size, window in zip(
            self.hop_sizes, self.window_sizes, self.windows
        ):
            # [B x 2, win // 2 + 1, T / hop]
            stft = torch.stft(
                input_and_target,
                window_size,
                hop_size,
                window=window,
                return_complex=True,
            )
            # [B, win // 2 + 1, T / hop]
            mag_stft_input, mag_stft_target = stft.abs().chunk(2, dim=0)
            # []
            mss_loss = mss_loss + F.mse_loss(mag_stft_input, mag_stft_target)
        return mss_loss, {}


class AggregatedReconstructionLossWithFeatures(ReconstructionLossWithFeatures):
    def __init__(self, losses: List[ReconstructionLossWithFeatures]) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses)

    @cuda_synchronized_timer(DO_PROFILING, prefix="AggregatedReconstructionLoss")
    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Features]:
        aggregated_loss = input.new_zeros(1)
        aggregated_features: Features = {}
        for loss_function in self.losses:
            loss, features = loss_function(input, target)
            aggregated_loss += loss
            aggregated_features.update(features)
        return aggregated_loss, aggregated_features


class RelativePitchDifferenceLoss(nn.Module):
    """Relative pitch difference loss.

    Introduced in Gfeller et al., 2020, _Spice: Self-supervised pitch estimation_
    """

    def __init__(
        self,
        cqt_center: int,
        cqt_shift_min: int,
        cqt_shift_max: int,
        pitch_freq: int,
        delta: float,
        sigma: float,
    ) -> None:
        super().__init__()
        self.cqt_center = cqt_center
        self.cqt_shift_min = cqt_shift_min
        self.cqt_shift_max = cqt_shift_max
        self.pitch_freq = pitch_freq
        self.delta = delta
        self.sigma = sigma

    @cuda_synchronized_timer(DO_PROFILING, prefix="RelativePitchDifferenceLoss")
    def forward(
        self,
        cqt: torch.Tensor,
        pitch: torch.Tensor,
        pitch_encoder: PitchEncoder,
        bsize: int,
    ) -> torch.Tensor:
        random_transposition = torch.randint(
            self.cqt_shift_min,
            self.cqt_shift_max + 1,  # for inclusive range
            (bsize,),
            device=cqt.device,
        )
        # real start index
        start = random_transposition + self.cqt_center
        # sampled
        transposed_cqt = torch.stack(
            [cqt_[i : i + self.pitch_freq] for cqt_, i in zip(cqt, start)]
        )
        # [B, N]
        predicted_transposed_pitch, _, _, _ = pitch_encoder(transposed_cqt)

        # pitch consistency
        return F.huber_loss(
            predicted_transposed_pitch.log2()
            + self.sigma * random_transposition[:, None],
            pitch.log2(),
            delta=self.delta,
        )

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)


class PitchSupervisionLoss(nn.Module):
    """Mean-square error loss between predicted pitches and ground truth pitches."""

    def __init__(self, sample_rate: int, cqt_hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.cqt_hop_length = cqt_hop_length

    def forward(
        self, audios: torch.Tensor, cqts: torch.Tensor, pred_pitches: torch.Tensor
    ) -> torch.Tensor:
        device = pred_pitches.device
        gt_pitches = []
        remove_pred_indices = []
        for idx, audio in enumerate(audios):
            gt_pitch = crepe_pitch_estimation(
                audio.unsqueeze(0),
                self.sample_rate,
                self.cqt_hop_length,
                device,
            )
            # handle crepe potential crash
            if gt_pitch is not None:
                gt_pitches.append(
                    crepe_pitch_estimation(
                        audio.unsqueeze(0),
                        self.sample_rate,
                        self.cqt_hop_length,
                        device,
                    )
                )
            else:
                remove_pred_indices.append(idx)
        gt_pitches = torch.cat(gt_pitches, dim=0).to(pred_pitches.device)[
            ..., : cqts.shape[-1]
        ]
        # delete in reverse order to not throw off subsequent indexes in list
        for idx in sorted(remove_pred_indices, reverse=True):
            del pred_pitches[indx]
        return nn.functional.mse_loss(pred_pitches, gt_pitches)

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)


class GeneratorLoss(nn.Module):
    """Class to compute losses of generator modules."""

    def __init__(
        self,
        input_sample_rate: int,
        output_sample_rate: int,
        segment_length_s: float,
        cqt_hop_length: int,
        cqt_bins_per_octave: int,
        linguistic_loss_start_weight: float,
        linguistic_loss_end_weight: int,
        non_scalar_logging_steps: int,
        vc_source_audio_paths: Dict[str, str],
        vc_target_audio_paths: Dict[str, str],
        contrastive_loss: ContrastiveLoss,
        reconstruction_losses: List[ReconstructionLossWithFeatures],
        pitch_prediction_loss: RelativePitchDifferenceLoss,
        generator: Generator,
        discriminator: MultiPeriodDiscriminator,
        generator_optimizer: Optimizer,
        discriminator_optimizer: Optimizer,
        initial_training_step: int,
        max_training_step: int,
    ):
        super().__init__()

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.cqt_hop_length = cqt_hop_length
        self.cqt_bins_per_octave = cqt_bins_per_octave
        self.segment_length_s = segment_length_s
        self.linguistic_loss_start_weight = linguistic_loss_start_weight
        self.linguistic_loss_end_weight = linguistic_loss_end_weight
        self.non_scalar_logging_steps = non_scalar_logging_steps
        self.vc_source_audio_paths = vc_source_audio_paths
        self.vc_target_audio_paths = vc_target_audio_paths
        self.max_training_step = max_training_step
        self.initial_training_step = initial_training_step

        self.content_weight_increment = (
            linguistic_loss_end_weight - linguistic_loss_start_weight
        ) / max_training_step
        self.content_weight = (
            linguistic_loss_start_weight
            + initial_training_step * self.content_weight_increment
        )

        self.contrastive_loss = contrastive_loss
        self.reconstruction_losses = AggregatedReconstructionLossWithFeatures(
            reconstruction_losses
        )
        self.pitch_prediction_loss = pitch_prediction_loss
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.cmap = np.array(plt.get_cmap("viridis").colors)
        self.source_dict = {}
        for src_audio_name, src_audio_path in vc_source_audio_paths.items():
            source_audio, src_sr = torchaudio.load(os.path.join(root, src_audio_path))
            self.source_dict[src_audio_name] = T.Resample(
                src_sr, self.input_sample_rate
            )(source_audio).float()
        self.target_dict = {}
        for tgt_audio_name, tgt_audio_path in vc_target_audio_paths.items():
            if tgt_audio_path is not None:
                target_audio, tgt_sr = torchaudio.load(
                    os.path.join(root, tgt_audio_path)
                )
                self.target_dict[tgt_audio_name] = T.Resample(
                    tgt_sr, self.output_sample_rate
                )(target_audio).float()

    def update_warmup(self) -> None:
        """Update the content loss weights."""
        self.content_weight += self.content_weight_increment

    @cuda_synchronized_timer(DO_PROFILING, prefix="GeneratorLoss")
    def linguistic_loss(
        self,
        audio: torch.Tensor,
        linguistic_features: torch.Tensor,
        perturbed_audio_secondary: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ContrastiveMetrics]:
        # [B, linguistic_hidden_channels, S]
        linguistic_features_alternate_perturbations = self.generator.analyze_linguistic(
            audio,
            enable_information_perturbator=True,
            perturbed_inputs=perturbed_audio_secondary,
        )
        contrastive_loss, positive_loss, negative_loss = self.contrastive_loss(
            linguistic_features, linguistic_features_alternate_perturbations
        )
        return contrastive_loss, {
            "cont_loss": contrastive_loss.detach(),
            "metric_pos": positive_loss.detach(),
            "metric_neg": negative_loss.detach(),
        }

    @cuda_synchronized_timer(DO_PROFILING, prefix="GeneratorLoss")
    def discriminative_loss(
        self, audio: torch.Tensor, synth: torch.Tensor
    ) -> AdversarialDiscriminativeLosses:
        (
            logits_per_period_reconstructed,
            feature_maps_per_period_reconstructed,
        ) = self.discriminator(synth)
        _, feature_maps_per_period_original = self.discriminator(audio)

        discriminator_fake_loss = audio.new_zeros(1)
        features_matching_loss = audio.new_zeros(1)

        # discriminator loss
        for logits_reconstructed in logits_per_period_reconstructed:
            discriminator_fake_loss = (
                discriminator_fake_loss + (1 - logits_reconstructed).square().mean()
            )

        # feature-matching loss
        for fmap_f, fmap_r in zip(
            feature_maps_per_period_reconstructed, feature_maps_per_period_original
        ):
            for ff, fr in zip(fmap_f, fmap_r):
                features_matching_loss = features_matching_loss + (ff - fr).abs().mean()

        return {
            "d_fake": discriminator_fake_loss,
            "features_matching_loss": features_matching_loss,
        }

    @cuda_synchronized_timer(DO_PROFILING, prefix="GeneratorLoss")
    def forward(
        self,
        audio: torch.Tensor,
        synth: torch.Tensor,
        pitch: torch.Tensor,
        cqt: torch.Tensor,
        linguistic_features: torch.Tensor,
        it: int,
        perturbed_audio_secondary: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        GeneratorLossMetrics,
        GeneratorLossImages,
        GeneratorLossAudios,
        GeneratorLossPlots,
    ]:
        device = audio.device
        bsize, _ = audio.shape

        (
            aggregated_reconstruction_loss,
            reconstruction_features,
        ) = self.reconstruction_losses(audio, synth)

        if isinstance(self.pitch_prediction_loss, RelativePitchDifferenceLoss):
            pitch_loss = self.pitch_prediction_loss(
                cqt=cqt,
                pitch=pitch,
                pitch_encoder=self.generator.pitch_encoder,
                bsize=bsize,
            )
        elif isinstance(self.pitch_prediction_loss, PitchSupervisionLoss):
            pitch_loss = self.pitch_prediction_loss(
                audios=audio, cqts=cqt, pred_pitches=pitch
            )

        (
            linguistic_contrastive_loss,
            linguistic_contrastive_metrics,
        ) = self.linguistic_loss(
            audio,
            linguistic_features=linguistic_features,
            perturbed_audio_secondary=perturbed_audio_secondary,
        )

        discriminative_losses = self.discriminative_loss(audio, synth)

        # NOTE: this rough scaling was introduced in the legacy repository (@revsic) and appears
        # to be taken from the paper GANSpeech by Yang et al., published at INTERSPEECH'21
        # Paper available at: https://arxiv.org/abs/2106.15153
        # An open-source implementation of GANSpeech is not available, so this implementation
        # with a .detach() call is just hypothetical.
        weight = (
            aggregated_reconstruction_loss
            / discriminative_losses["features_matching_loss"]
        ).detach()

        self.update_warmup()

        loss = (
            discriminative_losses["d_fake"]
            + weight * discriminative_losses["features_matching_loss"]
            + aggregated_reconstruction_loss
            + self.content_weight * linguistic_contrastive_loss
        )
        if pitch_loss.requires_grad:
            loss += pitch_loss

        metrics: GeneratorLossMetrics = {
            "gen/loss": loss.detach().item(),
            "gen/d-fake": discriminative_losses["d_fake"].detach().item(),
            "gen/fmap": discriminative_losses["features_matching_loss"].detach().item(),
            "gen/rctor": aggregated_reconstruction_loss.detach().item(),
            "gen/pitch": pitch_loss.detach().item(),
            "gen/cont": linguistic_contrastive_loss.detach().item(),
            "metric/cont-pos": linguistic_contrastive_metrics["metric_pos"]
            .detach()
            .item(),
            "metric/cont-neg": linguistic_contrastive_metrics["metric_neg"]
            .detach()
            .item(),
            "common/warmup": self.content_weight,
            "common/weight": weight.item(),
            "common/learning-rate-g": self.generator_optimizer.param_groups[0]["lr"],
            "common/learning-rate-d": self.discriminator_optimizer.param_groups[0][
                "lr"
            ],
        }

        images: GeneratorLossImages = {}
        audios: GeneratorLossAudios = {}
        plots: GeneratorLossPlots = {}
        mode = "train" if self.generator.training else "validation"
        if it % self.non_scalar_logging_steps == 0:
            images = {
                f"mel-gt/{mode}": self.mel_img(
                    reconstruction_features["mel_original"]
                    .detach()
                    .cpu()
                    .numpy()[LOG_IDX]
                ),
                f"mel-synth/{mode}": self.mel_img(
                    reconstruction_features["mel_reconstructed"]
                    .detach()
                    .cpu()
                    .numpy()[LOG_IDX]
                ),
                f"log-cqt/{mode}": self.mel_img(
                    cqt.clamp_min(1e-5).log().cpu().detach().numpy()[LOG_IDX]
                ),
            }
            audios = {
                f"speech/{mode}": audio.detach().cpu().numpy()[LOG_IDX],
                f"synth/{mode}": synth.detach().cpu().numpy()[LOG_IDX],
            }
            if mode == "validation":
                for (src_audio_name, src_audio), (
                    tgt_audio_name,
                    tgt_audio,
                ) in itertools.product(
                    self.source_dict.items(), self.target_dict.items()
                ):
                    audios.update(
                        {
                            f"conversion/vctk-{src_audio_name}-to-{tgt_audio_name}": self.generator.voice_conversion(
                                src_audio, tgt_audio, device
                            )
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        }
                    )
            plots = {
                f"pitch/{mode}_{i}": self.plot_pitch(cqt_item, pitch_item)
                for i, (cqt_item, pitch_item) in enumerate(
                    zip(
                        cqt.cpu().numpy()[:LOG_PLOTS, ...],
                        pitch.cpu().detach().numpy()[:LOG_PLOTS, ...],
                    )
                )
            }
        return loss, metrics, images, audios, plots

    @cuda_synchronized_timer(DO_PROFILING, prefix="GeneratorLoss")
    def mel_img(self, mel: np.ndarray) -> np.ndarray:
        """Generate mel-spectrogram images.
        Args:
            mel: [np.float32; [mel, T]], mel-spectrogram.
        Returns:
            [np.float32; [3, mel, T]], mel-spectrogram in viridis color map.
        """
        # minmax norm in range(0, 1)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-7)
        # in range(0, 255)
        mel = (mel * 255).astype(np.uint8)
        # [mel, T, 3]
        mel = self.cmap[mel]
        # [3, mel, T], make origin lower
        mel = np.flip(mel, axis=0).transpose(2, 0, 1)
        return mel

    @cuda_synchronized_timer(DO_PROFILING, prefix="GeneratorLoss")
    def plot_pitch(self, cqt: np.ndarray, pitch: np.ndarray) -> Figure:
        fig, ax = plt.subplots()
        img = librosa.display.specshow(
            librosa.amplitude_to_db(cqt, ref=np.max),
            sr=self.input_sample_rate,
            x_axis="time",
            y_axis="cqt_hz",
            hop_length=self.cqt_hop_length,
            fmin=32.7,
            bins_per_octave=self.cqt_bins_per_octave,
            ax=ax,
        )
        plt.plot(
            np.arange(0, self.segment_length_s, self.segment_length_s / len(pitch)),
            pitch,
            color="green",
            label="f0",
        )
        ax.legend()
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        return fig

    @copy_docstring_and_signature(forward)
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)
