from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.grads import grad_norm as lightning_grad_norm
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter

from src.data.datamodule.backbone import AudioData
from src.losses.backbone.discriminator import LeastSquaresDiscriminatorLoss
from src.losses.backbone.generator import (
    GeneratorLoss,
    GeneratorLossAudios,
    GeneratorLossImages,
    GeneratorLossMetrics,
    GeneratorLossPlots,
)
from src.networks.backbone.generator import Generator
from src.networks.misc.mpd import MultiPeriodDiscriminator
from src.utilities.profiling import DO_PROFILING, cuda_synchronized_timer


class Backbone(pl.LightningModule):
    def __init__(
        self,
        input_sample_rate: int,
        output_sample_rate: int,
        generator: Generator,
        discriminator: MultiPeriodDiscriminator,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        generator_loss: GeneratorLoss,
        discriminator_loss: LeastSquaresDiscriminatorLoss,
        grad_norm_logging_interval_batches: int,
        gradient_clip_val: Optional[float],
        gradient_clip_algorithm: Optional[str],
        accumulate_grad_batches: int,
    ):
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.configure_optimizers()
        self.automatic_optimization = False
        self.grad_norm_logging_interval_batches = grad_norm_logging_interval_batches
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.accumulate_grad_batches = accumulate_grad_batches

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        return [self.generator_optimizer, self.discriminator_optimizer], []

    def training_step(
        self,
        batch: AudioData,
        batch_idx: int,
    ) -> None:
        src_audios = batch["src_audios"]
        tgt_audios = batch["tgt_audios"]
        perturbed_audio_1 = batch.get("perturbed_audio_1", None)
        perturbed_audio_2 = batch.get("perturbed_audio_2", None)
        precomputed_cqt = batch.get("cqt", None)
        precomputed_mel = batch.get("mel_spec", None)

        generator_optimizer, discriminator_optimizer = self.optimizers()

        # Synthesize "fake" data
        synth, analysis_features, _ = self.generator(
            src_audios,
            perturbed_inputs=perturbed_audio_1,
            enable_information_perturbator=True,
            precomputed_cqt=precomputed_cqt,
            precomputed_mel=precomputed_mel,
        )

        # Train discriminator
        self.toggle_optimizer(discriminator_optimizer)
        discriminator_optimizer.zero_grad()

        logits_real, _ = self.discriminator(tgt_audios)
        logits_fake, _ = self.discriminator(synth.detach())

        discriminator_loss, discriminator_metrics = self.discriminator_loss(
            logits_fake, logits_real
        )

        self.manual_backward(discriminator_loss)

        # gradient clipping needs to be performed manually in case of manual optimization
        if (
            self.gradient_clip_val is not None
            and self.gradient_clip_algorithm is not None
        ):
            self.clip_gradients(
                discriminator_optimizer,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm=self.gradient_clip_algorithm,
            )

        self._track_grad_norm(
            self.discriminator,
            discriminator_metrics,
            batch_index=batch_idx,
            prefix="grad-norm/disc/",
        )
        self.log_dict(discriminator_metrics, sync_dist=False, rank_zero_only=True)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad()
        self.untoggle_optimizer(discriminator_optimizer)

        # Train generator
        self.toggle_optimizer(generator_optimizer)
        generator_optimizer.zero_grad()

        # We have only updated the discriminator in the previous steps,
        # so the fake data remains up-to-date
        (
            aggregated_generator_loss,
            generator_metrics,
            generator_images,
            generator_audios,
            generator_plots,
        ) = self.generator_loss(
            audio=tgt_audios,
            synth=synth,
            pitch=analysis_features["pitch"],
            cqt=analysis_features["cqt"],
            linguistic_features=analysis_features["linguistic"],
            it=batch_idx,
            perturbed_audio_secondary=perturbed_audio_2,
        )

        self.manual_backward(aggregated_generator_loss)

        # gradient clipping needs to be performed manually in case of manual optimization
        if (
            self.gradient_clip_val is not None
            and self.gradient_clip_algorithm is not None
        ):
            self.clip_gradients(
                generator_optimizer,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm=self.gradient_clip_algorithm,
            )

        self._track_grad_norm(
            self.generator,
            generator_metrics,
            batch_index=batch_idx,
            prefix="grad-norm/gen/",
        )

        self.log_all(
            generator_metrics, generator_images, generator_audios, generator_plots
        )

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            generator_optimizer.step()
            generator_optimizer.zero_grad()
        self.untoggle_optimizer(generator_optimizer)

    def validation_step(
        self,
        batch: AudioData,
        batch_idx: int,
    ) -> None:
        src_audios = batch["src_audios"]
        tgt_audios = batch["tgt_audios"]
        perturbed_audio_2 = batch.get("perturbed_audio_2", None)

        # Evaluate generator
        synth, analysis_features, _ = self.generator(
            src_audios,
            enable_information_perturbator=False,
        )
        _, generator_metrics, images, audios, plots = self.generator_loss(
            audio=tgt_audios,
            synth=synth,
            pitch=analysis_features["pitch"],
            cqt=analysis_features["cqt"],
            linguistic_features=analysis_features["linguistic"],
            it=batch_idx,
            perturbed_audio_secondary=perturbed_audio_2,
        )
        self.log_all(generator_metrics, images, audios, plots)

        # Evaluate discriminator
        logits_real, _ = self.discriminator(tgt_audios)
        logits_fake, _ = self.discriminator(synth)
        _, discriminator_metrics = self.discriminator_loss(logits_fake, logits_real)
        self.log_dict(discriminator_metrics, sync_dist=True)

    @cuda_synchronized_timer(DO_PROFILING, prefix="Backbone")
    def log_all(
        self,
        losses: GeneratorLossMetrics,
        images: GeneratorLossImages,
        audios: GeneratorLossAudios,
        plots: GeneratorLossPlots,
    ) -> None:
        assert self.logger is not None, "you didn't initialize a logger"
        assert isinstance(
            self.logger, TensorBoardLogger
        ), "expects self.logger to be a TensorBoardLogger"

        tensorboard: SummaryWriter = self.logger.experiment
        self.log_dict(losses, sync_dist=False, rank_zero_only=True)
        for name, image in images.items():
            tensorboard.add_image(name, image, int(self.global_step / 2))
        for name, audio in audios.items():
            tensorboard.add_audio(
                name,
                audio,
                int(self.global_step / 2),
                sample_rate=self.output_sample_rate,
            )
        for name, plot in plots.items():
            tensorboard.add_figure(name, plot, int(self.global_step / 2))

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if strict:
            missing_keys = super().load_state_dict(state_dict=state_dict, strict=False)
            not_wav2vec2_keys = [
                key for key in missing_keys[0] if "wav2vec2" not in key
            ]
            if len(not_wav2vec2_keys) == 0:
                raise Exception(f"Missing state dict keys: {not_wav2vec2_keys}")
        else:
            return super().load_state_dict(state_dict=state_dict, strict=strict)

    def _track_grad_norm(
        self,
        model: nn.Module,
        metrics_dict: Dict[str, Union[float, torch.Tensor]],
        batch_index: int,
        prefix: str = "",
    ) -> None:
        """
        Gotchas:
        - Must be careful to only retrieve norm of the gradients AFTER the backward pass!
        - Retrieving this is pretty slow, so do not compute this everytime!
        """
        if batch_index % self.grad_norm_logging_interval_batches != 0:
            return
        (
            grad_norms,
            mean_param_norm,
        ) = Backbone._mean_grad_and_param_norms(model)
        if prefix != "":
            grad_norms = {prefix + key: value for key, value in grad_norms.items()}
        metrics_dict.update(grad_norms)
        metrics_dict[f"{prefix}param-norm"] = mean_param_norm

    @staticmethod
    @cuda_synchronized_timer(DO_PROFILING, prefix="Backbone")
    @torch.no_grad()
    def _mean_grad_and_param_norms(
        model: nn.Module,
    ) -> Tuple[Dict[str, float], float]:
        grad_norms = lightning_grad_norm(model, 2)

        parameters = model.parameters()
        param_norm = np.mean(
            [torch.norm(p).item() for p in parameters if p.dtype == torch.float32]
        ).item()
        return grad_norms, param_norm
