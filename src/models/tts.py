from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.grads import grad_norm as lightning_grad_norm
from torch import nn

from src.data.datamodule.tts import TextToSpeechData
from src.inference.tts import TextToSpeechInferencer
from src.losses.tts import TextToSpeechLoss
from src.networks.tts.tts import TextToSpeechNetwork
from src.utilities.profiling import DO_PROFILING, cuda_synchronized_timer
from src.utilities.tts_evaluation import TTSEvaluationModule


class TextToSpeechModel(pl.LightningModule):
    """NANSY-TTS model."""

    def __init__(
        self,
        network: TextToSpeechNetwork,
        optimizer: torch.optim.Optimizer,
        loss: TextToSpeechLoss,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        grad_norm_logging_interval_batches: int,
        cer_texts_file: Optional[str],
    ):
        """Initializer."""
        super().__init__()

        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss

        self.grad_norm_logging_interval_batches = grad_norm_logging_interval_batches

        self.inferencer = None
        self.tensorboard = None
        self.cer_texts_file = cer_texts_file
        self.tts_cer_module = None

    def training_step(self, batch: TextToSpeechData, batch_idx: int) -> torch.Tensor:
        (
            phoneme_sequence,
            audio_phoneme_alignment,
            backbone_analysis_features,
            randomly_sliced_audio,
        ) = (
            batch["phoneme_sequence"],
            batch["phoneme_duration_f_sequence"],
            batch["backbone_analysis_features"],
            batch["style_audio"],
        )
        upsample_lens = {
            "cqt_len": (backbone_analysis_features["pitch"] != 0).sum(axis=1),
            "linguistic_len": (
                backbone_analysis_features["linguistic"][:, 0, :] != 0
            ).sum(axis=1),
        }
        encoding_features, decoding_features = self.network(
            phoneme_sequence,
            randomly_sliced_audio,
            audio_phoneme_alignment,
            upsample_lens,
        )
        loss, metrics = self.loss(
            encoding_features["duration"].squeeze(1),
            decoding_features["linguistic"],
            decoding_features["p_amp"].squeeze(1),
            decoding_features["ap_amp"].squeeze(1),
            decoding_features["pitch"].squeeze(1),
            audio_phoneme_alignment,
            backbone_analysis_features["linguistic"],
            backbone_analysis_features["p_amp"],
            backbone_analysis_features["ap_amp"],
            backbone_analysis_features["pitch"],
            self.network.pitch_stats,
            "training",
        )
        self._track_grad_norm(
            self.network,
            metrics,
            batch_index=batch_idx,
            prefix="grad-norm/",
        )
        self.log_dict(metrics, sync_dist=False, rank_zero_only=True)

        return loss

    def validation_step(self, batch: TextToSpeechData, batch_idx: int) -> torch.Tensor:
        (
            phoneme_sequence,
            audio_phoneme_alignment,
            backbone_analysis_features,
            randomly_sliced_audio,
        ) = (
            batch["phoneme_sequence"],
            batch["phoneme_duration_f_sequence"],
            batch["backbone_analysis_features"],
            batch["style_audio"],
        )
        upsample_lens = {
            "cqt_len": (backbone_analysis_features["pitch"] != 0).sum(axis=1),
            "linguistic_len": (
                backbone_analysis_features["linguistic"][:, 0, :] != 0
            ).sum(axis=1),
        }
        encoding_features, decoding_features = self.network(
            phoneme_sequence,
            randomly_sliced_audio,
            audio_phoneme_alignment,
            upsample_lens,
        )
        loss, metrics = self.loss(
            encoding_features["duration"].squeeze(1),
            decoding_features["linguistic"],
            decoding_features["p_amp"].squeeze(1),
            decoding_features["ap_amp"].squeeze(1),
            decoding_features["pitch"].squeeze(1),
            audio_phoneme_alignment,
            backbone_analysis_features["linguistic"],
            backbone_analysis_features["p_amp"],
            backbone_analysis_features["ap_amp"],
            backbone_analysis_features["pitch"],
            self.network.pitch_stats,
            "validation",
        )
        self.log_dict(metrics, sync_dist=False, rank_zero_only=True)
        return loss

    def configure_optimizers(self) -> Any:
        if self.scheduler:
            return [self.optimizer], [
                {"scheduler": self.scheduler, "interval": "epoch"}
            ]
        return self.optimizer

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
        ) = TextToSpeechModel._mean_grad_and_param_norms(model)
        if prefix != "":
            grad_norms = {prefix + key: value for key, value in grad_norms.items()}
        metrics_dict.update(grad_norms)
        metrics_dict[f"{prefix}param-norm"] = mean_param_norm

    @staticmethod
    @cuda_synchronized_timer(DO_PROFILING, prefix="TextToSpeechModel")
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

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if strict:
            missing_keys = super().load_state_dict(state_dict=state_dict, strict=False)
            not_wav2vec2_keys = [key for key in missing_keys[0] if "w2v_3rd" not in key]
            assert (
                len(not_wav2vec2_keys) == 0
            ), f"Following keys are missing in the checkpoint: {not_wav2vec2_keys}"
            return self
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    def on_validation_epoch_end(self):
        if not self.cer_texts_file:
            return

        assert self.logger is not None, "you didn't initialize a logger"
        assert isinstance(
            self.logger, TensorBoardLogger
        ), "expects self.logger to be a TensorBoardLogger"

        if self.tensorboard is None:
            self.tensorboard = self.logger.experiment
        if self.tts_cer_module is None:
            self.tts_cer_module = TTSEvaluationModule(
                evaluation_data=self.cer_texts_file
            )
        if self.inferencer is None:
            self.inferencer = TextToSpeechInferencer(device=self.device)
            self.inferencer.tts_network = self.network
            self.inferencer.instantiate_backbone_from_tts()

        self.network.eval()
        self.inferencer.generator.eval()
        with torch.no_grad():
            generated_audios = self.tts_cer_module.generate_audios(self.inferencer)
        for (_, text), audio in generated_audios.items():
            self.tensorboard.add_audio(
                text,
                audio,
                self.global_step,
                sample_rate=self.network.sample_rate,
            )
        transcriptions = self.tts_cer_module.generate_transcriptions(generated_audios)
        cer_score = self.tts_cer_module.compute_score(transcriptions)
        self.tensorboard.add_scalar("CER/validation", cer_score, self.global_step)
