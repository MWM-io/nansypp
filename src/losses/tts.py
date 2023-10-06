from typing import Dict, Tuple, Union

import torch
from torch import nn

TextToSpeechLossMetrics = Dict[str, Union[float, torch.Tensor]]


class TextToSpeechLoss(nn.Module):
    """Text-to-speech loss module."""

    def __init__(
        self,
        pitch_weight: float,
        linguistic_weight: float,
        duration_weight: float,
        linguistic_multiplicator: float = 100,
    ):
        """Initializer."""
        super().__init__()

        self.pitch_weight = pitch_weight
        self.linguistic_weight = linguistic_weight
        self.duration_weight = duration_weight
        self.linguistic_multiplicator = linguistic_multiplicator
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    @staticmethod
    def min_max_norm(value: torch.Tensor, min_max: Dict) -> torch.Tensor:
        return (value - min_max["min"]) / (min_max["max"] - min_max["min"])

    def pitch_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        stats: Dict,
    ) -> torch.Tensor:
        return self.mae(
            self.min_max_norm(prediction, stats) * mask,
            self.min_max_norm(target, stats) * mask,
        )

    def forward(
        self,
        pred_duration: torch.Tensor,
        decoded_linguistic_features: torch.Tensor,
        decoded_p_amp: torch.Tensor,
        decoded_ap_amp: torch.Tensor,
        decoded_pitch: torch.Tensor,
        ref_duration: torch.Tensor,
        ref_linguistic_features: torch.Tensor,
        ref_p_amp: torch.Tensor,
        ref_ap_amp: torch.Tensor,
        ref_pitch: torch.Tensor,
        pitch_stats: Dict,
        mode: str,
    ) -> Tuple[torch.Tensor, TextToSpeechLossMetrics]:
        linguistic_mask = ref_linguistic_features != 0
        pitch_mask = ref_pitch != 0
        duration_mask = ref_duration != 0

        linguistic_loss = self.mse(
            self.linguistic_multiplicator
            * decoded_linguistic_features
            * linguistic_mask,
            self.linguistic_multiplicator * ref_linguistic_features * linguistic_mask,
        )
        p_amp_loss = self.pitch_loss(
            decoded_p_amp, ref_p_amp, pitch_mask, pitch_stats["p_amp"]
        )
        ap_amp_loss = self.pitch_loss(
            decoded_ap_amp, ref_ap_amp, pitch_mask, pitch_stats["ap_amp"]
        )
        pitch_loss = self.pitch_loss(
            decoded_pitch, ref_pitch, pitch_mask, pitch_stats["pitch"]
        )
        duration_loss = self.mse(
            pred_duration * duration_mask,
            ref_duration.to(torch.float32) * duration_mask,
        )

        loss = (
            self.linguistic_weight * linguistic_loss
            + self.pitch_weight * (p_amp_loss + ap_amp_loss + pitch_loss)
            + self.duration_weight * duration_loss
        )
        metrics: TextToSpeechLossMetrics = {
            f"tts/{mode}/loss": loss.detach().item(),
            f"tts/{mode}/linguistic": linguistic_loss.detach().item(),
            f"tts/{mode}/p_amp": p_amp_loss.detach().item(),
            f"tts/{mode}/ap_amp": ap_amp_loss.detach().item(),
            f"tts/{mode}/pitch": pitch_loss.detach().item(),
            f"tts/{mode}/duration": duration_loss.detach().item(),
        }

        return loss, metrics
