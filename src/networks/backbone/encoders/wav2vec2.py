from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio
from torch import nn
from transformers import Wav2Vec2Model
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput

from src.dataclasses.backbone import PretrainedWav2Vec2Config


class Wav2Vec2Wrapper(nn.Module):
    """Wrapping huggingface wav2vec2.0."""

    wav2vec2: Wav2Vec2Model
    resample: torchaudio.transforms.Resample

    channels: int
    layer_for_linguistic_features: int
    sample_rate: int

    def __init__(
        self,
        global_sample_rate: int,
        pretrained_model: PretrainedWav2Vec2Config,
        layer_for_linguistic_features: int,
        trim_unused_layers: bool,
    ):
        """Load the wav2vec2.0 pretrained model and instantiate an nn.Resampler for adapting sample-rates.

        Args:
            global_sample_rate:
                The general sample rate of the NANSY++ model.
            pretrained_model_name_and_sample_rate (default uses Facebook's XLSR-53):
                Name and sample-rate of the pretrained model, .
            layer_for_linguistic_features:
                Hidden layer from the wav2vec2 model used for extracting linguistic features.
            trim_unused_layers:
                If True, removes all layers in the Wav2Vec2Encoder after `layer_for_linguistic_features`.
        """
        super().__init__()

        self.name = pretrained_model.name
        self.wav2vec2_sample_rate = pretrained_model.sample_rate
        self.trim_unused_layers = trim_unused_layers

        self.wav2vec2: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(
            self.name, local_files_only=True
        )
        assert isinstance(
            self.wav2vec2, Wav2Vec2Model
        ), "Wav2Vec2Model initialization failed"

        # aliases
        self.channels = self.wav2vec2.config.output_hidden_size

        self.strides = np.prod(self.wav2vec2.config.conv_stride).item()
        self.layer_for_linguistic_features = (
            layer_for_linguistic_features
            or Wav2Vec2Wrapper.LAYER_FOR_LINGUISTIC_FEATURES
        )
        if self.trim_unused_layers:
            self._trim_wav2vec2_encoder_at_layer_(self.layer_for_linguistic_features)

        # resampler
        self.global_sample_rate = global_sample_rate
        self.resample = torchaudio.transforms.Resample(
            self.global_sample_rate, self.wav2vec2_sample_rate
        )
        self.eval()

    @torch.no_grad()
    def forward(
        self, audio: torch.Tensor, audio_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract the features from audio.
        Args:
            audio: [torch.float32; [B, T']], audio, [-1, 1]-ranged.
            audio_lengths: [torch.long; [B]], individual length of the audio samples in the batch,
                masking the inputs if provided.
        Returns:
            linguistic: [torch.float32; [B, S, C]], linguistic encodings,
                where C = `channels`
                      S = T // `conv_stride`, T = ceil(T' / `sr` x `sr_w2v2`)
        """
        # [B, T]
        audio = self.resample(audio)

        # B, T
        batch_size, duration = audio.shape
        if audio_lengths is None:
            audio_lengths = torch.full(
                (batch_size,), duration, dtype=torch.long, device=audio.device
            )
        else:
            # rearange to 16khz audio frames
            audio_lengths = torch.ceil(
                audio_lengths / self.global_sample_rate * self.wav2vec2_sample_rate
            ).to(torch.long)

        # [B, T]
        mask = (
            torch.arange(duration, device=audio_lengths.device)[None]
            < audio_lengths[:, None]
        ).to(torch.float32)

        ## normalize the inputs before feeding them to wav2vec2
        ## reference: Wav2VecFeatureExtractor
        # [B]
        mean = (audio * mask).sum(dim=-1) / audio_lengths.to(torch.float32)
        # [B]
        var = ((audio - mean[:, None]) * mask).square().sum(dim=-1) / audio_lengths.to(
            torch.float32
        )
        # small additive value for numerical stability of square root
        eps = 1e-7
        # [B, T]
        normed = (audio - mean[:, None]) / (var[:, None] + eps).sqrt() * mask

        # run predict
        output: Wav2Vec2BaseModelOutput = self.wav2vec2(
            normed, attention_mask=mask.to(torch.long), output_hidden_states=True
        )
        assert output.hidden_states is not None  # helps the type-checker

        # [B, S, C(=1024)]
        linguistic_features = output.hidden_states[self.layer_for_linguistic_features]
        return linguistic_features

    def _trim_wav2vec2_encoder_at_layer_(self, layer: int) -> None:
        del self.wav2vec2.encoder.layers[layer + 1 :]

    def train(self, mode: bool = True) -> "Wav2Vec2Wrapper":
        """Supports only evaluation."""
        if mode:
            import warnings

            warnings.warn(
                "Wav2Vec2Wrapper does not support training mode, using frozen model."
            )
        return super().train(False)

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Do not return the state dict."""
        return {}

    def _load_from_state_dict(self, *args, **kwargs):
        """Do not load state dict."""
        pass
