import json
from typing import Dict, Optional, Tuple, TypedDict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from src.data.datamodule.tts import PitchStats
from src.data.text.text_processor import TextProcessor
from src.networks.tts.decoders.amplitude import AmplitudeDecoder
from src.networks.tts.decoders.duration import DurationPredictor
from src.networks.tts.decoders.f0 import F0Decoder
from src.networks.tts.decoders.linguistic import LinguisticDecoder
from src.networks.tts.encoders.phoneme import PhonemeEncoder
from src.networks.tts.encoders.style import StyleEncoder


class EncodingFeatures(TypedDict):
    style_embedding: torch.Tensor
    duration: torch.Tensor
    upsampled_linguistic_phoneme_features: torch.Tensor
    upsampled_pitch_phoneme_features: torch.Tensor


class DecodingFeatures(TypedDict):
    linguistic: torch.Tensor
    p_amp: torch.Tensor
    ap_amp: torch.Tensor
    pitch: torch.Tensor


class TextToSpeechNetwork(nn.Module):
    """NANSY-TTS network."""

    def __init__(
        self,
        phoneme_encoder: PhonemeEncoder,
        style_encoder: StyleEncoder,
        duration_predictor: DurationPredictor,
        linguistic_decoder: LinguisticDecoder,
        amplitude_decoder: AmplitudeDecoder,
        f0_decoder: F0Decoder,
        pitch_stats: PitchStats,
        text_processor: TextProcessor,
        sample_rate: int,
        backbone_info_file: str,
    ):
        """Initializer."""
        super().__init__()
        self.text_processor = text_processor
        self.phoneme_encoder = phoneme_encoder
        self.style_encoder = style_encoder
        self.duration_predictor = duration_predictor
        self.linguistic_decoder = linguistic_decoder
        self.amplitude_decoder = amplitude_decoder
        self.f0_decoder = f0_decoder
        self.pitch_stats = pitch_stats
        self.sample_rate = sample_rate
        with open(backbone_info_file, "r") as backbone_info:
            self.backbone_info = json.load(backbone_info)

    def encode(
        self,
        phoneme_sequence: torch.Tensor,
        waveform: torch.Tensor,
        alignment: Optional[torch.Tensor],
        upsample_lens: Optional[Dict[str, torch.Tensor]] = None,
    ) -> EncodingFeatures:
        """
        Extract TTS features from input audio and phoneme sequence
        Args:
            phoneme_sequence: [torch.float32; [B, T_text]], encoded text.
            waveform: [torch.float32; [B, T]], speech example audio.
            alignment: Optional[torch.float32; [B, T_text]], duration of each phoneme.
            upsample_lens: Optional[Dict[str, [torch.int64; B]]], lengths of features to generate.
        Returns:
            encoding_features: EncodingFeatures.
        """
        # Encoding
        style_embedding = self.style_encoder(waveform)
        phoneme_features = self.phoneme_encoder(phoneme_sequence, style_embedding)
        # Duration prediction and upsampling
        duration = self.duration_predictor(phoneme_features, style_embedding)
        if alignment is not None and upsample_lens is not None:
            align_to_upsample = alignment.unsqueeze(1)
            cqt_len = upsample_lens["cqt_len"]
            linguistic_len = upsample_lens["linguistic_len"]
        else:
            align_to_upsample = duration
            align_to_upsample[align_to_upsample <= 0] = 1
            source_duration = align_to_upsample.sum(axis=-1).unique()
            linguistic_len = (
                source_duration * self.text_processor.linguistic_features_length_ratio
            ).int()
            cqt_len = (
                source_duration * self.text_processor.pitch_features_length_ratio
            ).int()
        upsampled_linguistic_phoneme_features = (
            self.upsample_padded(  # upsample_features
                phoneme_features,
                align_to_upsample,
                linguistic_len,
            )
        )
        upsampled_pitch_phoneme_features = self.upsample_padded(  # upsample_features
            phoneme_features,
            align_to_upsample,
            cqt_len,
        )
        return {
            "style_embedding": style_embedding,
            "duration": duration,
            "upsampled_linguistic_phoneme_features": upsampled_linguistic_phoneme_features,
            "upsampled_pitch_phoneme_features": upsampled_pitch_phoneme_features,
        }

    def decode(
        self,
        upsampled_linguistic_phoneme_features: torch.Tensor,
        upsampled_pitch_phoneme_features: torch.Tensor,
        style_embedding: torch.Tensor,
    ) -> DecodingFeatures:
        """
        Generate linguistic and pitch featrues for audio generation.
        Args:
            upsampled_linguistic_phoneme_features:
                [torch.float32; [B, tts_linguistic_hidden_channels, T_ling]],
                encoded linguistic features.
            upsampled_pitch_phoneme_features:
                [torch.float32; [B, tts_pitch_hidden_channels, T_pitch]],
                encoded pitch features.
            style_embedding:
                [torch.float32; [B, N_style, 1]],
                embeddings of speech example audio.
        Returns:
            decoding_features: DecodingFeatures.
        """
        linguistic_features = self.linguistic_decoder(
            upsampled_linguistic_phoneme_features, style_embedding
        )
        p_amp_norm, ap_amp_norm, amplitude_hiddens = self.amplitude_decoder(
            upsampled_pitch_phoneme_features, style_embedding
        )
        f0_contour_norm = self.f0_decoder(
            upsampled_pitch_phoneme_features, amplitude_hiddens, style_embedding
        )

        p_amp = self.rescale(p_amp_norm, self.pitch_stats["p_amp"])
        ap_amp = self.rescale(ap_amp_norm, self.pitch_stats["ap_amp"])
        f0_contour = self.rescale(f0_contour_norm, self.pitch_stats["pitch"])

        return {
            "linguistic": linguistic_features,
            "p_amp": p_amp,
            "ap_amp": ap_amp,
            "pitch": f0_contour,
        }

    def forward(
        self,
        phoneme_sequence: torch.Tensor,
        waveform: torch.Tensor,
        alignment: Optional[torch.Tensor] = None,
        upsample_lens: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[EncodingFeatures, DecodingFeatures]:
        """
        Extract TTS features from input audio and phoneme sequence
        and generate linguistic and pitch featrues for audio generation.
        Args:
            phoneme_sequence:
                [torch.float32; [B, T_text]], encoded text.
            waveform:
                [torch.float32; [B, T]], speech example audio.
            alignment:
                Optional[torch.float32; [B, T_text]], duration of each phoneme.
            upsample_lens:
                Optional[Dict[str, [torch.int64; B]]], lengths of features to generate.
        Returns:
            encoding_features: EncodingFeatures.
            decoding_features: DecodingFeatures.
        """
        encoding_features = self.encode(
            phoneme_sequence, waveform, alignment, upsample_lens
        )
        # Decoding
        decoding_features = self.decode(
            encoding_features["upsampled_linguistic_phoneme_features"],
            encoding_features["upsampled_pitch_phoneme_features"],
            encoding_features["style_embedding"],
        )
        return encoding_features, decoding_features

    @staticmethod
    def rescale(value: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Min-max normalization for A_p, A_ap and F0.
        """
        return value * (stats["max"] - stats["min"]) + stats["min"]

    @staticmethod
    def resample_duration(duration: torch.Tensor, target_duration: int):
        """
        Upsampling predicted phoneme duration to lengths of linguistic or pitch features.
        """
        source_duration = duration.sum(axis=-1).unique().item()
        length = duration.shape[-1]
        integer_division = (duration / source_duration * target_duration).type(
            torch.int
        )
        modulo = (duration / source_duration * target_duration) % 1.0
        to_add = target_duration - integer_division.sum(axis=1)
        add_from = length - to_add
        _, order = torch.sort(modulo, dim=1)
        for i, value in enumerate(add_from):
            integer_division[i][order[i, value:]] += 1
        assert (len(integer_division.sum(axis=1).unique()) == 1) and (
            integer_division.sum(axis=1).unique().item() == target_duration
        )
        return integer_division

    @staticmethod
    def upsample_features(
        features: torch.Tensor,
        duration: torch.Tensor,
        upsample_len: Optional[int],
    ) -> torch.Tensor:
        """Upsample features given predicted duration.
        Args:
            features: [B, 128, Ntext]
            duration: [B, 1, Ntext]
        Returns:
            upsampled_features: [B, 128, N]
        """
        if len(duration.shape) > 2:
            duration = duration.squeeze(1)
        assert len(duration.sum(axis=-1).unique()) == 1

        if upsample_len is not None:
            duration = TextToSpeechNetwork.resample_duration(duration, upsample_len)

        upsampled_features = torch.stack(
            [
                torch.repeat_interleave(curr_feature, curr_duration, dim=-1)
                for curr_feature, curr_duration in zip(features, duration)
            ]
        )
        return upsampled_features

    @staticmethod
    def upsample_padded(features, duration, upsample_len):
        """
        Upsampling padded features.
        """
        upsampled_features = []
        for i in range(features.shape[0]):
            features_i = features[i, :, duration[i, 0] != 0]
            duration_i = duration[i, :, duration[i, 0] != 0]
            upsample_len_i = upsample_len[i]

            upsampled_features_i = TextToSpeechNetwork.upsample_features(
                features_i.unsqueeze(0), duration_i.unsqueeze(0), upsample_len_i
            )
            upsampled_features.append(upsampled_features_i)

        padded_upsampled_features = pad_sequence(
            [i[0].T for i in upsampled_features],
            batch_first=True,
            padding_value=0,
        ).transpose(1, 2)
        return padded_upsampled_features
