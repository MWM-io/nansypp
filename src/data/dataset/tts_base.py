import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.datamodule.tts import TextToSpeechData
from src.data.text.text_processor import TextProcessor
from src.data.utils import fast_load
from src.networks.backbone.generator import AnalysisFeatures


class TTSBaseDataset(Dataset[TextToSpeechData]):
    """
    TextToSpeechDataset.
    Base dataset for TTS. As TTS expects ground truth segment + style segment,
    the simpliest dataset expects a csv file
    where each line precies a pair of ground truth and style segments.
    """

    def __init__(
        self,
        dataset_dir: str,
        descriptor_file: str,
        alignment_file: str,
        sample_rate: int,
        text_processor: TextProcessor,
    ):
        self.descriptor_file = os.path.join(dataset_dir, descriptor_file)
        self.segments = pd.read_csv(self.descriptor_file)
        alignments = pd.read_csv(alignment_file)
        self.segments = self.segments.merge(alignments, on="audio")

        self.sample_rate = sample_rate
        self.text_processor = text_processor

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> TextToSpeechData:
        sample = self.segments.iloc[index]
        duration = sample["duration"]
        style_start = sample.get("style_start", 0)
        style_end = sample.get("style_end", duration)

        style_segment: torch.Tensor = fast_load(
            path=sample["decoded"],
            num_frames=int((style_end - style_start) * self.sample_rate),
            frame_offset=int(style_start * self.sample_rate),
        )
        alignment = self.text_processor.load_alignment(sample["align"])
        (
            phoneme_sequence,
            phoneme_duration_f_sequence,
        ) = self.text_processor.get_alignment_segment(alignment, 0, duration)
        batch = torch.load(sample["batch"], map_location="cpu")
        items = AnalysisFeatures(
            {key: value[sample["batch_i"]].unsqueeze(0) for key, value in batch.items()}
        )
        return {
            "phoneme_sequence": phoneme_sequence,
            "phoneme_duration_f_sequence": phoneme_duration_f_sequence,
            "style_audio": style_segment,
            "backbone_analysis_features": items,
        }
