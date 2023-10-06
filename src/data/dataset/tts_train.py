import json
import os
from typing import Dict, List

import autoroot
import numpy as np
import torch

from src.data.datamodule.tts import (
    FEATURES_LENGTH_PARENT,
    FIX_LEN_FEATURES,
    TextToSpeechData,
)
from src.data.dataset.tts_base import TTSBaseDataset
from src.data.utils import fast_load
from src.networks.backbone.generator import AnalysisFeatures


class TTSTrainDataset(TTSBaseDataset):
    """
    Text to Speech dataset that returns a list of items of same length for each index.
    """

    def __init__(
        self,
        max_seconds_per_batch: float,
        features_lengths_file: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not os.path.isabs(features_lengths_file):
            features_lengths_file = os.path.join(autoroot.root, features_lengths_file)
        with open(features_lengths_file, "r") as file:
            self.features_lengths = json.load(file)[str(self.sample_rate)]

        self.max_seconds_per_batch = max_seconds_per_batch
        min_duration = self.segments["duration"].min() * 10 // 10
        max_diration = self.segments["duration"].max() * 10 // 10 + 1
        durations = np.arange(min_duration, max_diration, 0.05)
        self.segments_per_duration = {}
        self.num_samples_per_duration = {}
        for segment_min_duration, segment_max_duration in zip(
            durations[:-1], durations[1:]
        ):
            segment_min_duration = round(segment_min_duration, 3)
            curr_segments = self.segments[
                (segment_min_duration < self.segments["duration"])
                & (self.segments["duration"] < segment_max_duration)
            ].reset_index(drop=True)
            if len(curr_segments) > 0:
                self.segments_per_duration[segment_min_duration] = curr_segments
                self.num_samples_per_duration[segment_min_duration] = len(curr_segments)
        self.durations = np.array(list(self.segments_per_duration.keys()))
        self.batches_per_duration = {
            key: int(np.ceil(key * value / self.max_seconds_per_batch))
            for key, value in self.num_samples_per_duration.items()
        }
        self.samples_per_batch_per_duration = {
            key: int(value / self.batches_per_duration[key])
            for key, value in self.num_samples_per_duration.items()
        }
        self.length = sum(self.batches_per_duration.values())
        self.order = []
        for key, value in self.batches_per_duration.items():
            for batch_i in range(value):
                self.order.append((key, batch_i))
        self.order = np.array(self.order)
        self.precompute_batch_indeces()

    def precompute_batch_indeces(self):
        self.batches_indices = {}
        for duration in self.durations:
            positions = np.repeat(
                np.arange(self.batches_per_duration[duration]),
                self.samples_per_batch_per_duration[duration],
            )[: self.num_samples_per_duration[duration]]
            np.random.shuffle(positions)
            self.batches_indices[duration] = positions
        np.random.shuffle(self.order)

    def __len__(self) -> int:
        return self.length

    def get_batch(self, order_i):
        duration, batch_i = order_i
        return self.segments_per_duration[duration].iloc[
            np.where(self.batches_indices[duration] == batch_i)[0]
        ]

    def _get_features_segment(
        self, complete_features: AnalysisFeatures, start: float, end: float
    ) -> AnalysisFeatures:
        items = {feature: complete_features[feature] for feature in FIX_LEN_FEATURES}
        for feature, feature_len in FEATURES_LENGTH_PARENT.items():
            start_f = self.features_lengths[str(float(start))][feature_len]
            delta_f = self.features_lengths[str(float(end - start))][feature_len]
            items[feature] = (
                complete_features[feature][
                    ...,
                    start_f : start_f + delta_f,
                ]
                .clone()
                .detach()
                .to("cpu")
            )
        return AnalysisFeatures(items)

    def get_train_pair(self, sample: Dict, duration: float) -> TextToSpeechData:
        style_segment: torch.Tensor = fast_load(
            path=sample["decoded"],
            num_frames=int(duration * self.sample_rate),
            frame_offset=0,
        )
        alignment = self.text_processor.load_alignment(sample["align"])
        (
            phoneme_sequence,
            phoneme_duration_f_sequence,
        ) = self.text_processor.get_alignment_segment(
            alignment,
            0,
            duration,
        )
        gt_audio_features = torch.load(sample["batch"], map_location="cpu")
        items = self._get_features_segment(
            gt_audio_features,
            0,
            duration,
        )
        return {
            "phoneme_sequence": phoneme_sequence,
            "phoneme_duration_f_sequence": phoneme_duration_f_sequence,
            "style_audio": style_segment,
            "backbone_analysis_features": items,
        }

    def __getitem__(self, index: int) -> List[TextToSpeechData]:
        batch_data = self.get_batch(self.order[index])
        duration = self.order[index][0]
        if index == self.length - 1:
            self.precompute_batch_indeces()
        return [
            self.get_train_pair(sample.to_dict(), duration)
            for _, sample in batch_data.iterrows()
        ]
