import os
from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from typing_extensions import TypedDict

from src.networks.backbone.generator import AnalysisFeatures

FIELDS_TO_PAD = ["phoneme_sequence", "phoneme_duration_f_sequence"]
FIELD_WITH_DICT = "backbone_analysis_features"

FEATURES_LENGTH_PARENT = {
    "cqt": "cqt",
    "pitch": "cqt",
    "p_amp": "cqt",
    "ap_amp": "cqt",
    "linguistic": "linguistic",
}
FIX_LEN_FEATURES = ["timbre_global", "timbre_bank"]


class TextToSpeechData(TypedDict):
    phoneme_sequence: torch.Tensor
    phoneme_duration_f_sequence: torch.Tensor
    style_audio: torch.Tensor
    backbone_analysis_features: AnalysisFeatures


class PitchStats:
    def __init__(self, stats_dir: str, file: str) -> None:
        self.stats = torch.load(os.path.join(stats_dir, file))

    def __getitem__(self, attr):
        return self.stats[attr]


class TextToSpeechDataModule(pl.LightningDataModule):
    train_dataset: Dataset[TextToSpeechData]
    val_dataset: Dataset[TextToSpeechData]

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        num_val_workers: int,
        shuffle: bool,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.num_val_workers = num_val_workers
        self.shuffle = shuffle

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    @staticmethod
    def collate_tts(items: List) -> TextToSpeechData:
        if isinstance(items, list) and isinstance(items[0], list):
            items = [i for item in items for i in item]
        other_fields = set(items[0].keys()) - set(FIELDS_TO_PAD + [FIELD_WITH_DICT])
        batch = {
            field: torch.vstack([i[field] for i in items]) for field in other_fields
        }
        for field_to_pad in FIELDS_TO_PAD:
            batch[field_to_pad] = (
                pad_sequence(
                    [i[field_to_pad] for i in items],
                    batch_first=True,
                    padding_value=0,
                )
                .clone()
                .detach()
            )
        batch[FIELD_WITH_DICT] = {
            key: torch.vstack([i[FIELD_WITH_DICT][key] for i in items]).clone().detach()
            for key in items[0][FIELD_WITH_DICT].keys()
        }
        return batch

    def train_dataloader(self) -> DataLoader[TextToSpeechData]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_tts,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> DataLoader[TextToSpeechData]:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.collate_tts,
            num_workers=self.num_val_workers,
            pin_memory=True,
            shuffle=False,
        )
