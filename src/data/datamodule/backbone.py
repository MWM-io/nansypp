from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from typing_extensions import (  # must import TypedDict from typing_extensions for use with NotRequired
    NotRequired,
    TypedDict,
)

from src.data.dataset.backbone import BackboneDataset
from src.networks.misc.perturbator import InformationPerturbator
from src.networks.misc.transform import ConstantQTransform, MelSpectrogram


class AudioData(TypedDict):
    src_audios: torch.Tensor
    tgt_audios: torch.Tensor
    src_audios: torch.Tensor
    tgt_audios: torch.Tensor
    perturbed_audio_1: NotRequired[torch.Tensor]
    perturbed_audio_2: NotRequired[torch.Tensor]
    cqt: NotRequired[torch.Tensor]
    mel_spec: NotRequired[torch.Tensor]


class BackboneDataModule(pl.LightningDataModule):
    train_dataset: Dataset[AudioData]
    val_dataset: Dataset[AudioData]

    def __init__(
        self,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        train_dataset: BackboneDataset,
        val_dataset: Optional[BackboneDataset],
        val_split: Optional[float],
        val_seed: Optional[int],
        information_perturbator: Optional[InformationPerturbator],
        cqt: Optional[ConstantQTransform],
        mel_spec: Optional[MelSpectrogram],
    ) -> None:
        """
        Args:
            batch_size: batch size.
            pin_memory: pin memory.
            num_workers: number of workers.
            train_dataset: training dataset.
            val_dataset: validation dataset. Optional.
            val_split: fraction of training dataset to be used as validation dataset if validation not specified. Optional.
            val_seed: seed to be used when sampling fraction in case validation not specified. Optional.
            information_perturbator: information perturbator.
            cqt: constant Q-transform computer.
            mel_spec: log-mel scale spectrogram computer.
        """
        super().__init__()
        assert (
            val_dataset is not None or val_split is not None
        ), "either val_dataset or val_split should be specified."

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.segment_length_s = train_dataset.segment_length_s
        self.val_split = val_split
        self.val_seed = val_seed

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.information_perturbator = (
            information_perturbator.cpu()
            if information_perturbator is not None
            else None
        )
        self.cqt = cqt
        self.mel_spec = mel_spec

    def collate_fn(self, items):
        # FIXME: should provide information_perturbator=None and cqt=None to validation
        src_audios = torch.vstack([item["src_audio"] for item in items])
        tgt_audios = torch.vstack([item["tgt_audio"] for item in items])
        batch: AudioData = AudioData(src_audios=src_audios, tgt_audios=tgt_audios)
        if self.information_perturbator is not None:
            batch["perturbed_audio_1"] = self.information_perturbator(src_audios)
            batch["perturbed_audio_2"] = self.information_perturbator(src_audios)
        if self.cqt is not None:
            batch["cqt"] = self.cqt(src_audios)
        if self.mel_spec is not None:
            batch["mel_spec"] = self.mel_spec(src_audios)
        return batch

    def setup(self, stage: Optional[str] = None) -> None:
        if self.val_dataset is None:
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset,
                lengths=[1 - self.val_split, self.val_split],
                generator=(
                    torch.Generator().manual_seed(self.val_seed)
                    if self.val_seed is not None
                    else None
                ),
            )

        print(
            "Training dataset length:",
            round(len(self.train_dataset) * self.segment_length_s / 3600, 1),
            f"hours of audio.",
        )
        print(
            "Evaluation dataset length:",
            round(len(self.val_dataset) * self.segment_length_s / 3600, 1),
            "hours of audio.",
        )

    def train_dataloader(self) -> DataLoader[AudioData]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[AudioData]:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
