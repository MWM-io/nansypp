import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import typer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.utils import decoded_audio_duration, fast_load
from src.inference.backbone import BackboneInferencer

app = typer.Typer()


class ExtractStatsDataset(Dataset):
    def __init__(
        self,
        sample_rate: int,
        file: Optional[str] = None,
        segments: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__()
        if file is not None:
            self.segments = pd.read_csv(file)
        elif segments is not None:
            self.segments = segments.copy()
        else:
            raise AttributeError("Either file or segments should be provided.")
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.segments.iloc[index]
        sample_start = sample.get("gt_start", 0)
        sample_end = sample.get("gt_end", 100)
        if "decoded" in sample:
            num_frames = int(
                min(
                    sample_end - sample_start,
                    decoded_audio_duration(sample.decoded, self.sample_rate),
                )
                * self.sample_rate
            )
            audio_sample = fast_load(
                path=sample.decoded,
                num_frames=num_frames,
                frame_offset=int(sample_start) * self.sample_rate,
            )
        elif "audio" in sample:
            audio_sample, sample_rate = torchaudio.load(sample.audio)
            if sample_rate != self.sample_rate:
                audio_sample = T.Resample(sample_rate, self.sample_rate)(audio_sample)
            audio_sample = audio_sample[0]
        else:
            raise AttributeError
        return audio_sample


def main(
    data_file: Path,
    sample_rate: int,
    path_to_save: Path,
    exp_dir: Path,
    ckpt_name: str,
    batch_size: int = 1,
    device: str = "cuda:0",
):
    """
    Script to compute TTS targets for backbone model.
    Args:
        - data_file: path to csv file describing TTS training segments
        - path_to_save: path to dir to save resulting data description file
            and computed statistics
        - exp_dir: experiment dir of backbone
        - ckpt_name: checkpoint name to use
        - config_name: backbone config name from experiment dir
        - batch_size: for statistics computation
        - device
    Example:
    python -m src.data.preprocessing.precompute_tts_targets \
        path/to/dataset.csv \
        path/to/save/ \
        path/to/backbone/experiment/ \
        <checkpoint_name>.ckpt
    """
    backbone_info = {
        "exp_dir": str(exp_dir),
        "ckpt_name": ckpt_name,
    }

    inferencer = BackboneInferencer(str(exp_dir), ckpt_name, device)
    dataset = ExtractStatsDataset(sample_rate=sample_rate, file=data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    batch_path = path_to_save.joinpath("batches/")
    batch_path.mkdir(parents=True, exist_ok=True)
    batch_to_path = lambda i: batch_path.joinpath(f"{i}.pt")

    with open(path_to_save.joinpath("backbone_info.json"), "w") as backbone_file:
        json.dump(backbone_info, backbone_file)

    dataframe = pd.read_csv(data_file)
    dataframe["duration"] = dataframe["decoded"].apply(
        lambda i: decoded_audio_duration(i, sample_rate)
    )
    dataframe["batch"] = (dataframe.index // batch_size).map(batch_to_path)
    dataframe["batch_i"] = dataframe.index % batch_size
    dataframe.to_csv(path_to_save.joinpath(data_file.name), index=False)

    pitch_stats = {
        "p_amp": {
            "min": 1e8,
            "max": -1e8,
        },
        "ap_amp": {
            "min": 1e8,
            "max": -1e8,
        },
        "pitch": {
            "min": 1e8,
            "max": -1e8,
        },
    }
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        stats = inferencer.generator.analyze(
            batch.to(device), enable_information_perturbator=False
        )
        for key, value in pitch_stats.items():
            pitch_stats[key]["min"] = min(value["min"], stats[key].min().item())
            pitch_stats[key]["max"] = max(value["max"], stats[key].max().item())
        torch.save(stats, batch_to_path(i))
    torch.save(pitch_stats, path_to_save.joinpath("pitch_stats.pt"))


if __name__ == "__main__":
    typer.run(main)
