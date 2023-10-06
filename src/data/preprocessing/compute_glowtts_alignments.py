# https://drive.google.com/file/d/1JiCMBVTG4BMREK8cT3MYck1MgYvwASL0/view
GLOW_TTS_CKPT_PATH = "path/to/glow-tts/pretrained.pth"

import os
import sys

import autoroot

GLOW_TTS_REPO_PATH = os.path.join(autoroot.root, "glow-tts/")

sys.path.append(GLOW_TTS_REPO_PATH)


import warnings

import pandas as pd
import torch
import typer
from data_utils import TextMelCollate, TextMelLoader
from text import _id_to_symbol, symbols
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Annotated

import models
import utils

warnings.simplefilter(action="ignore", category=FutureWarning)

app = typer.Typer()


def repr_seg(label, start, end, score=1.0):
    return f"{label}\t({score:4.2f}): [{start:5d}, {end:5d})"


def save_from_attn(attn, x, x_lengths, txt_path):
    index_phonemes = [_id_to_symbol[i] for i in x[:x_lengths].tolist()]
    limits = [0] + attn.sum(axis=1)[:x_lengths].cumsum(dim=0).int().tolist()

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w") as file:
        for i, start, end in zip(index_phonemes, limits[:-1], limits[1:]):
            print(repr_seg("{" + i + "}", start, end), file=file)


def main(
    config: Annotated[str, typer.Argument(help="GlowTTS config file path")],
    input_dir: Annotated[
        str, typer.Argument(help="path to input directory containing raw audio")
    ],
    output_dir: Annotated[
        str,
        typer.Argument(help="path to output directory that will contain alignments"),
    ],
    device: Annotated[str, typer.Argument()] = "cuda:0",
):
    audio_path_to_txt_path = (
        lambda i: os.path.join(output_dir, i.replace(input_dir, "")).rsplit(".", 1)[0]
        + ".txt"
    )

    hps = utils.get_hparams_from_file(config)
    torch.manual_seed(hps.train.seed)

    model = models.FlowGenerator(
        len(symbols) + getattr(hps.data, "add_blank", False),
        out_channels=hps.data.n_mel_channels,
        **hps.model,
    ).to(device)
    utils.load_checkpoint(GLOW_TTS_CKPT_PATH, model)
    model.decoder.store_inverse()  # do not calcuate jacobians for fast decoding
    _ = model.to(device)
    _ = model.eval()

    train_dataset = TextMelLoader(hps.data.training_files, hps.data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=1, rank=0, shuffle=False
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=16,
        shuffle=False,
        batch_size=hps.train.batch_size,
        pin_memory=True,
        drop_last=False,
        collate_fn=TextMelCollate(1),
        sampler=train_sampler,
    )

    paths_to_align_paths = {}
    for batch in tqdm(train_loader, dynamic_ncols=True):
        x, x_lengths, y, y_lengths, paths = batch
        (
            (_, _, _, _, _),
            (_, _, _),
            (attn, _, _),
        ) = model(
            x.to(device),
            x_lengths.to(device),
            y.to(device),
            y_lengths.to(device),
            gen=False,
        )
        attn = attn.squeeze(1)
        size = len(paths)
        for index in range(size):
            txt_path = audio_path_to_txt_path(paths[index][0])
            save_from_attn(
                attn[index],
                x[index],
                x_lengths[index],
                txt_path,
            )
            paths_to_align_paths[paths[index][0]] = txt_path
        del x, x_lengths, y, y_lengths
        torch.cuda.empty_cache()
    dataframe = pd.DataFrame.from_dict(
        paths_to_align_paths, orient="index", columns=["align"]
    ).reset_index(names="audio")
    dataframe.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)


if __name__ == "__main__":
    typer.run(main)
