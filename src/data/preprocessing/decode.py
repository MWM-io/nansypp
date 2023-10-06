import glob
import os
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm
from typing_extensions import Annotated

app = typer.Typer()


def decode_audio(audio_path: str, sample_rate: int) -> np.ndarray:
    """
    Decode audio at given sampling rate.

    Args:
        audio_path: path to audio file.
        sample_rate: sampling rate to use for decoding.
    """
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio


def process_subset(
    audio_dict: Dict[str, str], list_idx: List[int], sample_rate: int
) -> None:
    """
    Decode list of audio.
    Args:
        audio_list: list of audio paths.
        list_idx: list of indices to process in loop.
        output_dir: path to output directory.
        sample_rate: sampling rate to use for decoding.
    """
    audio_list = list(audio_dict.keys())
    for idx in tqdm(list_idx):
        output_path = audio_list[idx]
        audio_path = audio_dict[output_path]
        if not os.path.isfile(output_path):
            audio = decode_audio(audio_path, sample_rate)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, audio)


def main(
    input_dir: Annotated[
        str, typer.Option("-i", help="path to input directory containing raw audio")
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "-o", help="path to output directory that will contain decoded audio"
        ),
    ],
    sample_rate: Annotated[
        int,
        typer.Option("-sr", help="sampling rate"),
    ] = 44100,
    n_processes: Annotated[
        int,
        typer.Option("-np", help="number of processes"),
    ] = 6,
):
    audio_path_to_decode_path = (
        lambda i: os.path.join(output_dir, i.replace(input_dir, "")).rsplit(".", 1)[0]
        + ".npy"
    )

    os.makedirs(output_dir, exist_ok=True)
    audio_list = (
        glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
        + glob.glob(os.path.join(input_dir, "**/*.flac"), recursive=True)
        + glob.glob(os.path.join(input_dir, "**/*.mp4"), recursive=True)
        + glob.glob(os.path.join(input_dir, "**/*.mp3"), recursive=True)
        + glob.glob(os.path.join(input_dir, "**/*.sph"), recursive=True)
        + glob.glob(os.path.join(input_dir, "**/*.ogg"), recursive=True)
    )
    audio_dict = {audio_path_to_decode_path(path): path for path in audio_list}
    dataframe = pd.DataFrame.from_dict(
        audio_dict, orient="index", columns=["audio"]
    ).reset_index(names="decoded")
    dataframe.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)
    # Divide dataset equally into sublists for each process
    sublists = np.array_split(np.arange(len(audio_dict)), n_processes)

    procs = []
    # instantiating processes
    for sublist in sublists:
        proc = Process(
            target=process_subset,
            args=(audio_dict, sublist, sample_rate),
        )
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

    print("\nAll process finished")


if __name__ == "__main__":
    typer.run(main)
