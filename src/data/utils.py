import os

import numpy as np
import torch


def fast_load(
    path: str, num_frames: int, frame_offset: int, dtype: str = "float32"
) -> torch.Tensor:
    """
    Fast loading of decoded audio .npy file.
    """
    if dtype == "float32":
        with open(path, "rb") as file:
            file.seek(128 + 4 * frame_offset)
            content = file.read(4 * num_frames)
            array = np.frombuffer(content, dtype="float32")
    elif dtype == "float64":
        with open(path, "rb") as file:
            file.seek(128 + 8 * frame_offset)
            content = file.read(8 * num_frames)
            array = np.float32(np.frombuffer(content, dtype="float64"))
    else:
        raise ValueError(f"Unexpected dtype {dtype}")

    return torch.from_numpy(array.copy())


def decoded_audio_duration(
    path: str, sample_rate: int, dtype: str = "float32"
) -> float:
    """
    Assess decoded audio .npy file duration from file size.
    """
    if dtype == "float32":
        return (os.path.getsize(path) - 128) / (4 * sample_rate)
    if dtype == "float64":
        return (os.path.getsize(path) - 128) / (8 * sample_rate)
    raise ValueError(f"Unexpected dtype {dtype}")
