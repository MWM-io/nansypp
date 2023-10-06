"""
https://jonathanbgn.com/2021/08/30/audio-augmentation.html
"""

import math
import os
import pathlib
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from src.data.utils import decoded_audio_duration, fast_load


class Augmentation(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.augmentation_number = 0

    @abstractmethod
    def __call__(self, audio_data):
        pass

    @abstractmethod
    def sample(self, size, audio_length):
        pass


class RandomSpeedChange(Augmentation):
    def __init__(self, sample_rate: int):
        self.augmentation_number = 1
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        speed_factor = random.choice([0.9, 1.0, 1.1])
        if speed_factor == 1.0:  # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects
        )
        return transformed_audio

    def sample(self, size, audio_length):
        pass


class RandomBackgroundNoise(Augmentation):
    def __init__(
        self,
        sample_rate: int,
        noise_dir: str,
        min_snr_db: int,  # 0
        max_snr_db: int,  # 15
        noise_scale: float,
        augmentation_number: int,
        length_s: float,
    ):
        """
        Args:
            sample_rate: sample rate.
            noise_dir: directory containing noise audio samples
            min_snr_db: minimum source-to-noise-ratio in decibel used to generate signal scaling audio data
            max_snr_db: maximum source-to-noise-ratio in decibel used to generate signal scaling audio data
            noise_scale: noise signal weight
            augmentation_number: augmentation index used when composing
            length_s: segment length of audio data from dataset in seconds
        """
        self.augmentation_number = augmentation_number
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.noise_scale = noise_scale
        self.length_s = length_s
        self.length_f = int(sample_rate * length_s)

        if not os.path.exists(noise_dir):
            raise IOError(f"Noise directory `{noise_dir}` does not exist")
        # find all NPY files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob("*.npy"))
        if len(self.noise_files_list) == 0:
            raise IOError(
                f"No decoded .npy file found in the noise directory `{noise_dir}`"
            )
        self.noise_files_dict = {
            path: int(decoded_audio_duration(path, sample_rate) * sample_rate)
            for path in tqdm(self.noise_files_list)
        }

    def __call__(self, audio_data, noises=None):
        """Add random noise to the audio_data.
        Args:
            audio_data: [torch.float32; [B, T]], input tensor.
        Returns:
            [torch.float32; [B, T]], generated augmentation.
        """
        shape = audio_data.shape
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape((1, -1))
        N, audio_length = audio_data.shape
        if noises is None:
            noises = self.sample(N, audio_length)
        noises_to_add = noises[:N, :audio_length].to(audio_data.device)
        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2, dim=1)
        noise_power = noises_to_add.norm(p=2, dim=1)
        scale = (snr * noise_power / (audio_power + 1e-6)).reshape(-1, 1)
        result = (scale * audio_data + self.noise_scale * noises_to_add) / 2
        result = result.reshape(shape)
        return result

    def sample(self, size, audio_length):
        file_indices = np.random.choice(len(self.noise_files_list), size, replace=False)
        return torch.vstack(
            [
                fast_load(
                    self.noise_files_list[file_idx],
                    audio_length,
                    np.random.randint(
                        0,
                        self.noise_files_dict[self.noise_files_list[file_idx]]
                        - audio_length,
                    ),
                )
                for file_idx in file_indices
            ]
        )


class ComposeTransform(Augmentation):
    def __init__(self, transforms):
        self.augmentation_number = len(transforms)
        self.transforms = transforms

    def __call__(self, audio_data):
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data

    def sample(self, size, audio_length):
        pass
