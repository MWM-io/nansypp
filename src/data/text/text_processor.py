import os
from typing import List, Tuple

import autoroot
import torch

from src.data.text import _symbol_to_id, cmudict, text_to_sequence
from src.data.text.symbols import _pad, symbols

LINGUISTIC_FRAMES_PER_S = 50
PITCH_FRAMES_PER_S = {
    16000: 63,
    44100: 173,
}


class TextProcessor:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.labels = symbols
        self.alignment_whitelist = "1234567890. "
        self.dictionary = {c: i for i, c in enumerate(self.labels)}
        self.align_frame_to_wave_frame = 86
        self.linguistic_features_length_ratio = (
            LINGUISTIC_FRAMES_PER_S / self.align_frame_to_wave_frame
        )
        self.pitch_features_length_ratio = (
            PITCH_FRAMES_PER_S[self.sample_rate] / self.align_frame_to_wave_frame
        )
        self.cmudict = cmudict.CMUDict(
            os.path.join(autoroot.root, "static/tts/cmu_dictionary")
        )
        self.text_cleaners = ["english_cleaners"]
        self.pad_frame = _pad

    def text_to_encoded_phoneme_sequence(self, text: str) -> torch.Tensor:
        return torch.IntTensor(
            text_to_sequence(text.strip(), self.text_cleaners, self.cmudict)
        )

    def encode_phoneme_sequence(self, sequence: List[str]) -> torch.Tensor:
        return torch.tensor([_symbol_to_id[char] for char in sequence])

    def load_alignment(
        self,
        alignment_path: str,
    ) -> List[List]:
        with open(alignment_path, "r") as file:
            lines = file.readlines()
        alignment = []
        for line in lines:
            phoneme = line.split("\t")[0]
            if "{" in phoneme and "}" in phoneme:
                phoneme = phoneme.replace("{", "").replace("}", "")
            start, stop = list(
                map(
                    int,
                    "".join(
                        filter(self.alignment_whitelist.__contains__, line)
                    ).split()[1:],
                )
            )
            alignment.append([phoneme, start, stop])
        return alignment

    def get_alignment_segment(
        self,
        alignment: List[List],
        start_s: float,
        end_s: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alignment_frame_start = int(start_s * self.align_frame_to_wave_frame)
        alignment_frame_end = int(end_s * self.align_frame_to_wave_frame)
        alignment_segment = list(
            filter(
                lambda i: i[1] < alignment_frame_end and alignment_frame_start < i[2],
                alignment,
            )
        )
        alignment_segment[0][1] = max(alignment_frame_start, alignment_segment[0][1])
        alignment_segment[-1][2] = min(alignment_frame_end, alignment_segment[-1][2])

        start_segment, end_segment = [], []
        if alignment_segment[0][1] > alignment_frame_start:
            start_segment.append(
                [self.pad_frame, alignment_frame_start, alignment_segment[0][1]]
            )
        if alignment_segment[-1][2] < alignment_frame_end:
            end_segment.append(
                [self.pad_frame, alignment_segment[-1][2], alignment_frame_end]
            )
        alignment_segment = start_segment + alignment_segment + end_segment

        phoneme_sequence = self.encode_phoneme_sequence(
            [char for (char, _, _) in alignment_segment]
        )
        phoneme_duration_f_sequence = torch.tensor(
            [int((stop - start)) for (_, start, stop) in alignment_segment]
        )
        return phoneme_sequence, phoneme_duration_f_sequence
