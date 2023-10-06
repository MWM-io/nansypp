import glob
import itertools
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.utils import decoded_audio_duration, fast_load


class BackboneDataset(Dataset[torch.Tensor]):
    """Audio dataset returning randomly drawn chunks of audio from audio files."""

    def __init__(
        self,
        segment_length_s: float,
        input_data_dirs: Optional[List[str]],
        input_file_list: Optional[str],
        input_sample_rate: int,
        output_data_dirs: Optional[List[str]],
        output_file_list: Optional[str],
        output_sample_rate: int,
        verbose: bool = True,
    ):
        """
        Args:
            segment_length_s: duration in seconds of sliced segments returned by the dataset.
            input_data_dirs: directory containing decoded audio that will be used as input
            input_file_list: path to .npy file containing a list of files to be used as input. Alternative to input_data_dir.
            input_sample_rate: sample rate of decoded files used as input (specified through input_data_dir or input_file_list)
            output_data_dirs: directory containing decoded audio that will be used as output. Optional.
            output_file_list: path to .npy file containing a list of files to be used as output. Alternative to output_data_dir. Optional.
            output_sample_rate: sample rate of decoded files used as output (specified through output_data_dir or output_file_list)
        """
        self.segment_length_s = segment_length_s
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = (
            output_sample_rate if output_sample_rate is not None else input_sample_rate
        )

        self.input_num_frames = int(self.input_sample_rate * self.segment_length_s)
        self.output_num_frames = int(self.output_sample_rate * self.segment_length_s)

        assert (
            input_data_dirs is not None or input_file_list is not None
        ), "either input_data_dir or input_file_list must be defined"

        input_segments = self._get_segments(
            input_data_dirs, None, input_sample_rate, verbose=verbose
        )

        # Intersect segments if input and output sample rates are different
        if self.output_sample_rate != self.input_sample_rate:
            assert (
                output_data_dirs is not None or output_file_list is not None
            ), "either output_data_dir or output_file_list must be defined if output_sample_rate is not None"

            output_segments = self._get_segments(
                output_data_dirs,
                output_file_list,
                output_sample_rate,
                verbose=verbose,
            )

            input_segments_dict = {
                (os.path.basename(path), offset): (path, offset)
                for path, offset in input_segments
            }
            output_segments_dict = {
                (os.path.basename(path), offset): (path, offset)
                for path, offset in output_segments
            }
            self.segments_dict = {
                (filename, offset): {
                    "input_path": input_segments_dict[filename, offset][0],
                    "output_path": output_segments_dict[filename, offset][0],
                }
                for (filename, offset) in list(
                    set(input_segments_dict.keys()).intersection(
                        set(output_segments_dict.keys())
                    )
                )
            }
        else:
            self.segments_dict = {
                (os.path.basename(path), offset): {
                    "input_path": path,
                    "output_path": path,
                }
                for path, offset in input_segments
            }
        self.segments_keys = list(self.segments_dict.keys())

    def __len__(self) -> int:
        """Number of audio segments in the dataset."""
        return len(self.segments_dict)

    def _get_segments(
        self,
        data_dirs: Optional[List[str]],
        file_list: Optional[str],
        sample_rate: int,
        verbose: bool,
    ) -> List[Tuple[str, float]]:
        if file_list is not None:
            filepaths = np.load(file_list)
        else:
            filepaths = []
            for data_dir in data_dirs:
                filepaths += glob.glob(f"{data_dir}/**/*.npy", recursive=True)
        return list(
            itertools.chain.from_iterable(
                [
                    [
                        (filepath, num_seconds_offset)
                        for num_seconds_offset in range(
                            int(
                                decoded_audio_duration(filepath, sample_rate)
                                - self.segment_length_s
                            )
                            + 1
                        )
                    ]
                    for filepath in (tqdm(filepaths) if verbose else filepaths)
                    if decoded_audio_duration(filepath, sample_rate)
                    >= self.segment_length_s
                ]
            )
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_filename, time_offset = self.segments_keys[index]
        audio_dict = self.segments_dict[(audio_filename, time_offset)]
        src_audio: torch.Tensor = fast_load(
            path=audio_dict["input_path"],
            num_frames=self.input_num_frames,
            frame_offset=int(time_offset * self.input_sample_rate),
        )
        tgt_audio: torch.Tensor = fast_load(
            path=audio_dict["output_path"],
            num_frames=self.output_num_frames,
            frame_offset=int(time_offset * self.output_sample_rate),
        )
        return {"src_audio": src_audio, "tgt_audio": tgt_audio}
