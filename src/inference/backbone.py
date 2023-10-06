import os
import warnings
from typing import Tuple

import hydra
import hydra.utils as hu
import numpy as np
import pyrootutils
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
import typer
from omegaconf import DictConfig
from typing_extensions import Annotated

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import src.dataclasses.backbone
from src.networks.backbone.generator import Generator

app = typer.Typer()


warnings.filterwarnings("ignore")


class BackboneInferencer:
    generator: Generator
    config: DictConfig
    input_sr: int
    output_sr: int

    def __init__(
        self,
        exp_dir: str,
        checkpoint_name: str,
        device: str,
    ):
        """
        Args:
            exp_dir: experiment directory. Should be an absolute path (Hydra constraint).
            checkpoint_name: checkpoint file name.
            device: device on which inferencer should run. Example: "cpu", "cuda", "cuda:0".
        """
        self.device = device
        self.instantiate_generator(exp_dir)

        checkpoint_path = os.path.join(exp_dir, "checkpoints", checkpoint_name)
        self.load_generator_weights(checkpoint_path)

    def instantiate_generator(self, exp_dir: str) -> None:
        config_dir = os.path.join(exp_dir, ".hydra")
        hydra.core.global_hydra.GlobalHydra.get_state().clear()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize_config_dir(
            config_dir=config_dir
            if os.path.isabs(config_dir)
            else os.path.join(root, config_dir)
        ):
            config = hydra.compose(config_name="config")
        self.input_sr = config.train_dataset.input_sample_rate
        self.output_sr = config.train_dataset.output_sample_rate
        if "additive_noise" in config.generator.information_perturbator:
            config.generator.information_perturbator.additive_noise = None
        self.generator: Generator = hu.instantiate(config.generator)
        self.config = config

    def load_generator_weights(self, checkpoint_path: str) -> None:
        # load weights
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        generator_state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k and "generator_loss" not in k
        }
        inconsistent_keys = self.generator.load_state_dict(
            generator_state_dict, strict=False
        )
        assert (
            len(
                [
                    key
                    for key in inconsistent_keys.missing_keys
                    if ("wav2vec2" not in key) and ("cqt" not in key)
                ]
            )
            == 0
        )
        _ = self.generator.to(self.device)
        _ = self.generator.eval()

    def voice_conversion(
        self, source_audio: torch.Tensor, target_audio: torch.Tensor
    ) -> Tuple[np.ndarray, int]:
        """Apply target audio voice timbre to source audio content.
        Args:
            source_audio: [torch.Tensor; [1, Ns]] tensor with mono audio content at sampling rate defined by model config.
            target_audio: [torch.Tensor; [1, Nt]] tensor with mono audio content at sampling rate defined by model config.
        """
        return (
            self.generator.voice_conversion(
                source_audio=source_audio,
                target_audio=target_audio,
                device=torch.device(self.device),
                f0_statistics=True,
                enable_information_perturbator=False,
            )
            .cpu()
            .squeeze(0)
            .numpy(),
            self.output_sr,
        )

    def reconstruct(self, audio: torch.Tensor) -> Tuple[np.ndarray, int]:
        """Reconstruct audio signal through features extraction followed by audio synthesis.
        Args:
            audio: [torch.Tensor; [1, Ns]] tensor with mono audio content at sampling rate defined by model config.
        """
        return self.voice_conversion(audio, audio)


def main(
    exp_dir: Annotated[str, typer.Argument(help="path to experiment directory")],
    checkpoint_name: Annotated[str, typer.Argument(help="checkpoint name")],
    device: Annotated[
        str, typer.Argument(help="device name: cpu, cuda:0, cuda:1, etc.")
    ] = "cuda",
    source_audio_path: Annotated[
        str, typer.Argument(help="path to source raw audio")
    ] = "static/samples/vctk/p225_001.wav",
    target_audio_path: Annotated[
        str, typer.Argument(help="path to target raw audio")
    ] = "static/samples/vctk/p226_002.wav",
    output_dir: Annotated[
        str, typer.Argument(help="path to directory where to save generated audio")
    ] = "static/tmp",
):
    os.makedirs(output_dir, exist_ok=True)
    inferencer = BackboneInferencer(
        exp_dir if os.path.isabs(exp_dir) else os.path.join(root, exp_dir),
        checkpoint_name,
        device,
    )
    source_audio, source_sr = torchaudio.load(source_audio_path)
    source_audio = T.Resample(source_sr, inferencer.input_sr)(source_audio)
    target_audio, target_sr = torchaudio.load(target_audio_path)
    target_audio = T.Resample(target_sr, inferencer.output_sr)(target_audio)
    converted, sr = inferencer.voice_conversion(source_audio, target_audio)
    output_path = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(source_audio_path))[0]}-to-{os.path.splitext(os.path.basename(target_audio_path))[0]}.wav",
    )
    print(f"Synthesized audio saved at: {output_path}")
    sf.write(output_path, converted, sr)


if __name__ == "__main__":
    typer.run(main)
