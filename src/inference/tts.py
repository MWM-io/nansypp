import os
import warnings

import hydra
import hydra.utils as hu
import pyrootutils
import torch
import torchaudio
import torchaudio.transforms as T
import typer
from typing_extensions import Annotated

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from src.inference.backbone import BackboneInferencer
from src.networks.tts.tts import TextToSpeechNetwork

app = typer.Typer()

warnings.filterwarnings("ignore")


class TextToSpeechInferencer:
    def __init__(
        self,
        device="cuda:0",
    ):
        super().__init__()
        self.device = device
        self.backbone_exp_dir = None
        self.backbone_checkpoint_name = None
        self.tts_exp_dir = None
        self.tts_checkpoint_path = None
        self.generator = None
        self.tts_network = None

    def instantiate_backbone(self, exp_dir: str, checkpoint_name: str) -> None:
        if (exp_dir, checkpoint_name) != (
            self.backbone_exp_dir,
            self.backbone_checkpoint_name,
        ):
            self.backbone_exp_dir = exp_dir
            self.backbone_checkpoint_name = checkpoint_name
            self.generator = BackboneInferencer(
                exp_dir,
                checkpoint_name,
                self.device,
            ).generator

    def instantiate_backbone_from_tts(self) -> None:
        self.instantiate_backbone(
            self.tts_network.backbone_info["exp_dir"],
            self.tts_network.backbone_info["ckpt_name"],
        )

    def instantiate_tts(self, exp_dir: str) -> None:
        if not os.path.isabs(exp_dir):
            exp_dir = os.path.join(root, exp_dir)
        if exp_dir == self.tts_exp_dir:
            return
        self.tts_exp_dir = exp_dir
        hydra.core.global_hydra.GlobalHydra.get_state().clear()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize_config_dir(config_dir=os.path.join(exp_dir, ".hydra")):
            config = hydra.compose(config_name="config")
        self.tts_network: TextToSpeechNetwork = hu.instantiate(config.network)

    def load_tts_weights(self, checkpoint_name: str) -> None:
        tts_checkpoint_path = os.path.join(
            self.tts_exp_dir, "checkpoints", checkpoint_name
        )
        if self.tts_checkpoint_path == tts_checkpoint_path:
            return
        self.tts_checkpoint_path = tts_checkpoint_path
        state_dict = torch.load(self.tts_checkpoint_path, map_location="cpu")[
            "state_dict"
        ]
        tts_state_dict = {k.replace("network.", ""): v for k, v in state_dict.items()}

        inconsistent_keys = self.tts_network.load_state_dict(
            tts_state_dict, strict=False
        )
        assert (
            len(
                [
                    key
                    for key in inconsistent_keys.missing_keys
                    if "wav2vec2" not in key and "w2v_3rd" not in key
                ]
            )
            == 0
        )
        _ = self.tts_network.to(self.device)
        _ = self.tts_network.eval()

    def __call__(
        self,
        text,
        style_audio,
        phoneme_duration_f_sequence=None,
    ):
        if phoneme_duration_f_sequence is not None:
            phoneme_duration_f_sequence = phoneme_duration_f_sequence.reshape(
                (1, -1)
            ).to(self.device)

        phoneme_sequence = (
            self.tts_network.text_processor.text_to_encoded_phoneme_sequence(text).to(
                self.device
            )
        )

        _, tts_decoding_features = self.tts_network(
            phoneme_sequence.reshape((1, -1)),
            style_audio.reshape((1, -1)).to(self.device),
            phoneme_duration_f_sequence,
        )
        timbre_global, timbre_bank = self.generator.analyze_timbre(style_audio)
        _, signal = self.generator.synthesize(
            tts_decoding_features["pitch"].squeeze(1),
            tts_decoding_features["p_amp"].squeeze(1),
            tts_decoding_features["ap_amp"].squeeze(1),
            tts_decoding_features["linguistic"],
            timbre_global,
            timbre_bank,
        )
        return signal


def main(
    exp_dir: Annotated[str, typer.Argument(help="path TTS training directory")],
    checkpoint_name: Annotated[str, typer.Argument(help="name of TTS checkpoint")],
    audio_path: Annotated[
        str, typer.Argument(help="path to audio sample of voice to generate")
    ],
    text: Annotated[str, typer.Argument(help="text to generate")],
    out_path: Annotated[str, typer.Argument(help="path to save generated audio")],
    device: Annotated[str, typer.Option("-d", help="text to generate")] = "cuda:0",
):
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    inferencer = TextToSpeechInferencer(device=device)
    inferencer.instantiate_tts(exp_dir)
    inferencer.load_tts_weights(checkpoint_name)
    inferencer.instantiate_backbone_from_tts()
    audio, sample_rate = torchaudio.load(audio_path)
    if sample_rate != inferencer.generator.input_sample_rate:
        audio = T.Resample(sample_rate, inferencer.generator.input_sample_rate)(audio)
    signal = inferencer(text, audio.to(device))
    torchaudio.save(out_path, signal.cpu(), inferencer.generator.output_sample_rate)
    print(f"Synthesized audio saved at: {out_path}")


if __name__ == "__main__":
    typer.run(main)
