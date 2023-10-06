import json

import numpy as np
import torch
import typer
from tqdm import tqdm

from src.inference.backbone import BackboneInferencer

app = typer.Typer()


def main(
    exp_dir: str,
    ckpt_name: str,
    save_path: str = "static/tts/features_lengths.json",
    device="cuda:0",
):
    generator = BackboneInferencer(str(exp_dir), ckpt_name, device).generator

    keys = ["cqt", "linguistic"]
    sample_rates = [16000, 44100]
    all_results = {}

    for sample_rate in tqdm(sample_rates):
        results = {}
        results[0] = {"cqt": 0, "linguistic": 0}
        for audio_length in tqdm(np.arange(0.25, 25.1, 0.05)):
            results[audio_length] = {}
            len_f = int(audio_length * sample_rate)
            sample = torch.zeros((1, len_f)).to("cuda")
            features = generator.analyze(sample, enable_information_perturbator=False)
            for key in keys:
                results[audio_length][key] = features[key].shape[-1]
            assert features["pitch"].shape[-1] == features["cqt"].shape[-1]
            assert features["p_amp"].shape[-1] == features["cqt"].shape[-1]
            assert features["ap_amp"].shape[-1] == features["cqt"].shape[-1]

            assert features["timbre_global"].shape[-1] == 192
            assert features["timbre_bank"].shape[-1] == 50
        all_results[sample_rate] = results

    with open(save_path, "w") as file:
        json.dump(all_results, file)


if __name__ == "__main__":
    typer.run(main)
