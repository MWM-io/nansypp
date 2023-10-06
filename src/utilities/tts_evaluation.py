import json
import os
import string
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import autoroot
import torch
import torchaudio
import torchaudio.transforms as T
from evaluate import load
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.inference.tts import TextToSpeechInferencer


class TTSEvaluationModule:
    def __init__(
        self,
        evaluation_data: str,
    ):
        self.tts_output_sample_rate = None

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.stt_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-medium"
        )
        self.stt_model_sample_rate = 16000
        self.stt_model.config.forced_decoder_ids = None

        self.resampler = None

        if not os.path.isabs(evaluation_data):
            evaluation_data = os.path.join(autoroot.root, evaluation_data)
        with open(evaluation_data, "r") as file:
            realtive_evaluation_data = json.load(file)
        self.evaluation_data = {}
        for path, text in realtive_evaluation_data.items():
            if not os.path.isabs(path):
                path = os.path.join(autoroot.root, path)
            self.evaluation_data[path] = text
        self.style_audios = list(self.evaluation_data.keys())
        self.phrases = list(self.evaluation_data.values())

        self.cer = load("cer")

    @staticmethod
    def _process_text(text: str) -> str:
        text = " ".join(text.lower().split())
        text = "".join(list(filter(lambda i: i in string.ascii_lowercase + " ", text)))
        return text

    def compute_score(self, transcriptions: Dict[Tuple[str, str], str]) -> float:
        predictions = []
        references = []
        for (_, reference), prediction in transcriptions.items():
            references.append(self._process_text(reference))
            predictions.append(self._process_text(prediction))
        return self.cer.compute(
            predictions=predictions,
            references=references,
        )

    def generate_transcriptions(
        self, generated_audios: Dict[Tuple[str, str], torch.Tensor]
    ) -> Dict[Tuple[str, str], str]:
        self.resampler = T.Resample(
            self.tts_output_sample_rate, self.stt_model_sample_rate
        )
        transcriptions = {}
        for key, audio in tqdm(
            generated_audios.items(), total=len(generated_audios), dynamic_ncols=True
        ):
            audio = self.resampler(audio.detach().cpu())
            sample = audio.detach().cpu().numpy().reshape(-1)
            input_features = self.processor(
                sample, sampling_rate=self.stt_model_sample_rate, return_tensors="pt"
            ).input_features
            predicted_ids = self.stt_model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            transcriptions[key] = transcription
        return transcriptions

    def generate_audios(
        self,
        tts_inferencer: TextToSpeechInferencer,
        get_time: bool = False,
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        self.tts_output_sample_rate = tts_inferencer.generator.output_sample_rate
        generated_audios = {}
        start_time = datetime.now()
        for audio_file, text in tqdm(
            self.evaluation_data.items(),
            total=len(self.evaluation_data),
            dynamic_ncols=True,
        ):
            initial_audio, audio_sample_rate = torchaudio.load(audio_file)
            style_audio_tensor = T.Resample(
                audio_sample_rate, tts_inferencer.generator.input_sample_rate
            )(initial_audio).to(tts_inferencer.device)
            generated_audio = tts_inferencer(text, style_audio_tensor)
            generated_audios[(audio_file, text)] = generated_audio.detach().cpu()
            torch.cuda.empty_cache()
        end_time = datetime.now()
        exec_time = end_time - start_time
        if not get_time:
            return generated_audios
        else:
            return generated_audios, exec_time
