import os

import pyrootutils
import soundfile as sf
import streamlit as st
import torch

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from app.components import load_audio, select_checkpoint
from src.dataclasses.backbone import RunDirBase
from src.inference.tts import TextToSpeechInferencer

st.title("Nansy-TTS")
st.write("")


log_dir = os.path.join(root, "static/runs", RunDirBase.RUNS_TTS)
sample_dir = os.path.join(root, "static/samples/vctk")
tmp_dir = os.path.join(root, "static/tmp")
os.makedirs(tmp_dir, exist_ok=True)

sample_paths = {
    "p238 [female]": f"{sample_dir}/p238_001.wav",
    "p248 [female]": f"{sample_dir}/p248_002.wav",
    "p261 [female]": f"{sample_dir}/p261_003.wav",
    "p326 [male]": f"{sample_dir}/p326_004.wav",
    "p347 [male]": f"{sample_dir}/p347_005.wav",
}


# Select device
cuda_device_ordinal = st.sidebar.selectbox(
    "Select GPU", options=list(range(torch.cuda.device_count())), index=0
)
device = f"cuda:{cuda_device_ordinal}"


@st.cache_resource()
def load_nansytts(
    exp_dir: str,
    ckpt_name: str,
) -> TextToSpeechInferencer:
    inferencer = TextToSpeechInferencer(device)
    inferencer.instantiate_tts(exp_dir)
    inferencer.load_tts_weights(ckpt_name)
    inferencer.instantiate_backbone_from_tts()
    return inferencer


_, exp_date, ckpt_name, exp_dir = select_checkpoint(log_dir)

if exp_dir is not None and ckpt_name is not None:
    inferencer = load_nansytts(
        exp_dir,
        ckpt_name,
    )
    input_sr = inferencer.generator.input_sample_rate
    output_sr = inferencer.generator.output_sample_rate
    st.info("Style audio")
    style_name, style_path, style_audio = load_audio(
        "uploaded_style", input_sr, tmp_dir, sample_paths, 0
    )
    style_audio = style_audio.to(device)
    st.info("Text input")
    text = st.text_input("Write text", "To be or not to be that is the question")
    st.info("Generated audio")
    path = os.path.join(tmp_dir, "tts_output.wav")
    signal = inferencer(text, style_audio)
    sf.write(path, signal.squeeze(0).detach().cpu().numpy(), samplerate=output_sr)
    st.audio(path)
    torch.cuda.empty_cache()
