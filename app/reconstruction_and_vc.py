# pylint: disable-msg=C0103
# pylint: disable-msg=C0116
import os
import socket
from datetime import datetime

import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

# autoroot should be imported here for PYTHON_PATH-setup, do not remove it!
import soundfile as sf
import streamlit as st
import torch
import torchaudio
import torchaudio.transforms as T

from app.components import load_audio, select_checkpoint
from src.dataclasses.backbone import RunDirBase
from src.inference.backbone import BackboneInferencer

# Title stack
st.title("🎙️ Nansy ++")
st.write("")

machine = socket.gethostname()
task = st.sidebar.selectbox("Select task", options=["Reconstruct", "Convert"], index=1)

log_dir = os.path.join(root, "static/runs", RunDirBase.RUNS_BACKBONE)
sample_dir = os.path.join(root, "static/samples/vctk")
tmp_dir = os.path.join(root, "static/tmp")
os.makedirs(tmp_dir, exist_ok=True)

sample_paths = {
    "p225 [female]": f"{sample_dir}/p225_001.wav",
    "p226 [male]": f"{sample_dir}/p226_002.wav",
    "p227 [male]": f"{sample_dir}/p227_003.wav",
    "p228 [female]": f"{sample_dir}/p228_004.wav",
    "p229 [female]": f"{sample_dir}/p229_005.wav",
}


@st.cache_resource
def load_backbone(
    exp_dir: str, checkpoint_name: str, device: str
) -> BackboneInferencer:
    return BackboneInferencer(exp_dir, checkpoint_name, device)


# Select device
cuda_device_ordinal = st.sidebar.selectbox(
    "Select GPU", options=list(range(torch.cuda.device_count())), index=0
)
device = f"cuda:{cuda_device_ordinal}"

# Load generator
cfg_name, exp_date, ckpt_name, exp_dir = select_checkpoint(log_dir)


if ckpt_name is not None and cfg_name is not None:
    inferencer = load_backbone(exp_dir, ckpt_name, device)
    cfg, gen = inferencer.config, inferencer.generator
    input_sr = inferencer.input_sr
    output_sr = inferencer.output_sr

    source_audio, target_audio = None, None
    if task == "Reconstruct":
        st.info("Original")
        source_name, source_path, source_audio = load_audio(
            "uploaded_source", input_sr, tmp_dir, sample_paths, 0
        )
        target_name, target_audio = source_name, source_audio
    elif task == "Convert":
        st.info("Source")
        source_name, source_path, source_audio = load_audio(
            "uploaded_source", input_sr, tmp_dir, sample_paths, 0
        )
        st.info("Target")
        target_name, target_path, target_audio = load_audio(
            "uploaded_target", input_sr, tmp_dir, sample_paths, 1
        )

    if source_audio is not None and target_audio is not None and task is not None:
        # Inference stack
        st.info(task + "ed 🦜")
        raw_path = f"{tmp_dir}/{source_name}-to-{target_name}-{cfg_name}-{exp_date}-{ckpt_name}.wav"
        raw_converted, _ = inferencer.voice_conversion(
            source_audio=source_audio, target_audio=target_audio
        )
        sf.write(
            raw_path,
            raw_converted,
            samplerate=output_sr,
        )
        st.audio(raw_path)

    torch.cuda.empty_cache()
