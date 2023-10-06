import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import fleep
import librosa
import streamlit as st
import torch
import torchaudio.transforms as T


def select_checkpoint(log_dir: str) -> Tuple[Any, Any, Any, Any]:
    cfg_options = sorted(os.listdir(log_dir))
    cfg_name = st.sidebar.selectbox("Select config", options=cfg_options, index=0)
    exp_options = sorted(
        os.listdir(os.path.join(log_dir, cfg_name)) if cfg_name is not None else [],
        key=lambda date_string: datetime.strptime(
            "-".join(date_string.split("-")[:5]), "%Y-%m-%d_%H-%M-%S"
        ),
    )
    exp_date = st.sidebar.selectbox("Select experience", options=exp_options, index=0)
    exp_dir = os.path.join(log_dir, cfg_name, exp_date)
    ckpt_options = sorted(
        [
            ckpt_name
            for ckpt_name in os.listdir(os.path.join(exp_dir, "checkpoints"))
            if "last" not in ckpt_name
        ]
        if os.path.isdir(os.path.join(exp_dir, "checkpoints"))
        else [],
        key=lambda ckpt_name: int(os.path.splitext(ckpt_name)[0].split("=")[-1]),
    )
    ckpt_name = st.sidebar.selectbox("Select checkpoint", options=ckpt_options, index=0)
    return cfg_name, exp_date, ckpt_name, exp_dir


def load_audio(
    audio_name: Optional[str],
    sample_rate: int,
    tmp_dir: str,
    sample_paths: Dict[str, str],
    key: int,
) -> Tuple[Optional[str], Optional[str], Optional[torch.Tensor]]:
    audio_selection_method = st.selectbox(
        "Audio selection method",
        options=["Upload an audio file", "Choose from VCTK dataset"],
        index=1,
        key=100 * key,
    )
    audio_path, audio = None, None
    if audio_selection_method == "Upload an audio file":
        uploaded_file = st.file_uploader(
            "", type=["wav", "mp3", "m4a", "flac", "ogg"], key=100 * key + 1
        )
        if uploaded_file is not None:
            ext = fleep.get(uploaded_file.read(128)).extension[0]
            audio_path = f"{tmp_dir}/{audio_name}.{ext}"
            with open(audio_path, "wb") as file:
                file.write(uploaded_file.getbuffer())
            st.audio(audio_path)
    elif audio_selection_method == "Choose from VCTK dataset":
        audio_name = st.selectbox(
            "List of audios",
            options=list(sample_paths.keys()),
            key=100 * key + 2,
            index=key,
        )
        if audio_name is not None:
            audio_path = sample_paths[audio_name]
            st.audio(audio_path)
    if audio_path is not None:
        audio = torch.from_numpy(
            librosa.load(audio_path, sr=sample_rate, mono=True)[0]
        ).unsqueeze(0)

    return audio_name, audio_path, audio
