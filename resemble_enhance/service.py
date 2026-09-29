"""Lógica compartida de inferencia para la demo Gradio y la API."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch

from .audio_io import load_audio, save_audio, write_mp3
from .branding import PROJECT_SLUG
from .device import get_inference_device
from .enhancer.inference import denoise, enhance
from .youtube_audio import download_audio_clip

DEVICE = get_inference_device()
WORK = Path(tempfile.gettempdir()) / PROJECT_SLUG / "api"
WORK.mkdir(parents=True, exist_ok=True)


def _mono_tensor(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = load_audio(path)
    return wav.mean(dim=0), sr


def run_denoise(input_path: Path, output_path: Path) -> tuple[Path, int]:
    dwav, sr = _mono_tensor(input_path)
    out, sr = denoise(dwav, sr, DEVICE)
    save_audio(output_path, out[None], sr)
    return output_path, sr


def run_enhance(
    input_path: Path,
    output_path: Path,
    *,
    nfe: int = 64,
    solver: str = "midpoint",
    lambd: float = 0.5,
    tau: float = 0.5,
) -> tuple[Path, int]:
    dwav, sr = _mono_tensor(input_path)
    out, sr = enhance(dwav, sr, DEVICE, nfe=nfe, solver=solver.lower(), lambd=lambd, tau=tau)
    save_audio(output_path, out[None], sr)
    return output_path, sr


def run_enhance_mp3(
    input_path: Path,
    output_path: Path,
    *,
    nfe: int = 64,
    solver: str = "midpoint",
    lambd: float = 0.5,
    tau: float = 0.5,
) -> Path:
    wav_path = output_path.with_suffix(".wav")
    _, sr = run_enhance(
        input_path,
        wav_path,
        nfe=nfe,
        solver=solver,
        lambd=lambd,
        tau=tau,
    )
    data, _ = load_audio(wav_path)
    write_mp3(output_path, np.asarray(data.squeeze(0).cpu()), sr)
    wav_path.unlink(missing_ok=True)
    return output_path


def run_youtube_clip(url: str, start: float, end: float, output_path: Path) -> Path:
    clip = download_audio_clip(url.strip(), start, end)
    shutil.copy2(clip, output_path)
    return output_path


def new_output_path(suffix: str) -> Path:
    return WORK / f"{uuid.uuid4().hex}{suffix}"
