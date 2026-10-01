"""Lógica compartida de inferencia para la demo Gradio y la API."""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audio_io import load_audio, save_audio, write_mp3
from .device import get_inference_device
from .enhancer.inference import denoise, enhance
from .youtube_audio import download_audio_clip

DEVICE = get_inference_device()
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_work_dir() -> Path:
    override = os.environ.get("AUDIOENHANCE_WORK_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _REPO_ROOT / ".cache" / "api"


WORK = _resolve_work_dir()


def ensure_work_dir() -> Path:
    try:
        WORK.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if WORK.exists() and not WORK.is_dir():
            raise RuntimeError(f"AUDIOENHANCE work dir blocked (not a folder): {WORK}") from exc
        raise
    return WORK


ensure_work_dir()

# Una inferencia a la vez (CUDA + SSE en hilo de fondo).
INFERENCE_LOCK = threading.Lock()

ProgressCallback = Callable[[dict[str, Any]], None]


def _emit(cb: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if cb is not None:
        cb(payload)


def _mono_tensor(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = load_audio(path)
    return wav.mean(dim=0), sr


def run_denoise(
    input_path: Path,
    output_path: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, int]:
    _emit(progress_callback, {"stage": "loading_audio", "operation": "denoise", "percent": 0.0})
    dwav, sr = _mono_tensor(input_path)
    _emit(progress_callback, {"stage": "model", "operation": "denoise", "percent": 2.0})

    def _on_inference(event: dict[str, Any]) -> None:
        _emit(progress_callback, {"operation": "denoise", **event})

    with INFERENCE_LOCK:
        out, sr = denoise(
            dwav,
            sr,
            DEVICE,
            progress_callback=_on_inference if progress_callback else None,
        )
        _emit(progress_callback, {"stage": "saving", "operation": "denoise", "percent": 99.0})
        save_audio(output_path, out[None], sr)
    _emit(progress_callback, {"stage": "complete", "operation": "denoise", "percent": 100.0})
    return output_path, sr


def run_enhance(
    input_path: Path,
    output_path: Path,
    *,
    nfe: int = 64,
    solver: str = "midpoint",
    lambd: float = 0.1,
    tau: float = 0.5,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, int]:
    _emit(progress_callback, {"stage": "loading_audio", "operation": "enhance", "percent": 0.0})
    dwav, sr = _mono_tensor(input_path)
    _emit(progress_callback, {"stage": "model", "operation": "enhance", "percent": 2.0})

    def _on_inference(event: dict[str, Any]) -> None:
        _emit(progress_callback, {"operation": "enhance", **event})

    with INFERENCE_LOCK:
        out, sr = enhance(
            dwav,
            sr,
            DEVICE,
            nfe=nfe,
            solver=solver.lower(),
            lambd=lambd,
            tau=tau,
            progress_callback=_on_inference if progress_callback else None,
        )
        _emit(progress_callback, {"stage": "saving", "operation": "enhance", "percent": 99.0})
        save_audio(output_path, out[None], sr)
    _emit(progress_callback, {"stage": "complete", "operation": "enhance", "percent": 100.0})
    return output_path, sr


def run_enhance_mp3(
    input_path: Path,
    output_path: Path,
    *,
    nfe: int = 64,
    solver: str = "midpoint",
    lambd: float = 0.1,
    tau: float = 0.5,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    wav_path = output_path.with_suffix(".wav")

    def _on_wav(event: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        pct = float(event.get("percent", 0))
        if event.get("stage") == "complete":
            progress_callback({**event, "stage": "encoding", "percent": 98.0, "operation": "enhance_mp3"})
            return
        progress_callback({**event, "percent": round(pct * 0.97, 1), "operation": "enhance_mp3"})

    _, sr = run_enhance(
        input_path,
        wav_path,
        nfe=nfe,
        solver=solver,
        lambd=lambd,
        tau=tau,
        progress_callback=_on_wav if progress_callback else None,
    )
    _emit(progress_callback, {"stage": "encoding", "operation": "enhance_mp3", "percent": 98.5})
    data, _ = load_audio(wav_path)
    write_mp3(output_path, np.asarray(data.squeeze(0).cpu()), sr)
    wav_path.unlink(missing_ok=True)
    _emit(progress_callback, {"stage": "complete", "operation": "enhance_mp3", "percent": 100.0})
    return output_path


def run_youtube_clip(url: str, start: float, end: float, output_path: Path) -> Path:
    clip = download_audio_clip(url.strip(), start, end)
    shutil.copy2(clip, output_path)
    return output_path


def new_output_path(suffix: str) -> Path:
    ensure_work_dir()
    return WORK / f"{uuid.uuid4().hex}{suffix}"


def write_upload_bytes(path: Path, data: bytes) -> None:
    ensure_work_dir()
    if path.parent != WORK:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
