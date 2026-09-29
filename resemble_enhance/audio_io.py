"""Load/save audio without TorchCodec (TorchAudio 2.9+ default on Windows is fragile)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor

PathLike = Union[str, Path]


def load_audio(path: PathLike) -> tuple[Tensor, int]:
    """Return waveform (channels, time) float32 and sample rate."""
    path = Path(path)
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        wav = torch.from_numpy(data.T.copy())
    except (RuntimeError, OSError, sf.LibsndfileError):
        y, sr = librosa.load(str(path), sr=None, mono=False)
        if y.ndim == 1:
            wav = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
        else:
            wav = torch.from_numpy(y.astype(np.float32))

    return wav, int(sr)


def save_audio(path: PathLike, wav: Tensor, sr: int, channels_first: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = wav.detach().cpu()
    if x.dim() == 1:
        data = x.numpy()
    elif channels_first:
        data = x.numpy().T
    else:
        data = x.numpy()
    sf.write(str(path), data, sr)


def write_mp3(path: PathLike, wav: np.ndarray, sr: int) -> Path:
    """Encode mono/stereo float waveform to MP3 via ffmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg no está en el PATH (necesario para exportar MP3).")

    x = np.asarray(wav, dtype=np.float32)
    if x.ndim == 1:
        data = x
    else:
        data = x.T

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        sf.write(str(wav_path), data, sr)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(wav_path),
                "-codec:a",
                "libmp3lame",
                "-qscale:a",
                "2",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        wav_path.unlink(missing_ok=True)
    return path
