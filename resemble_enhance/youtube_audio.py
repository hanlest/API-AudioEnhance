from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
import yt_dlp

from .branding import PROJECT_SLUG


def _work_dir() -> Path:
    root = Path(tempfile.gettempdir()) / PROJECT_SLUG / "youtube"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _wav_duration(path: Path) -> float:
    return float(sf.info(str(path)).duration)


def _ffmpeg_trim(src: Path, dst: Path, start: float, end: float) -> Path:
    if end <= start:
        raise ValueError("El fin del recorte debe ser mayor que el inicio.")
    length = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ss",
        str(start),
        "-t",
        str(length),
        "-ac",
        "1",
        "-ar",
        "44100",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"ffmpeg no pudo recortar el audio. {detail[:500]}") from e
    return dst


def fetch_video_info(url: str) -> tuple[str, float]:
    url = url.strip()
    if not url:
        raise ValueError("Ingresa una URL de YouTube.")

    opts: dict = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title") or "Sin título"
    duration = float(info.get("duration") or 0)
    if duration <= 0:
        raise ValueError("No se pudo obtener la duración del video.")
    return title, duration


def _pick_source_file(work: Path) -> Path:
    wavs = sorted(work.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if wavs:
        return wavs[0]
    others = sorted(
        [p for p in work.iterdir() if p.is_file() and p.suffix.lower() not in {".wav", ".part"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if others:
        return others[0]
    raise RuntimeError("No se pudo descargar el audio del video.")


def download_audio_clip(url: str, start: float, end: float) -> Path:
    url = url.strip()
    if not url:
        raise ValueError("Ingresa una URL de YouTube.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg no está en el PATH (necesario para recortar audio de YouTube).")
    if end <= start:
        raise ValueError("El fin del recorte debe ser mayor que el inicio.")

    work = _work_dir()
    for old in work.iterdir():
        if old.is_file():
            old.unlink()

    out_wav = work / "clip.wav"
    base_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(work / "track.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }

    with yt_dlp.YoutubeDL(base_opts) as ydl:
        ydl.download([url])

    source = _pick_source_file(work)
    _ffmpeg_trim(source, out_wav, start, end)

    expected = end - start
    actual = _wav_duration(out_wav)
    if abs(actual - expected) > max(2.0, expected * 0.05):
        raise RuntimeError(
            f"El recorte no coincidió: duración {actual:.1f}s, se esperaban {expected:.1f}s."
        )

    return out_wav
