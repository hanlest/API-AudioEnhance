"""
API REST de API-AudioEnhance.

Swagger UI:  http://127.0.0.1:8000/docs
ReDoc:       http://127.0.0.1:8000/redoc
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from resemble_enhance.branding import PROJECT_NAME
from resemble_enhance.device import describe_device
from resemble_enhance.service import DEVICE, run_denoise, run_enhance, run_enhance_mp3, run_youtube_clip, new_output_path

app = FastAPI(
    title=f"{PROJECT_NAME} API",
    description=(
        f"Mejora y denoise de voz con IA ({PROJECT_NAME}).\n\n"
        "Sube un archivo de audio o recorta un clip de YouTube y obtén WAV o MP3 procesado."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Solver(str, Enum):
    midpoint = "midpoint"
    rk4 = "rk4"
    euler = "euler"


class HealthResponse(BaseModel):
    status: str = "ok"


class InfoResponse(BaseModel):
    device: str
    device_description: str


class YoutubeClipRequest(BaseModel):
    url: str = Field(..., examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
    start_seconds: float | None = Field(None, ge=0, description="Inicio del recorte en segundos")
    end_seconds: float | None = Field(None, gt=0, description="Fin del recorte en segundos")
    start_hms: str | None = Field(None, examples=["00:01:30"], description="Alternativa: HH:MM:SS")
    end_hms: str | None = Field(None, examples=["00:02:00"], description="Alternativa: HH:MM:SS")

    @model_validator(mode="after")
    def _times(self):
        from resemble_enhance.time_utils import parse_hms

        if self.start_seconds is not None and self.end_seconds is not None:
            return self
        if self.start_hms and self.end_hms:
            self.start_seconds = parse_hms(self.start_hms)
            self.end_seconds = parse_hms(self.end_hms)
            return self
        raise ValueError("Indica start_seconds/end_seconds o start_hms/end_hms.")

    @property
    def start(self) -> float:
        assert self.start_seconds is not None
        return self.start_seconds

    @property
    def end(self) -> float:
        assert self.end_seconds is not None
        return self.end_seconds


def _cleanup(path: str) -> None:
    Path(path).unlink(missing_ok=True)


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Archivo de audio vacío.")
    dest.write_bytes(data)


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    """Comprueba que el servicio responde."""
    return HealthResponse()


@app.get("/v1/info", response_model=InfoResponse, tags=["Sistema"])
def info():
    """Dispositivo usado para inferencia (CPU/CUDA)."""
    return InfoResponse(device=DEVICE, device_description=describe_device(DEVICE))


@app.post(
    "/v1/denoise",
    tags=["Audio"],
    summary="Quitar ruido",
    response_class=FileResponse,
)
async def api_denoise(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Audio de entrada (WAV, MP3, FLAC, etc.)")],
):
    """Devuelve un WAV con denoise aplicado."""
    inp = new_output_path(Path(file.filename or "in").suffix or ".wav")
    out = new_output_path(".wav")
    try:
        await _save_upload(file, inp)
        run_denoise(inp, out)
    except Exception as e:
        inp.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        inp.unlink(missing_ok=True)

    background_tasks.add_task(_cleanup, str(out))
    return FileResponse(out, media_type="audio/wav", filename="denoised.wav")


@app.post(
    "/v1/enhance",
    tags=["Audio"],
    summary="Mejorar audio",
    response_class=FileResponse,
)
async def api_enhance(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Audio de entrada")],
    nfe: Annotated[int, Query(ge=1, le=128, description="Evaluaciones del solucionador CFM")] = 64,
    solver: Annotated[Solver, Query(description="Solucionador ODE")] = Solver.midpoint,
    lambd: Annotated[float, Query(ge=0, le=1, description="Fuerza de denoise en el enhancer")] = 0.5,
    tau: Annotated[float, Query(ge=0, le=1, description="Temperatura previa CFM")] = 0.5,
):
    """Devuelve un WAV mejorado."""
    inp = new_output_path(Path(file.filename or "in").suffix or ".wav")
    out = new_output_path(".wav")
    try:
        await _save_upload(file, inp)
        run_enhance(inp, out, nfe=nfe, solver=solver.value, lambd=lambd, tau=tau)
    except Exception as e:
        inp.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        inp.unlink(missing_ok=True)

    background_tasks.add_task(_cleanup, str(out))
    return FileResponse(out, media_type="audio/wav", filename="enhanced.wav")


@app.post(
    "/v1/enhance/mp3",
    tags=["Audio"],
    summary="Mejorar audio (MP3)",
    response_class=FileResponse,
)
async def api_enhance_mp3(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Audio de entrada")],
    nfe: Annotated[int, Query(ge=1, le=128)] = 64,
    solver: Annotated[Solver, Query()] = Solver.midpoint,
    lambd: Annotated[float, Query(ge=0, le=1)] = 0.5,
    tau: Annotated[float, Query(ge=0, le=1)] = 0.5,
):
    """Devuelve un MP3 mejorado (requiere ffmpeg en PATH)."""
    inp = new_output_path(Path(file.filename or "in").suffix or ".wav")
    out = new_output_path(".mp3")
    try:
        await _save_upload(file, inp)
        run_enhance_mp3(inp, out, nfe=nfe, solver=solver.value, lambd=lambd, tau=tau)
    except Exception as e:
        inp.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        inp.unlink(missing_ok=True)

    background_tasks.add_task(_cleanup, str(out))
    return FileResponse(out, media_type="audio/mpeg", filename="enhanced.mp3")


@app.post(
    "/v1/youtube/clip",
    tags=["YouTube"],
    summary="Descargar recorte de YouTube",
    response_class=FileResponse,
)
def api_youtube_clip(body: YoutubeClipRequest, background_tasks: BackgroundTasks):
    """
    Descarga el audio de un video de YouTube y devuelve solo el tramo indicado (WAV).

    Requiere `ffmpeg` y `yt-dlp`. La descarga completa puede tardar en videos largos.
    """
    if body.end <= body.start:
        raise HTTPException(status_code=400, detail="end debe ser mayor que start.")
    out = new_output_path(".wav")
    try:
        run_youtube_clip(body.url, body.start, body.end, out)
    except Exception as e:
        out.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

    background_tasks.add_task(_cleanup, str(out))
    return FileResponse(out, media_type="audio/wav", filename="youtube_clip.wav")


def main():
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(
            __import__("os").environ.get("AUDIOENHANCE_API_PORT")
            or __import__("os").environ.get("RESEMBLE_API_PORT")
            or "8000"
        ),
        reload=False,
    )


if __name__ == "__main__":
    main()
