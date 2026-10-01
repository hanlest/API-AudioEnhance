"""
API REST de API-AudioEnhance.

Swagger UI:  http://127.0.0.1:8000/docs  (OpenAPI en /docs/openapi.json)
Health:      http://127.0.0.1:8000/health
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from api_logging import uvicorn_log_config
from api_sse import processing_sse_response, take_job
from resemble_enhance.branding import PROJECT_NAME
from resemble_enhance.device import describe_device
from resemble_enhance.service import (
    DEVICE,
    ensure_work_dir,
    new_output_path,
    run_denoise,
    run_enhance,
    run_enhance_mp3,
    run_youtube_clip,
    write_upload_bytes,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _normalize_base_path(raw: str | None) -> str:
    if not raw or not str(raw).strip():
        return ""
    path = str(raw).strip()
    if not path.startswith("/"):
        path = "/" + path
    return path.rstrip("/") or ""


# Ruta base de la documentación OpenAPI (Swagger/ReDoc/esquema). No prefija /health ni /v1/*.
DOCS_PATH = _normalize_base_path(os.environ.get("BASE_PATH")) or "/docs"
OPENAPI_URL = f"{DOCS_PATH}/openapi.json"
REDOC_URL = f"{DOCS_PATH}/redoc"

_api_description = (
    f"Mejora y denoise de voz con IA ({PROJECT_NAME}).\n\n"
    "Sube un archivo de audio o recorta un clip de YouTube y obtén WAV o MP3 procesado."
)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    work = ensure_work_dir()
    print(f"\033[33m[{PROJECT_NAME}] Upload/work dir: {work}\033[0m", flush=True)
    yield


app = FastAPI(
    title=f"{PROJECT_NAME} API",
    description=_api_description,
    version="1.0.0",
    docs_url=DOCS_PATH,
    redoc_url=REDOC_URL,
    openapi_url=OPENAPI_URL,
    lifespan=_lifespan,
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


_LAMBD_DESCRIPTION = (
    "Mezcla de denoise en el enhancer (0–1). Valores bajos (~0.1) preservan timbre; "
    "valores altos pueden sonar artificial o cambiar el acento."
)


def optional_lambd(
    lambd_query: Annotated[
        float | None,
        Query(ge=0, le=1, description=_LAMBD_DESCRIPTION),
    ] = None,
    lambd_form: Annotated[
        float | None,
        Form(ge=0, le=1, description=_LAMBD_DESCRIPTION),
    ] = None,
) -> float:
    """Query `lambd` o campo form `lambd` en el multipart; por defecto 0.1."""
    if lambd_form is not None:
        return lambd_form
    if lambd_query is not None:
        return lambd_query
    return 0.1


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
    write_upload_bytes(dest, data)


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    """Comprueba que el servicio responde."""
    return HealthResponse()


@app.get("/v1/info", response_model=InfoResponse, tags=["Sistema"])
def info():
    """Dispositivo usado para inferencia (CPU/CUDA)."""
    return InfoResponse(device=DEVICE, device_description=describe_device(DEVICE))


@app.get(
    "/v1/jobs/{job_id}/download",
    tags=["Audio"],
    summary="Descargar resultado de un job SSE",
    response_class=FileResponse,
)
def download_job(job_id: str, background_tasks: BackgroundTasks):
    """Obtiene el archivo generado tras un POST con `stream=true`. El job solo se puede consumir una vez."""
    job = take_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado o ya descargado.")
    path: Path = job["path"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Archivo de resultado no disponible.")
    background_tasks.add_task(_cleanup, str(path))
    return FileResponse(path, media_type=job["media_type"], filename=job["filename"])


@app.post(
    "/v1/denoise",
    tags=["Audio"],
    summary="Quitar ruido",
)
async def api_denoise(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Audio de entrada (WAV, MP3, FLAC, etc.)")],
    stream: Annotated[
        bool,
        Query(description="Si es true, responde SSE con progreso y `download_url` al finalizar."),
    ] = False,
):
    """Devuelve un WAV con denoise aplicado, o un stream SSE si `stream=true`."""
    inp = new_output_path(Path(file.filename or "in").suffix or ".wav")
    out = new_output_path(".wav")
    await _save_upload(file, inp)

    if stream:

        def worker(progress_callback):
            run_denoise(inp, out, progress_callback=progress_callback)
            return out

        return processing_sse_response(
            worker,
            media_type="audio/wav",
            filename="denoised.wav",
            on_finished=lambda: inp.unlink(missing_ok=True),
        )

    try:
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
)
async def api_enhance(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Audio de entrada")],
    nfe: Annotated[int, Query(ge=1, le=128, description="Evaluaciones del solucionador CFM")] = 64,
    solver: Annotated[Solver, Query(description="Solucionador ODE")] = Solver.midpoint,
    lambd: float = Depends(optional_lambd),
    tau: Annotated[float, Query(ge=0, le=1, description="Temperatura previa CFM")] = 0.5,
    stream: Annotated[bool, Query(description="SSE de progreso y `download_url` al terminar.")] = False,
):
    """Devuelve un WAV mejorado, o SSE si `stream=true`. Parámetro opcional `lambd` (query o form)."""
    inp = new_output_path(Path(file.filename or "in").suffix or ".wav")
    out = new_output_path(".wav")
    await _save_upload(file, inp)

    if stream:

        def worker(progress_callback):
            run_enhance(
                inp,
                out,
                nfe=nfe,
                solver=solver.value,
                lambd=lambd,
                tau=tau,
                progress_callback=progress_callback,
            )
            return out

        return processing_sse_response(
            worker,
            media_type="audio/wav",
            filename="enhanced.wav",
            on_finished=lambda: inp.unlink(missing_ok=True),
        )

    try:
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
)
async def api_enhance_mp3(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Audio de entrada")],
    nfe: Annotated[int, Query(ge=1, le=128)] = 64,
    solver: Annotated[Solver, Query()] = Solver.midpoint,
    lambd: float = Depends(optional_lambd),
    tau: Annotated[float, Query(ge=0, le=1)] = 0.5,
    stream: Annotated[bool, Query(description="SSE de progreso y `download_url` al terminar.")] = False,
):
    """Devuelve un MP3 mejorado (requiere ffmpeg en PATH), o SSE si `stream=true`."""
    inp = new_output_path(Path(file.filename or "in").suffix or ".wav")
    out = new_output_path(".mp3")
    await _save_upload(file, inp)

    if stream:

        def worker(progress_callback):
            run_enhance_mp3(
                inp,
                out,
                nfe=nfe,
                solver=solver.value,
                lambd=lambd,
                tau=tau,
                progress_callback=progress_callback,
            )
            return out

        return processing_sse_response(
            worker,
            media_type="audio/mpeg",
            filename="enhanced.mp3",
            on_finished=lambda: inp.unlink(missing_ok=True),
        )

    try:
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
            os.environ.get("AUDIOENHANCE_API_PORT")
            or os.environ.get("RESEMBLE_API_PORT")
            or "8000"
        ),
        reload=False,
        log_config=uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
