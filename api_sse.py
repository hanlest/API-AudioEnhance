"""Utilidades SSE y almacén temporal de resultados para la API."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi.responses import StreamingResponse

_jobs: dict[str, dict[str, Any]] = {}

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def register_job(path: Path, *, media_type: str, filename: str) -> str:
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "path": path,
        "media_type": media_type,
        "filename": filename,
    }
    return job_id


def take_job(job_id: str) -> dict[str, Any] | None:
    return _jobs.pop(job_id, None)


def sse_line(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def processing_sse_response(
    worker: Callable[[Callable[[dict[str, Any]], None]], Path],
    *,
    media_type: str,
    filename: str,
    on_finished: Callable[[], None] | None = None,
) -> StreamingResponse:
    """Ejecuta `worker(progress_callback)` en un hilo y emite eventos SSE."""

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def event_generator():
        loop = asyncio.get_running_loop()
        pending_complete: dict[str, Any] | None = None

        def progress_callback(event: dict[str, Any]) -> None:
            nonlocal pending_complete
            if event.get("stage") == "complete":
                pending_complete = event
                return
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def run_worker() -> None:
            try:
                result_path = worker(progress_callback)
                job_id = register_job(result_path, media_type=media_type, filename=filename)
                payload = {
                    **(pending_complete or {"stage": "complete", "percent": 100.0}),
                    "job_id": job_id,
                    "download_url": f"/v1/jobs/{job_id}/download",
                }
                loop.call_soon_threadsafe(queue.put_nowait, payload)
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"stage": "error", "percent": 0.0, "detail": str(exc)},
                )
            finally:
                if on_finished:
                    try:
                        on_finished()
                    except Exception:
                        pass
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=run_worker, daemon=True).start()

        while True:
            event = await queue.get()
            if event is None:
                break
            yield sse_line(event)
            if event.get("stage") == "error" or event.get("download_url"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
