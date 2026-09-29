import os

import torch


def get_inference_device() -> str:
    """cuda when available unless AUDIOENHANCE_DEVICE (o RESEMBLE_DEVICE)=cpu."""
    prefer = (
        os.environ.get("AUDIOENHANCE_DEVICE")
        or os.environ.get("RESEMBLE_DEVICE")
        or "cuda"
    ).strip().lower()
    if prefer in ("cpu", "none"):
        return "cpu"
    if torch.cuda.is_available():
        if prefer.startswith("cuda"):
            return prefer
        return "cuda"
    return "cpu"


def describe_device(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.cuda.current_device() if device == "cuda" else int(device.split(":")[1])
        name = torch.cuda.get_device_name(idx)
        return f"GPU: {name} ({device})"
    return f"CPU (PyTorch {torch.__version__})"
