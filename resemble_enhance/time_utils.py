"""Conversión entre segundos y HH:MM:SS."""


def format_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_cs = int(round(seconds * 100))
    h, rem = divmod(total_cs, 3600 * 100)
    m, rem = divmod(rem, 60 * 100)
    s = rem / 100.0
    if abs(s - round(s)) < 1e-6:
        return f"{h:02d}:{m:02d}:{int(round(s)):02d}"
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def parse_hms(text: str) -> float:
    text = str(text).strip()
    if not text:
        raise ValueError("Indica un tiempo (HH:MM:SS).")
    if ":" not in text:
        return float(text.replace(",", "."))
    parts = text.split(":")
    if len(parts) == 2:
        minutes, sec = parts
        return int(minutes) * 60 + float(sec.replace(",", "."))
    if len(parts) == 3:
        hours, minutes, sec = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(sec.replace(",", "."))
    raise ValueError("Formato inválido. Usa HH:MM:SS (ej. 01:23:45).")


def split_hms_fields(seconds: float) -> tuple[str, str, str]:
    hms = format_hms(seconds)
    hours, minutes, seconds_part = hms.split(":")
    return hours, minutes, seconds_part


def fields_to_seconds(hours: str, minutes: str, seconds: str) -> float:
    h = str(hours).strip() or "0"
    m = str(minutes).strip() or "0"
    s = str(seconds).strip() or "0"
    return parse_hms(f"{h}:{m}:{s}")
