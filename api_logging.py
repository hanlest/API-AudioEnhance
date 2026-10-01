"""Logging de uvicorn en stdout con INFO en amarillo (PM2 pinta stderr en rojo)."""

from __future__ import annotations

import logging

from uvicorn.logging import AccessFormatter, DefaultFormatter

_YELLOW = "\033[33m"
_RESET = "\033[0m"


class YellowDefaultFormatter(DefaultFormatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno <= logging.INFO:
            return f"{_YELLOW}{message}{_RESET}"
        return message


class YellowAccessFormatter(AccessFormatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno <= logging.INFO:
            return f"{_YELLOW}{message}{_RESET}"
        return message


def uvicorn_log_config() -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "api_logging.YellowDefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": False,
            },
            "access": {
                "()": "api_logging.YellowAccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "class": "logging.StreamHandler",
                "formatter": "access",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }
