"""Настройка loguru. Секреты в логи не попадают.

Простой перехватчик маскирует любые Api-Key / ключи, если они случайно
окажутся в форматируемой строке.
"""
from __future__ import annotations

import re
import sys

from loguru import logger

_SECRET_PATTERNS = [
    re.compile(r"(Api-Key\s+)([A-Za-z0-9._\-]+)", re.IGNORECASE),
    re.compile(r"(AQVN[A-Za-z0-9._\-]{10,})"),   # характерный префикс Yandex ключей
    re.compile(r"(sk-or-[A-Za-z0-9._\-]{10,})"), # OpenRouter
]


def _mask(message: str) -> str:
    out = message
    for pat in _SECRET_PATTERNS:
        if pat.groups >= 2:
            out = pat.sub(lambda m: m.group(1) + "***", out)
        else:
            out = pat.sub("***", out)
    return out


_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    logger.remove()

    def sink(msg):  # маскируем перед выводом
        sys.stderr.write(_mask(str(msg)))

    logger.add(
        sink,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=False,
    )
    _configured = True


__all__ = ["logger", "setup_logging"]
