"""Загрузка конфигурации (YAML) и секретов (.env).

Секреты берутся ТОЛЬКО из окружения/.env и никогда не сериализуются вместе с
основным конфигом. Значения ключей нигде не логируются.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Корень проекта = родитель каталога src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class Secrets:
    """Тонкий доступ к секретам. Печать/логирование значений запрещены."""

    def __init__(self) -> None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)

    @property
    def yandex_api_key(self) -> str:
        # поддерживаем оба имени: наше (.env этого проекта) и из ТЗ/референса
        return (os.environ.get("YANDEX_OCR_API_KEY")
                or os.environ.get("YANDEX_API_KEY")
                or os.environ.get("YC_API_KEY", ""))

    @property
    def yandex_folder_id(self) -> str:
        return os.environ.get("YANDEX_FOLDER_ID", "")

    @property
    def openrouter_api_key(self) -> str:
        return os.environ.get("OPENROUTER_API_KEY", "")

    def has_yandex(self) -> bool:
        return bool(self.yandex_api_key and self.yandex_folder_id)

    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)

    def __repr__(self) -> str:  # никаких значений — только наличие
        return (
            f"Secrets(yandex={'set' if self.has_yandex() else 'missing'}, "
            f"openrouter={'set' if self.has_openrouter() else 'missing'})"
        )


class Config:
    """Обёртка над словарём конфига с удобным доступом по «a.b.c»."""

    def __init__(self, data: dict[str, Any], path: Path) -> None:
        self._data = data
        self.path = path
        self.secrets = Secrets()

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    # --- Разрешение путей относительно корня проекта --------------------
    def path_of(self, dotted: str) -> Path:
        rel = self.get(dotted)
        if rel is None:
            raise KeyError(f"Нет пути в конфиге: {dotted}")
        return (PROJECT_ROOT / rel).resolve()

    def abs_path(self, rel: str) -> Path:
        return (PROJECT_ROOT / rel).resolve()


def bootstrap_poppler(explicit_path: str | None = None) -> str | None:
    """Гарантировать наличие poppler (pdftoppm) в PATH для pdf2image.

    Порядок: явный путь -> уже в PATH -> автопоиск в типовых местах (WinGet,
    Program Files, conda). Возвращает найденный bin-каталог или None.
    """
    import glob
    import shutil

    if explicit_path and (Path(explicit_path) / "pdftoppm.exe").exists():
        os.environ["PATH"] = explicit_path + os.pathsep + os.environ.get("PATH", "")
        return explicit_path
    if shutil.which("pdftoppm"):
        return None  # уже доступен
    home = os.path.expanduser("~")
    patterns = [
        os.path.join(home, "AppData", "Local", "Microsoft", "WinGet", "Packages",
                     "*Poppler*", "poppler-*", "Library", "bin"),
        r"C:\Program Files\poppler*\Library\bin",
        r"C:\Program Files\poppler*\bin",
        os.path.join(home, "*conda*", "Library", "bin"),
    ]
    for pat in patterns:
        for cand in glob.glob(pat):
            if (Path(cand) / "pdftoppm.exe").exists():
                os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")
                return cand
    return None


@lru_cache(maxsize=8)
def load_config(config_path: str | None = None) -> Config:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, path)
