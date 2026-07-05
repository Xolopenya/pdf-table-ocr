"""Клиент Yandex Vision OCR (на базе reference/dedmos_ocr, расширен).

Добавлено против референса: дисковый кэш по хэшу (изображение+модель+языки),
ретраи с бэкоффом и таймаут, режим dry_run, ключи из src.config.Secrets
(имена YANDEX_OCR_API_KEY / YANDEX_API_KEY). Ключи не логируются.

recognize_cached(...) -> нормализованный dict:
  {width, height, full_text, lines:[{text,bbox,conf}], tables:[{rows,cols,cells:[{r,c,rs,cs,text,bbox,conf}]}]}
model="handwritten" — русская рукопись; model="table" — сетка (cells r/c/rs/cs).
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import Config
from ..logging_setup import logger

OCR_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
MAX_PIXELS = 18_000_000
MAX_BYTES = 9_500_000


class YandexOCRError(RuntimeError):
    pass


# ---------- кодирование изображения (из референса) ----------
def _encode_png(img: np.ndarray) -> tuple[str, bytes]:
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS:
        k = (MAX_PIXELS / (h * w)) ** 0.5
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise YandexOCRError("Не удалось закодировать изображение в PNG.")
    raw = buf.tobytes()
    if len(raw) > MAX_BYTES:
        for q in (95, 90, 85, 75):
            ok, jb = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok and jb.nbytes <= MAX_BYTES:
                raw = jb.tobytes()
                break
    return base64.b64encode(raw).decode("ascii"), raw


def _bbox(node: dict[str, Any]) -> list[int] | None:
    bb = (node or {}).get("boundingBox") or {}
    verts = bb.get("vertices") or []
    xs, ys = [], []
    for v in verts:
        try:
            xs.append(int(float(v.get("x", 0))))
            ys.append(int(float(v.get("y", 0))))
        except (TypeError, ValueError):
            pass
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _conf(node: dict[str, Any]) -> float | None:
    for key in ("confidence", "conf", "score"):
        if key in node:
            try:
                return float(node[key])
            except (TypeError, ValueError):
                return None
    return None


def normalize_annotation(ann: dict[str, Any]) -> dict[str, Any]:
    width = int(float(ann.get("width", 0) or 0))
    height = int(float(ann.get("height", 0) or 0))
    lines: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    for block in ann.get("blocks", []) or []:
        for ln in block.get("lines", []) or []:
            for w in ln.get("words", []) or []:
                wt = (w.get("text") or "").strip()
                wb = _bbox(w)
                if wt and wb:
                    words.append({"text": wt, "bbox": wb, "conf": _conf(w)})
            text = (ln.get("text") or "").strip()
            if not text:
                text = " ".join((w.get("text") or "")
                                for w in (ln.get("words") or [])).strip()
            if not text:
                continue
            lines.append({"text": text, "bbox": _bbox(ln), "conf": _conf(ln)})
    tables: list[dict[str, Any]] = []
    for tbl in ann.get("tables", []) or []:
        cells = []
        for cell in tbl.get("cells", []) or []:
            cells.append({
                "r": int(float(cell.get("rowIndex", 0) or 0)),
                "c": int(float(cell.get("columnIndex", 0) or 0)),
                "rs": int(float(cell.get("rowSpan", 1) or 1)),
                "cs": int(float(cell.get("columnSpan", 1) or 1)),
                "text": (cell.get("text") or "").strip(),
                "bbox": _bbox(cell), "conf": _conf(cell),
            })
        tables.append({
            "rows": int(float(tbl.get("rowCount", 0) or 0)),
            "cols": int(float(tbl.get("columnCount", 0) or 0)),
            "bbox": _bbox(tbl), "cells": cells,
        })
    full_text = ann.get("fullText") or "\n".join(l["text"] for l in lines)
    return {"width": width, "height": height, "full_text": full_text,
            "lines": lines, "words": words, "tables": tables}


_EMPTY = {"width": 0, "height": 0, "full_text": "", "lines": [], "words": [], "tables": []}


class YandexOCR:
    def __init__(self, cfg: Config, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        oc = cfg.get("ocr_yandex", {})
        self.languages = oc.get("language_codes", ["ru", "en"])
        self.timeout = oc.get("timeout_sec", 30)
        self.max_retries = oc.get("max_retries", 3)
        self.backoff = oc.get("backoff_base_sec", 1.5)
        self.backoff_max = oc.get("backoff_max_sec", 8)
        self.use_cache = oc.get("use_cache", True)
        self.cache_dir = cfg.path_of("paths.cache_dir") / "yandex_ocr"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._secrets = cfg.secrets
        self.stats = {"call": 0, "cache": 0, "fail": 0}   # для видимого прогресса

    def _key(self, raw: bytes, model: str) -> str:
        h = hashlib.sha256(raw)
        h.update(model.encode())
        h.update(",".join(self.languages).encode())
        return h.hexdigest()

    def recognize_cached(self, img: np.ndarray, model: str = "handwritten") -> dict[str, Any]:
        b64, raw = _encode_png(img)
        cpath = self.cache_dir / f"{self._key(raw, model)}.json"
        if self.use_cache and cpath.exists():
            self.stats["cache"] += 1
            return json.loads(cpath.read_text(encoding="utf-8"))
        if self.dry_run:
            return dict(_EMPTY, _status="dry_run")
        if not self._secrets.has_yandex():
            logger.warning("Yandex OCR: ключи не заданы -> пропуск")
            return dict(_EMPTY, _status="no_keys")

        self.stats["call"] += 1
        import requests
        headers = {
            "Authorization": f"Api-Key {self._secrets.yandex_api_key}",
            "x-folder-id": self._secrets.yandex_folder_id,
            "Content-Type": "application/json",
            "x-data-logging-enabled": "false",
        }
        body = {"mimeType": "image/png", "languageCodes": self.languages,
                "model": model, "content": b64}
        last = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(OCR_URL, headers=headers,
                                  data=json.dumps(body), timeout=self.timeout)
                if r.status_code == 200:
                    ann = r.json().get("result", {}).get("textAnnotation")
                    if ann is None:
                        return dict(_EMPTY, _status="no_annotation")
                    norm = normalize_annotation(ann)
                    if self.use_cache:
                        cpath.write_text(json.dumps(norm, ensure_ascii=False),
                                         encoding="utf-8")
                    return norm
                if r.status_code in (429, 500, 502, 503):
                    last = f"HTTP {r.status_code}"
                    time.sleep(min(self.backoff ** attempt, self.backoff_max))
                    continue
                self.stats["fail"] += 1
                logger.error(f"Yandex OCR: HTTP {r.status_code}")
                return dict(_EMPTY, _status=f"http_{r.status_code}", needs_review=True)
            except Exception as e:  # noqa: BLE001
                last = type(e).__name__
                time.sleep(min(self.backoff ** attempt, self.backoff_max))
        self.stats["fail"] += 1
        logger.error(f"Yandex OCR: не удалось после {self.max_retries} попыток ({last})")
        return dict(_EMPTY, _status="retries_exhausted", needs_review=True)

    def words_of(self, img: np.ndarray) -> list[dict]:
        """Слова с bbox (в координатах переданного изображения) — handwritten."""
        return self.recognize_cached(img, model="handwritten").get("words", [])

    def read_cell(self, crop: np.ndarray, target: int = 130) -> tuple[str, float | None]:
        """Прочитать текст одной ячейки (handwritten). Возвращает (text, conf).

        Мелкие кропы (узкие графы № / глубина) апскейлим до target по короткой
        стороне — Yandex заметно лучше читает одиночные цифры/буквы увеличенными.
        Больший target — для повторного точечного добора бледных ячеек."""
        h, w = crop.shape[:2]
        short = min(h, w)
        if 0 < short < target:
            k = target / short
            crop = cv2.resize(crop, (int(w * k), int(h * k)), interpolation=cv2.INTER_CUBIC)
        norm = self.recognize_cached(crop, model="handwritten")
        txt = (norm.get("full_text") or "").strip()
        confs = [l["conf"] for l in norm.get("lines", []) if l.get("conf") is not None]
        conf = sum(confs) / len(confs) if confs else None
        return txt, conf
