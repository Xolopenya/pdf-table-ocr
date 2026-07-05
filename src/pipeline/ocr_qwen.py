"""Qwen-VL через OpenRouter — РАСКЛАДКА строк/колонок таблицы и key-value форм.

Роль (по ТЗ): Qwen силён в раскладке (сколько проходок, куда легли значения) и
разметке «-ll-»/пустых. Его ТЕКСТУ рукописи НЕ доверяем — текст ячеек читает
Yandex (см. hybrid). Схемный промпт даёт явный список колонок нужной таблицы,
temperature 0, строго один JSON без markdown.

Инфраструктура: дисковый кэш по хэшу (изображение+промпт+модель), ретраи/таймаут,
dry_run. Ключ OPENROUTER_API_KEY — из Secrets, не логируется.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from typing import Any

import cv2
import numpy as np

from ..config import Config
from ..logging_setup import logger
from .forms_layout import Spec

LOW_CONF = 0.4


def leaf_labels(spec: Spec) -> list[str]:
    """Плоский список печатных заголовков колонок спеки (в порядке слева-направо)."""
    out: list[str] = []
    for seg in spec.columns:
        if seg[0] == "single":
            out.append(seg[1])
        else:
            for sub in seg[2]:
                out.append(f"{seg[1]} — {sub[0]}")
    return out


def _table_prompt(spec: Spec) -> str:
    cols = leaf_labels(spec)
    col_lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cols))
    labelled = ""
    if spec.row_labels:
        rl = "; ".join(spec.row_labels)
        labelled = (f"\nСтроки таблицы ФИКСИРОВАНЫ и идут именно в этом порядке "
                    f"(это графа 1): {rl}. Верни ровно {len(spec.row_labels)} строк.")
    return (
        "Ты — OCR-раскладчик заполненного от руки бланка бурового журнала "
        "(рус. язык, золотодобыча). На изображении — таблица «"
        f"{spec.title}».\n"
        f"В таблице РОВНО {len(cols)} колонок слева-направо:\n{col_lines}\n"
        f"{labelled}\n\n"
        "Верни ПО ОДНОЙ СТРОКЕ на каждую запись-проходку, СТРОГО "
        f"{len(cols)} значений в строке — по одному на КАЖДУЮ графу в указанном "
        "порядке. НИКОГДА не пропускай графу и не сдвигай значения.\n"
        "Правила (КРИТИЧНО для попадания значения в свою графу):\n"
        "• Каждой графе — ровно один элемент массива, даже если она пустая.\n"
        "• Пустая графа -> \"\". Графа «литологический разрез» — это ГРАФИЧЕСКИЕ "
        "символы (точки/штриховка), НЕ текст: почти всегда ставь \"\" на её позицию.\n"
        "• Повтор «то же» (значок -ll-, —«—», —''—, кавычки, длинный прочерк) -> "
        "строкой \"-ll-\". НЕ подставляй вместо повтора значение из строки выше.\n"
        "• Числа переписывай с запятыми (например 3,6). Ничего не выдумывай.\n"
        "• Категория породы — римские цифры (I..VI). Объём — целое число.\n"
        "• Координаты сомнительных ячеек — в low_conf парами [строка, столбец] "
        "(нумерация с 0 внутри rows).\n\n"
        "Верни СТРОГО один JSON без пояснений и без ```:\n"
        '{"rows": [["c1","c2", ...], ...], "low_conf": [[r,c], ...]}'
    )


def _kv_prompt(field_names: list[str], title: str) -> str:
    fields = "\n".join(f"- {f}" for f in field_names)
    return (
        "Ты — OCR заполненного от руки бланка (рус. язык, буровой журнал, "
        f"золотодобыча). На изображении — форма «{title}».\n"
        "Извлеки значения следующих полей (печатная подпись поля -> рукописное "
        f"значение). Если поле пустое или нечитаемо — \"\".\n{fields}\n\n"
        "Числа с запятыми. Ничего не выдумывай. Верни СТРОГО один JSON без ``` "
        'вида {"fields": {"<имя поля>": "<значение>"}, "low_conf": ["<имя поля>", ...]}.'
    )


class QwenVL:
    def __init__(self, cfg: Config, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        qc = cfg.get("ocr_qwen", {})
        self.endpoint = qc.get("endpoint")
        self.model = qc.get("model")
        self.temperature = qc.get("temperature", 0)
        self.max_tokens = qc.get("max_tokens", 8192)
        self.timeout = qc.get("timeout_sec", 45)
        self.max_retries = qc.get("max_retries", 3)
        self.backoff = qc.get("backoff_base_sec", 2.0)
        self.backoff_max = qc.get("backoff_max_sec", 10)
        self.max_edge = qc.get("max_edge_px", 2200)
        self.use_cache = qc.get("use_cache", True)
        self.cache_dir = cfg.path_of("paths.cache_dir") / "qwen_vl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._secrets = cfg.secrets
        self.stats = {"call": 0, "cache": 0, "fail": 0}

    # ---- низкий уровень ----
    def _b64(self, img: np.ndarray) -> tuple[str, bytes]:
        h, w = img.shape[:2]
        long = max(h, w)
        if long > self.max_edge:
            k = self.max_edge / long
            img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if not ok:
            raise RuntimeError("Qwen: не удалось закодировать PNG")
        raw = buf.tobytes()
        return base64.b64encode(raw).decode("ascii"), raw

    def _key(self, raw: bytes, prompt: str) -> str:
        h = hashlib.sha256(raw)
        h.update(prompt.encode())
        h.update(self.model.encode())
        return h.hexdigest()

    def _call(self, img: np.ndarray, prompt: str) -> dict[str, Any] | None:
        """Вернуть распарсенный JSON-ответ модели или None."""
        b64, raw = self._b64(img)
        cpath = self.cache_dir / f"{self._key(raw, prompt)}.json"
        if self.use_cache and cpath.exists():
            self.stats["cache"] += 1
            return json.loads(cpath.read_text(encoding="utf-8"))
        if self.dry_run:
            return None
        if not self._secrets.has_openrouter():
            logger.warning("Qwen: OPENROUTER_API_KEY не задан -> пропуск раскладки")
            return None
        self.stats["call"] += 1
        import requests
        headers = {"Authorization": f"Bearer {self._secrets.openrouter_api_key}",
                   "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + b64}},
            ]}],
        }
        last = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(self.endpoint, headers=headers,
                                  data=json.dumps(body), timeout=self.timeout)
                if r.status_code == 200:
                    txt = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
                    parsed = _extract_json(txt)
                    if self.use_cache and parsed is not None:
                        cpath.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
                    return parsed
                if r.status_code in (429, 500, 502, 503):
                    last = f"HTTP {r.status_code}"
                    time.sleep(min(self.backoff ** attempt, self.backoff_max))
                    continue
                self.stats["fail"] += 1
                logger.error(f"Qwen: HTTP {r.status_code}")
                return None
            except Exception as e:  # noqa: BLE001
                last = type(e).__name__
                time.sleep(self.backoff ** attempt)
        logger.error(f"Qwen: не удалось после {self.max_retries} попыток ({last})")
        return None

    # ---- высокий уровень ----
    def layout_table(self, img: np.ndarray, spec: Spec) -> dict[str, Any]:
        """Раскладка таблицы -> reference-dict {rows,cols,grid,conf,low_conf}."""
        ncols = spec.ncols
        data = self._call(img, _table_prompt(spec))
        rows_raw = (data or {}).get("rows") or []
        low = {tuple(p) for p in ((data or {}).get("low_conf") or [])
               if isinstance(p, (list, tuple)) and len(p) == 2}
        grid: list[list[str]] = []
        for row in rows_raw:
            row = row if isinstance(row, list) else [row]
            row = [("" if v is None else str(v)) for v in row][:ncols]
            row += [""] * (ncols - len(row))
            grid.append(row)
        conf = [[(LOW_CONF if (r, c) in low else None) for c in range(ncols)]
                for r in range(len(grid))]
        return {"rows": len(grid), "cols": ncols, "grid": grid,
                "conf": conf, "low_conf": sorted(low), "_status": "ok" if data else "empty"}

    def key_value(self, img: np.ndarray, field_names: list[str], title: str) -> dict[str, Any]:
        data = self._call(img, _kv_prompt(field_names, title))
        fields = (data or {}).get("fields") or {}
        low = set((data or {}).get("low_conf") or [])
        out = {}
        for f in field_names:
            out[f] = {"value": str(fields.get(f, "") or ""), "low_conf": f in low}
        return {"fields": out, "_status": "ok" if data else "empty"}


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
