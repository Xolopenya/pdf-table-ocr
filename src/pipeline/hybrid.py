"""Гибридный движок — ДЕТЕРМИНИРОВАННАЯ раскладка по геометрии («вариант 1»).

Проблема индексной раскладки: пропуск пустой графы движком сдвигает все значения.
Решение — размещать по ГЕОМЕТРИИ, а не по порядку массива:

  • Колонка = X-полоса фикс-спеки (col_fracs), в которую попадает ячейка;
    пустая графа остаётся пустой, сдвига нет.
  • Строка = Y-полоса из горизонталей extract_grid; ключ строки главной таблицы —
    наличие чернил в колонке №проходок/глубина (пустые расчерченные строки не берём).
  • Слот = пересечение X-полосы и Y-полосы. Yandex читает текст КРОПА слота.
  • Qwen решает только ТИП содержания (значение/пусто/«-ll-») и служит сверкой;
    слот НЕ выбирает.
  • podschet/pokazateli/kontrol: строки фиксированы спекой (row_labels); метки не
    перезаписываются движком, значения клеятся к фикс-строкам по Y.

engine=hybrid — Yandex-текст по слотам + сверка с Qwen (расхождение → needs_review);
engine=yandex — только Yandex по слотам; engine=qwen — «сырая» индексная раскладка
Qwen (baseline для сравнения, показывает сдвиг).
"""
from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

from . import assemble
from .forms_layout import Spec
from .ocr_qwen import QwenVL
from .ocr_yandex import YandexOCR
from .preprocess import extract_grid, to_gray

HIGH_CONF = 0.95
DISAGREE_CONF = 0.45
YANDEX_EMPTY_CONF = 0.5


# ----------------------- геометрия колонок/строк --------------------------
def _col_bounds(width: int, spec: Spec) -> list[int]:
    """Границы X-полос колонок (px) относительно ширины региона объекта."""
    fr = spec.col_fracs
    if not fr:
        w = spec.widths
        tot = float(sum(w)) or 1.0
        fr = [x / tot for x in w]
    tot = float(sum(fr)) or 1.0
    xs, cum = [0], 0.0
    for f in fr:
        cum += f
        xs.append(int(round(cum / tot * width)))
    xs[-1] = width
    return xs


def _row_line_ys(obj: np.ndarray) -> list[int]:
    g = to_gray(obj)
    h, w = g.shape
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 40), 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (w // 12, 1)))
    proj = (horiz > 0).sum(axis=1)
    idx = np.where(proj >= 0.3 * w)[0]
    if idx.size == 0:
        return []
    ys, group = [], [int(idx[0])]
    for v in idx[1:]:
        if v - group[-1] <= 6:
            group.append(int(v))
        else:
            ys.append(int(round(np.mean(group))))
            group = [v]
    ys.append(int(round(np.mean(group))))
    return ys


def _uniform_bands(ys: list[int]) -> list[tuple[int, int]]:
    """Полосы строк ~равномерного шага (данные, без высоких строк шапки)."""
    if len(ys) < 2:
        return []
    ys = sorted(ys)
    diffs = np.diff(ys)
    med = float(np.median(diffs))
    return [(ys[i], ys[i + 1]) for i in range(len(ys) - 1)
            if 0.55 * med <= diffs[i] <= 1.7 * med]


def _dark_ratio(gray: np.ndarray, thr: float, box, pad: int) -> float:
    x0, y0, x1, y1 = box
    x0, y0 = max(x0 + pad, 0), max(y0 + pad, 0)
    x1, y1 = min(x1 - pad, gray.shape[1]), min(y1 - pad, gray.shape[0])
    if x1 - x0 < 6 or y1 - y0 < 6:
        return 0.0
    sub = gray[y0:y1, x0:x1]
    return float((sub < thr - 10).mean())


# ----------------------- сверка текста ------------------------------------
def _norm(s: str | None) -> str:
    if not s:
        return ""
    return "".join(ch for ch in s.lower().replace("ё", "е") if ch.isalnum())


def _agree(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


_CODE_MAP = {
    "nc": "пс", "nс": "пс", "пс": "пс", "nэ": "пс", "nз": "пс", "ne": "пс", "пc": "пс",
    "h3": "нз", "нз": "нз", "нэ": "нз", "n3": "нз",
    "cл": "сл", "сл": "сл", "ca": "сл", "сa": "сл", "cа": "сл", "cл.": "сл",
    "6/o": "б/о", "б/о": "б/о", "6/о": "б/о", "b/o": "б/о", "6о": "б/о",
}


def _dedup_numeric(grid, left_cols, right_cols) -> None:
    """Убрать дубли чисел: если в средних графах (13-15) лежит число, чьи цифры
    совпадают/входят в число из правых граф (16-18) — это артефакт кропа, чистим."""
    def digits(s):
        return "".join(c for c in (s or "") if c.isdigit())
    for row in grid:
        rd = [digits(row[c]) for c in right_cols if c < len(row)]
        for c in left_cols:
            if c >= len(row):
                continue
            d = digits(row[c])
            if len(d) >= 2 and any(x and (d.find(x) >= 0 or x.find(d) >= 0) for x in rd):
                row[c] = ""


def _norm_code(s: str) -> str:
    """Транслитерация Yandex для кодов графы 11: nc->пс, H3->нз, ca->сл, 6/o->б/о."""
    t = s.strip().lower().replace(" ", "").replace("_", "").strip(".-—–")
    return _CODE_MAP.get(t, _CODE_MAP.get(t.rstrip("."), s))


def _is_ditto_loose(s: str) -> bool:
    """Мусорный повтор Yandex (—''— читается как «-u_», «— и —», «-k-»):
    есть тире и <=2 буквенно-цифровых символа."""
    t = (s or "").strip()
    if not t:
        return False
    has_dash = any(c in t for c in "-—–_")
    alnum = sum(1 for c in t if c.isalnum())
    return has_dash and alnum <= 2


# ----------------------- основной вход ------------------------------------
def _col_of(cx: float, colx: list[int]) -> int | None:
    for i in range(len(colx) - 1):
        if colx[i] <= cx < colx[i + 1]:
            return i
    return None


# печатные слова шапки бланка — по ним отбрасываем строки-заголовки
_HEADER_KW = ("описан", "разрез", "глубин", "проход", "категор", "объ", "породы",
              "отметк", "литолог", "обсадк", "скважин", "диаметр", "теорет", "опроб",
              "процент", "содержан", "каменист", "ледянист", "манометр", "визуальн",
              "лаборатор", "промывк", "мерзлот", "водонос", "показател", "измерен")
# подписи-подвал под таблицей — отбрасываем хвостовые строки
_FOOTER_KW = ("буров", "мастер", "техник", "геолог", "начальник", "подпис", "роспис")

# печатный/подписной мусор, который НЕ должен попадать в данные (любой макет)
_JUNK_KW = _HEADER_KW + _FOOTER_KW + (
    "лимитн", "знаютн", "пробност", "олов", "проц", "поняет", "произвел", "фамил",
    "отряд", "участ", "сообщ", "лаборат", "результ", "должност", "дата", "новосиб",
    "весной", "вмечает", "молодост", "верональн", "сотермичн", "химическ", "ангел",
    "выборигор", "маркшейд", "главн", "старш")


def _is_junk_word(text: str) -> bool:
    t = (text or "").strip().lower().replace("ё", "е")
    if not t:
        return True
    letters = "".join(c for c in t if c.isalpha())
    if len(letters) >= 4 and any(kw in t for kw in _JUNK_KW):  # печатное слово шапки
        return True
    return False


def _clean_cell(text: str) -> str:
    """Убрать переносы строк внутри ячейки и лишние пробелы."""
    return " ".join((text or "").replace("\n", " ").split())


def _looks_key(text: str, col: int | None) -> bool:
    """Похоже ли слово на ключ строки: цифра № (col0) или число-глубина (col1)."""
    t = text.strip()
    if col == 0:
        return t[:2].strip("().").isdigit() if t else False
    if col == 1:
        return any(ch.isdigit() for ch in t) and len(t) <= 6
    return False


def _cluster_rows(words: list[dict], colx: list[int], y_tol: int):
    """Сгруппировать слова в строки по Y (независимо от рулей). Отбросить
    печатную шапку. Вернуть список строк: [{cy, cols:{ci:[(x,text,conf)]}}]."""
    items = []
    for w in words:
        bx0, by0, bx1, by1 = w["bbox"]
        items.append(((by0 + by1) / 2, (bx0 + bx1) / 2, bx0, w["text"], w.get("conf")))
    items.sort(key=lambda z: z[0])
    # Центры строк — по РАЗРЫВАМ (gap), без «плывущего» среднего: плотные строки
    # (напр. 11 проходок на правом листе) не схлопываются в меньшее число.
    centers = _cluster_centers([it[0] for it in items], y_tol)
    if not centers:
        return []
    rows = [{"cy": c, "cols": {}} for c in centers]
    for cy, cx, bx0, text, cf in items:
        ri = min(range(len(centers)), key=lambda k: abs(cy - centers[k]))
        ci = _col_of(cx, colx)
        if ci is not None:
            rows[ri]["cols"].setdefault(ci, []).append((bx0, text, cf))

    def has_key(row):
        return any(_looks_key(t, ci) for ci, wl in row["cols"].items() for _, t, _ in wl)

    def joined(row):
        return " ".join(t for wl in row["cols"].values() for _, t, _ in wl).lower()

    def is_header(row):
        return not has_key(row) and any(kw in joined(row) for kw in _HEADER_KW)

    def is_footer(row):
        return any(kw in joined(row) for kw in _FOOTER_KW)

    first = next((i for i, r in enumerate(rows) if has_key(r)), 0)
    out = []
    for r in rows[first:]:
        if is_footer(r):
            break                      # подвал -> дальше данных нет
        if not is_header(r):
            out.append(r)
    return out


def _cluster_centers(ys: list[float], tol: float) -> list[float]:
    if not ys:
        return []
    ys = sorted(ys)
    groups, cur = [], [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] <= tol:
            cur.append(y)
        else:
            groups.append(sum(cur) / len(cur)); cur = [y]
    groups.append(sum(cur) / len(cur))
    return groups


def _fixed_rows(words: list[dict], colx: list[int], height: int,
                header_rows: int, n_labels: int):
    """Таблицы с ФИКС-строками. Y-центры строк берём по ЯКОРЯМ — словам в колонке 0
    (печатные метки «Глубина/Мощность/…» стоят на Y своей строки), чтобы значение
    точно ложилось в свою строку. Если якорей не ровно n_labels — пропорционально."""
    col0 = [(w["bbox"][1] + w["bbox"][3]) / 2 for w in words
            if _col_of((w["bbox"][0] + w["bbox"][2]) / 2, colx) == 0]
    band_h = (height * (1 - header_rows / (header_rows + n_labels))) / max(n_labels, 1)
    centers = _cluster_centers(col0, 0.5 * band_h)
    if len(centers) != n_labels:                 # якоря ненадёжны -> пропорционально
        data_y0 = height * header_rows / (header_rows + n_labels)
        centers = [data_y0 + (i + 0.5) * band_h for i in range(n_labels)]
    elif len(centers) >= 2:
        band_h = (centers[-1] - centers[0]) / (n_labels - 1)
    rows = [{"cy": centers[i], "cols": {}} for i in range(n_labels)]
    for w in words:
        if _is_junk_word(w["text"]):     # печатный/подписной текст в данные не пускаем
            continue
        bx0, by0, bx1, by1 = w["bbox"]
        cy = (by0 + by1) / 2
        ci = _col_of((bx0 + bx1) / 2, colx)
        # строка = БЛИЖАЙШИЙ якорь; принимаем только близкие -> нет затекания
        ri = min(range(n_labels), key=lambda k: abs(cy - centers[k]))
        if ci is not None and abs(cy - centers[ri]) <= 0.5 * band_h:
            rows[ri]["cols"].setdefault(ci, []).append((bx0, w["text"], w.get("conf")))
    return rows


def recognize_object(enh_full: np.ndarray, bbox, spec: Spec, *, engine: str,
                     qwen: QwenVL, yandex: YandexOCR, cfg) -> dict[str, Any]:
    from collections import defaultdict

    x0, y0, x1, y1 = bbox
    obj = enh_full[y0:y1, x0:x1]
    gray = to_gray(obj)
    Hs, Ws = gray.shape
    ncols = spec.ncols
    label_mode = bool(spec.row_labels)
    colx = _col_bounds(Ws, spec)
    hcfg = cfg.get("hybrid", {})
    y_tol = hcfg.get("row_cluster_tol_px", 24)
    _, n_h, _ = extract_grid(gray)   # горизонтали — для сверки числа строк (b)

    # Qwen — раскладка (advisory: сверка + фолбэк), по индексу
    qgrid = qwen.layout_table(obj, spec)["grid"] if engine in ("hybrid", "qwen") else []

    # engine=qwen — «сырая» индексная раскладка (baseline, показывает сдвиг)
    if engine == "qwen":
        out = _index_grid(spec, qgrid, x0, y0)
        out["rows_geom"] = n_h
        return out

    # --- РАСКЛАДКА ПО ГЕОМЕТРИИ ---
    # Yandex-слова -> строки кластеризацией по Y (без опоры на рули: рукопись рвёт
    # линии). Колонка = X-полоса спеки по центру слова. Печатная шапка отброшена.
    words = yandex.words_of(obj)
    if label_mode:                      # фикс-строки: ровно столько строк, сколько меток
        rows_data = _fixed_rows(words, colx, Hs, spec.header_rows, len(spec.row_labels))
    else:
        rows_data = _cluster_rows(words, colx, y_tol)

    n_rows = len(rows_data)
    grid = [["" for _ in range(ncols)] for _ in range(n_rows)]
    conf = [[None for _ in range(ncols)] for _ in range(n_rows)]
    cells: list[dict[str, Any]] = []
    c_start = 1 if label_mode else 0
    cell_ink = hcfg.get("cell_ink_ratio", 0.02)
    pad = hcfg.get("cell_pad_px", 4)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cys = [rw["cy"] for rw in rows_data]
    pitch = float(np.median(np.diff(sorted(cys)))) if len(cys) > 1 else 0.05 * Hs

    # labelled: движок заполняет только fill_cols; col0=метка, «единица» из спеки
    if label_mode:
        cols_to_fill = list(spec.fill_cols) if spec.fill_cols else list(range(1, ncols))
    else:
        cols_to_fill = list(range(0, ncols))

    for r, row in enumerate(rows_data):
        cy = row["cy"]
        y0c, y1c = int(cy - 0.46 * pitch), int(cy + 0.46 * pitch)
        if label_mode:
            grid[r][0] = spec.row_labels[r] if r < len(spec.row_labels) else ""
            if spec.units:
                grid[r][1] = spec.units[r] if r < len(spec.units) else ""
        for ci in cols_to_fill:
            box = (colx[ci], y0c, colx[ci + 1], y1c)
            wl = row["cols"].get(ci, [])
            word_text = " ".join(t for _, t, _ in sorted(wl, key=lambda z: z[0])).strip()
            has_ink = _dark_ratio(gray, thr, box, pad) >= cell_ink
            if not has_ink and not word_text:
                continue  # геометрически пусто (чернил нет) -> НЕ заполняем
            qtext = qgrid[r][ci] if (r < len(qgrid) and ci < len(qgrid[r])) else ""
            q_ditto = qtext.strip() == "-ll-" or assemble.is_ditto(qtext)

            # ТИП содержания решает Qwen (его сила — раскладка/повторы). Если Qwen
            # говорит «повтор» -> «-ll-» (не тратим Yandex, не берём его мусор «и»).
            if engine == "hybrid" and q_ditto:
                final, source, cf, review, ytext = "-ll-", "qwen", None, False, ""
            else:
                if spec.word_placement:
                    # значение = отдельное слово по X-центру (число/код), без кропа
                    ytext, ycf = word_text, None
                else:
                    # покадровый OCR кропа (лучше на мелких №/глубинах), откат на слово
                    crop = obj[max(y0c - pad, 0):min(y1c + pad, Hs),
                               max(colx[ci] - pad, 0):min(colx[ci + 1] + pad, Ws)]
                    ytext, ycf = yandex.read_cell(crop) if crop.size else ("", None)
                    if not ytext:
                        ytext = word_text
                # Выбор источника ТЕКСТА. На чётком скане VLM (Qwen) читает точнее,
                # НО массив Qwen неверен для разрежённого правого листа (sampling) —
                # там раскладка Qwen ломается, берём геометрию Yandex.
                prefer_qwen = (engine == "hybrid" and spec.kind != "sampling_right"
                               and qtext.strip() != "")
                if prefer_qwen:
                    base = qtext
                    if ytext and _agree(qtext, ytext):
                        source, cf, review = "hybrid", HIGH_CONF, False
                    elif ytext:
                        source, cf, review = "qwen", DISAGREE_CONF, True
                    else:
                        source, cf, review = "qwen", YANDEX_EMPTY_CONF, False
                elif ytext:
                    base, source, cf, review = ytext, "yandex", ycf, False
                    if engine == "hybrid" and qtext.strip():
                        agree = _agree(qtext, ytext)
                        source, review = ("hybrid", False) if agree else ("yandex", True)
                elif engine == "hybrid" and qtext.strip():
                    base, source, cf, review = qtext, "qwen", YANDEX_EMPTY_CONF, True
                else:
                    continue
                base = _clean_cell(base)                 # убрать \n и лишние пробелы
                if not base:
                    continue
                if assemble.is_ditto(base):
                    final = "-ll-"
                elif spec.kind == "sampling_right" and ci == 0 and _is_ditto_loose(base):
                    final = "-ll-"                       # графа 9 (объём) — повтор
                else:
                    final = base
                    if spec.kind == "sampling_right" and ci == 2:  # графа 11 — коды
                        final = _norm_code(final)
                    elif spec.kind == "podschet":  # значение подсчёта = число
                        m = re.search(r"\d+[.,]?\d*", final)
                        if not m:
                            continue
                        final = m.group(0)
                ytext = ytext or base
            grid[r][ci] = final
            conf[r][ci] = cf
            cells.append({
                "r": r, "c": ci, "rs": 1, "cs": 1, "text": final, "conf": cf,
                "bbox": [colx[ci] + x0, y0c + y0, colx[ci + 1] + x0, y1c + y0],
                "qwen_text": qtext, "yandex_text": ytext, "source": source,
                "needs_review": review,
            })

    # --- ТОЧЕЧНЫЙ ДОБОР (step 4): повторный OCR бледных инк-но-пустых ячеек гр9-18 ---
    if spec.kind == "sampling_right" and engine != "qwen" and hcfg.get("upscale_retry", True):
        tgt = hcfg.get("upscale_retry_target", 220)
        for r, row in enumerate(rows_data):
            cy = row["cy"]
            y0c, y1c = int(cy - 0.46 * pitch), int(cy + 0.46 * pitch)
            for c in (2, 7, 8, 9):        # коды графы 11 + числа граф 16/17/18
                if grid[r][c].strip():
                    continue
                box = (colx[c], y0c, colx[c + 1], y1c)
                if _dark_ratio(gray, thr, box, pad) < cell_ink:
                    continue              # чернил нет -> действительно пусто
                crop = obj[max(y0c - pad, 0):min(y1c + pad, Hs),
                           max(colx[c] - pad, 0):min(colx[c + 1] + pad, Ws)]
                t2, cf2 = yandex.read_cell(crop, target=tgt) if crop.size else ("", None)
                t2 = _clean_cell(t2)
                if not t2:
                    continue
                t2 = "-ll-" if assemble.is_ditto(t2) else (_norm_code(t2) if c == 2 else t2)
                grid[r][c] = t2
                conf[r][c] = cf2
                cells.append({"r": r, "c": c, "rs": 1, "cs": 1, "text": t2, "conf": cf2,
                              "bbox": [colx[c] + x0, y0c + y0, colx[c + 1] + x0, y1c + y0],
                              "qwen_text": "", "yandex_text": t2,
                              "source": "yandex_retry", "needs_review": cf2 is None})

    if spec.kind == "sampling_right":     # чистим дубли чисел из граф 13-15
        _dedup_numeric(grid, [4, 5, 6], [7, 8, 9])
    # маску «протянутого -ll-» считаем на КОПИИ: сам grid держим сырым («-ll-»),
    # чтобы совпадало с разметкой GT (ditto -> <ditto>) и было честно в JSON.
    ditto_cols = list(range(1, ncols)) if label_mode else None
    expanded = assemble.expand_ditto([rw[:] for rw in grid], ditto_cols)
    return {
        "kind": spec.kind, "rows": n_rows, "cols": ncols,
        "grid": grid, "conf": conf, "expanded": expanded, "cells": cells,
        "rows_qwen": len(qgrid), "rows_geom": n_h,
        "rows_mismatch": bool(n_rows and n_h) and n_rows > n_h,
        "engine": engine,
    }


def _index_grid(spec: Spec, qgrid: list, x0: int, y0: int) -> dict[str, Any]:
    """engine=qwen: раскладка по индексу массива Qwen (baseline). Для labelled —
    фикс-строки/единицы из спеки, значения только в fill_cols."""
    ncols = spec.ncols
    if spec.row_labels:
        n = len(spec.row_labels)
        fill = spec.fill_cols or list(range(1, ncols))
        grid = [["" for _ in range(ncols)] for _ in range(n)]
        for r in range(n):
            grid[r][0] = spec.row_labels[r]
            if spec.units:
                grid[r][1] = spec.units[r] if r < len(spec.units) else ""
            for c in fill:
                v = qgrid[r][c] if r < len(qgrid) and c < len(qgrid[r]) else ""
                grid[r][c] = "-ll-" if assemble.is_ditto(v) else v
        expanded = assemble.expand_ditto([rw[:] for rw in grid], list(range(1, ncols)))
        return {"kind": spec.kind, "rows": n, "cols": ncols, "grid": grid,
                "conf": [[None] * ncols for _ in range(n)], "expanded": expanded,
                "cells": [], "rows_qwen": len(qgrid), "rows_geom": n,
                "rows_mismatch": False, "engine": "qwen"}
    grid = [["" for _ in range(ncols)] for _ in range(len(qgrid))]
    cells = []
    for r, row in enumerate(qgrid):
        for c in range(ncols):
            v = row[c] if c < len(row) else ""
            v = "-ll-" if assemble.is_ditto(v) else v
            grid[r][c] = v
            if v.strip():
                cells.append({"r": r, "c": c, "rs": 1, "cs": 1, "text": v,
                              "conf": None, "bbox": None, "qwen_text": v,
                              "yandex_text": None, "source": "qwen", "needs_review": False})
    expanded = assemble.expand_ditto([rw[:] for rw in grid], None)
    return {"kind": spec.kind, "rows": len(qgrid), "cols": ncols, "grid": grid,
            "conf": [[None] * ncols for _ in qgrid], "expanded": expanded, "cells": cells,
            "rows_qwen": len(qgrid), "rows_geom": len(qgrid), "rows_mismatch": False,
            "engine": "qwen"}
