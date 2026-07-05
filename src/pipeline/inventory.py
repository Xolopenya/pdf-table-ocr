"""Инвентаризация: локатор объектов (тип + область) на каждой логической половине.

Обязательный первый шаг — перечислить ВСЕ объекты, чтобы ничего не потерять.
КОЛОНКИ отсюда НЕ берём (только область объекта). Дискриминатор «главная таблица
vs передний разворот» — число ГОРИЗОНТАЛЕЙ (надёжны): у главной 27–32, у форм
и мини-таблиц заметно меньше.

Раскладка объектов по половинам фиксированного бланка:
  MAIN-разворот:  L = journal_left;  R = sampling_right + pokazateli + kontrol
  FRONT-разворот: L = samorodki + АКТ;  R = титул «ЖУРНАЛ» + podschet
Если регион не нашёлся геометрически — отдаём фолбэк по долям (объект не теряем).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..logging_setup import logger
from ..models import BBox, InventoryReport, ObjectType, PageObject
from .preprocess import deskew_projection, enhance, extract_grid, render_pdf, split_spread


def _imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if ok:
        path.write_bytes(buf.tobytes())


def _grid_regions(enh: np.ndarray, cfg: dict) -> list[BBox]:
    """Прямоугольные регионы-таблицы как связные компоненты маски сетки."""
    grid, _, _ = extract_grid(enh)
    close = cfg.get("region_close_px", 25)
    dil = cfg.get("region_dilate_px", 15)
    m = cv2.morphologyEx(grid, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (close, close)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (dil, dil)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    H, W = enh.shape[:2]
    min_area = cfg.get("min_region_area_frac", 0.015) * H * W
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w * h < min_area or w < 0.12 * W or h < 0.03 * H:
            continue
        boxes.append(BBox(x0=int(x), y0=int(y), x1=int(x + w), y1=int(y + h)))
    boxes.sort(key=lambda b: (b.y0, b.x0))
    return boxes


def _frac(H: int, W: int, fx0: float, fy0: float, fx1: float, fy1: float) -> BBox:
    return BBox(x0=int(W * fx0), y0=int(H * fy0), x1=int(W * fx1), y1=int(H * fy1))


def _count_vlines(enh: np.ndarray, region: BBox) -> int:
    """Число вертикальных рулей в регионе (close→open восстанавливает бледные).

    Только для РАЗЛИЧЕНИЯ главной таблицы и форм (не для колонок). У главной
    таблицы это >=4, у титульного блока с подчёркиваниями — ~0."""
    sub = enh[region.y0:region.y1, region.x0:region.x1]
    if sub.size == 0:
        return 0
    g = sub if sub.ndim == 2 else cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 41, 15)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    vmask = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(0.4 * h)))))
    proj = vmask.sum(axis=0) / 255.0
    idx = np.where(proj >= 0.3 * h)[0]
    if idx.size == 0:
        return 0
    groups = 1
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 12:
            groups += 1
    return groups


def _pick_main(enh: np.ndarray, regions: list[BBox], H: int, W: int) -> BBox | None:
    """Регион главной таблицы: высокий+широкий + реальная вертикальная решётка."""
    cand = [b for b in regions if b.h >= 0.55 * H and b.w >= 0.40 * W]
    cand.sort(key=lambda b: b.area, reverse=True)
    for b in cand:
        if _count_vlines(enh, b) >= 4:
            return b
    return None


def locate_objects_on_half(enh: np.ndarray, side: str, page: int, cfg: Config,
                           n_h: int) -> list[PageObject]:
    """Список объектов (тип + bbox) на одной половине."""
    inv = cfg.get("inventory", {})
    H, W = enh.shape[:2]
    full = BBox(x0=0, y0=0, x1=W, y1=H)
    regions = _grid_regions(enh, inv)
    main = _pick_main(enh, regions, H, W)
    is_main = main is not None
    objs: list[PageObject] = []

    def add(t, b, conf, note=None):
        objs.append(PageObject(obj_type=t, bbox=b, pdf_page=page, side=side,
                               confidence=conf, note=note))

    if is_main and side == "L":
        add(ObjectType.JOURNAL_LEFT, main, 0.9)
    elif is_main and side == "R":
        # Правый лист бланка расчерчен ОТ КРАЯ ДО КРАЯ: детектированный компонент
        # сливает опробование 9-18 с мини-таблицами. Границы граф 9-18 и мини-
        # таблиц берём из РЕГИОН-ШАБЛОНА фикс-бланка (доли), высоту — от детекции.
        y0f, y1f = main.y0 / H, main.y1 / H
        # граф 9-18: от левого руля бланка (~0.06) до правого края граф 18 (~0.49)
        add(ObjectType.SAMPLING_RIGHT, _frac(H, W, 0.058, y0f, 0.492, y1f), 0.85, "границы граф из шаблона")
        add(ObjectType.POKAZATELI, _frac(H, W, 0.49, 0.19, 0.75, 0.31), 0.6, "регион из шаблона")
        add(ObjectType.KONTROL, _frac(H, W, 0.49, 0.42, 0.75, 0.57), 0.6, "регион из шаблона")
    elif (not is_main) and side == "L":
        # самородки — верхняя НЕвысокая таблица; АКТ (форма) — ниже
        tops = [b for b in regions if b.y0 < 0.35 * H and b.h < 0.45 * H]
        top = min(tops, key=lambda b: b.y0) if tops else None
        if top is not None:
            add(ObjectType.SAMORODKI, top, 0.75)
            add(ObjectType.FORM_ACT, BBox(x0=0, y0=top.y1, x1=W, y1=H), 0.55, "форма key-value")
        else:
            add(ObjectType.SAMORODKI, _frac(H, W, 0.04, 0.10, 0.99, 0.30), 0.4, "регион из шаблона")
            add(ObjectType.FORM_ACT, _frac(H, W, 0.0, 0.32, 1.0, 0.95), 0.4, "регион из шаблона")
    elif (not is_main) and side == "R":
        # Титул сверху + подсчёт снизу сливаются в один компонент на бланке ->
        # делим по РЕГИОН-ШАБЛОНУ фикс-бланка (доли).
        add(ObjectType.FORM_JOURNAL, _frac(H, W, 0.0, 0.0, 1.0, 0.56), 0.55, "форма key-value / шаблон")
        add(ObjectType.PODSCHET, _frac(H, W, 0.02, 0.57, 0.99, 0.95), 0.6, "регион из шаблона")
    else:
        add(ObjectType.UNKNOWN, full, 0.1, "половина не определена")
    return objs


def inventory_page(pdf_path, page_idx: int, cfg: Config):
    """Разобрать одну страницу PDF. Возвращает список половин:
    [{side, enh, n_h, n_v, deskew, objects:[PageObject]}]."""
    dpi = cfg.get("render_dpi", 300)
    pages = render_pdf(str(pdf_path), dpi=dpi, first=page_idx + 1, last=page_idx + 1)
    if not pages:
        return []
    halves = split_spread(pages[0])
    out = []
    for side, half in halves:
        des, ang = deskew_projection(half)
        enh = enhance(des)
        _, n_h, n_v = extract_grid(enh)
        objs = locate_objects_on_half(enh, side, page_idx + 1, cfg, n_h)
        out.append({"side": side, "enh": enh, "n_h": n_h, "n_v": n_v,
                    "deskew": ang, "objects": objs})
    return out


_COLORS = {
    "journal_left": (0, 140, 255), "sampling_right": (0, 200, 0),
    "samorodki": (255, 0, 0), "podschet": (200, 0, 200),
    "pokazateli": (0, 200, 200), "kontrol": (128, 128, 0),
    "form_journal": (0, 0, 255), "form_act": (90, 90, 90), "unknown": (60, 60, 60),
}


def inventory_pdf(pdf_path, cfg: Config, dump: bool = True) -> InventoryReport:
    pdf_path = Path(pdf_path)
    stem = pdf_path.stem
    dpi = cfg.get("render_dpi", 300)
    pages = render_pdf(str(pdf_path), dpi=dpi)
    report = InventoryReport(pdf_path=str(pdf_path), pdf_stem=stem, n_pdf_pages=len(pages))
    interim = cfg.path_of("paths.interim_dir") / stem
    logger.info(f"Инвентаризация «{pdf_path.name}»: {len(pages)} стр. PDF @ {dpi} dpi")

    for pi, spread in enumerate(pages):
        for side, half in split_spread(spread):
            des, ang = deskew_projection(half)
            enh = enhance(des)
            _, n_h, n_v = extract_grid(enh)
            objs = locate_objects_on_half(enh, side, pi + 1, cfg, n_h)
            report.objects.extend(objs)
            if dump:
                tag = f"p{pi+1:02d}_{side}"
                _imwrite(interim / "halves" / f"{tag}.png", enh)
                ov = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR) if enh.ndim == 2 else enh.copy()
                for o in objs:
                    b = o.bbox
                    col = _COLORS.get(o.obj_type.value, (60, 60, 60))
                    cv2.rectangle(ov, (b.x0, b.y0), (b.x1, b.y1), col, 4)
                    cv2.putText(ov, f"{o.obj_type.value} {o.confidence:.2f}",
                                (b.x0 + 6, b.y0 + 36), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, col, 3, cv2.LINE_AA)
                _imwrite(interim / "overlays" / f"{tag}_overlay.png", ov)

    if dump:
        interim.mkdir(parents=True, exist_ok=True)
        (interim / "inventory.json").write_text(
            json.dumps(json.loads(report.model_dump_json()), ensure_ascii=False, indent=2),
            encoding="utf-8")
        counts = report.counts()
        lines = [f"PDF: {report.pdf_path}", f"Страниц: {report.n_pdf_pages}",
                 f"Объектов: {len(report.objects)}", "", "По типам:"]
        for k in sorted(counts):
            lines.append(f"  {k:16s} {counts[k]}")
        (interim / "inventory_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        logger.info("Типы объектов: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return report
