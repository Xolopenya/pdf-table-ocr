"""Оркестратор: PDF -> объекты -> движок -> .xlsx (+ .json, _meta).

Milestone: первая скважина = передний разворот (стр.1) + разворот главной
таблицы (стр.2). Каждый объект §2 -> отдельный лист. Движок (hybrid/qwen/yandex)
из конфига/CLI. Числа сверок и needs_review — в сводку.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..config import Config
from ..logging_setup import logger
from ..models import FORM_TYPES, ObjectType, TABLE_TYPES
from . import forms_kv
from . import hybrid as hy
from .export import export_document
from .forms_layout import SPECS
from .inventory import inventory_page
from .ocr_qwen import QwenVL
from .ocr_yandex import YandexOCR


def _sheet_name(kind: str, N: str, page: int) -> str:
    # ЕДИНЫЙ ключ на скважину для ВСЕХ объектов: скв{N}_<объект>
    return {
        "journal_left": f"скв{N}_гр1-8",
        "sampling_right": f"скв{N}_гр9-18",
        "podschet": f"скв{N}_подсчёт",
        "pokazateli": f"скв{N}_показатели",
        "kontrol": f"скв{N}_контроль",
        "samorodki": f"скв{N}_самородки",
        "form_journal": f"скв{N}_титул",
        "form_act": f"скв{N}_акт",
    }.get(kind, f"скв{N}_{kind}")


def _table_meta(sheet: str, kind: str, table: dict[str, Any]) -> list[dict]:
    rows = []
    for c in table.get("cells", []):
        rows.append({
            "sheet": sheet, "obj_type": kind, "r": c["r"], "c": c["c"],
            "rs": c.get("rs", 1), "cs": c.get("cs", 1), "bbox": c.get("bbox"),
            "text_source": c.get("source"), "qwen_text": c.get("qwen_text"),
            "yandex_text": c.get("yandex_text"), "conf": c.get("conf"),
            "needs_review": c.get("needs_review", False), "text": c.get("text"),
        })
    return rows


def _form_meta(sheet: str, kind: str, form: dict[str, Any]) -> list[dict]:
    rows = []
    for i, f in enumerate(form["fields"]):
        rows.append({
            "sheet": sheet, "obj_type": kind, "r": i, "c": 0,
            "text_source": f.get("source"), "text": f.get("value"),
            "needs_review": f.get("needs_review", False), "qwen_text": f.get("value"),
        })
    return rows


def _guess_borehole(forms: list[dict]) -> str:
    for form in forms:
        for f in form["fields"]:
            if "скважина" in f["key"].lower():
                m = re.search(r"\d+", f.get("value", "") or "")
                if m:
                    return m.group(0)
    return "1"


def _collect_objects(halves):
    table_objs, form_objs = [], []
    for hp in halves:
        for obj in hp["objects"]:
            entry = {"kind": obj.obj_type.value, "bbox": obj.bbox.as_list(),
                     "enh": hp["enh"], "page": obj.pdf_page, "side": obj.side}
            (table_objs if obj.obj_type in TABLE_TYPES else
             form_objs if obj.obj_type in FORM_TYPES else []).append(entry)
    return table_objs, form_objs


def _process_borehole(halves, cfg, engine, qwen, yandex, N: str):
    """Обработать один разворот-скважину. N — АВТОРИТЕТНЫЙ номер (по позиции/канону);
    OCR-номер используется только как ПОДСКАЗКА и пишется в _meta/титул с
    needs_review при расхождении. Возвращает (sheets, meta, summary, ocr_hint)."""
    table_objs, form_objs = _collect_objects(halves)
    bh_forms = [forms_kv.recognize_form(e["enh"], e["bbox"], e["kind"],
                                        engine="qwen", qwen=qwen, yandex=yandex, cfg=cfg)
                for e in form_objs]
    ocr_hint = _guess_borehole(bh_forms)          # прочитанный OCR-номер (подсказка)
    mism = ocr_hint not in ("", None) and ocr_hint != str(N)

    sheets, meta, summary = [], [], []
    for e in table_objs:
        spec = SPECS[e["kind"]]
        tab = hy.recognize_object(e["enh"], e["bbox"], spec, engine=engine,
                                  qwen=qwen, yandex=yandex, cfg=cfg)
        name = _sheet_name(e["kind"], N, e["page"])
        sheets.append({"name": name, "kind": e["kind"], "table": tab})
        meta.extend(_table_meta(name, e["kind"], tab))
        summary.append({"sheet": name, "kind": e["kind"],
                        "rows": len(tab.get("grid", [])), "divergence": _divergence(tab)})
    if cfg.get("extract_forms", True):
        for e in form_objs:
            form = forms_kv.recognize_form(e["enh"], e["bbox"], e["kind"],
                                           engine=engine, qwen=qwen, yandex=yandex, cfg=cfg)
            if e["kind"] == "form_journal":       # OCR-номер как поле-подсказка
                form["fields"].append({
                    "key": "OCR: Скважина № (подсказка)", "value": ocr_hint or "",
                    "source": "qwen", "conf": None, "needs_review": mism})
            name = _sheet_name(e["kind"], N, e["page"])
            sheets.append({"name": name, "kind": e["kind"], "form": form})
            meta.extend(_form_meta(name, e["kind"], form))
    meta.append({"sheet": f"скв{N}", "obj_type": "borehole_id", "r": 0, "c": 0,
                 "text": f"pos_N={N}", "qwen_text": ocr_hint, "needs_review": mism,
                 "text_source": "position/canonical"})
    return sheets, meta, summary, ocr_hint


def process_pdf(pdf_path, cfg: Config, *, engine: str = "hybrid", dry_run: bool = False,
                pages: list[int] | None = None, out_dir=None, save_debug: bool = False,
                progress=None) -> dict:
    """ЕДИНАЯ точка входа: ОДИН PDF -> один .xlsx (все скважины/объекты) + один .json.

    Внутри: рендер, split, deskew, enhance, детект объектов, раскладка (спека×строки),
    OCR (hybrid, кэш по хэшу), апскейл бледных ячеек, нормализация кодов/ditto, дедуп,
    сборка. GUI и CLI вызывают ровно её; массовых прогонов ядро само не запускает.

    progress(done, total, msg) — колбэк для видимого прогресса; msg включает счётчики
    вызовов Yandex/Qwen и попаданий в кэш. Результат по каждой завершённой скважине
    досохраняется (устойчивость к прерыванию).
    """
    pdf_path = Path(pdf_path)
    stem = pdf_path.stem
    qwen = QwenVL(cfg, dry_run=dry_run)
    yandex = YandexOCR(cfg, dry_run=dry_run)
    out_root = Path(out_dir) if out_dir else cfg.path_of("paths.output_dir")
    out_base = out_root / f"{stem}_{engine}"

    from pdf2image import pdfinfo_from_path   # poppler pdfinfo — без рендера
    n_pages = int(pdfinfo_from_path(str(pdf_path))["Pages"])
    starts = pages if pages is not None else list(range(0, n_pages, 2))
    n_bore = len(starts)

    # --- ИДЕНТИФИКАЦИЯ: номера скважин по ПОЗИЦИИ, не по OCR ---
    canonical = (cfg.get("borehole_numbers", {}) or {}).get(stem)
    if canonical is not None and pages is None:
        if len(canonical) != n_bore:
            raise ValueError(
                f"Идентификация: разворотов в файле {n_bore}, а в borehole_numbers "
                f"['{stem}'] — {len(canonical)} номеров {canonical}. "
                + (f"Лишних разворотов: {n_bore - len(canonical)}."
                   if n_bore > len(canonical)
                   else f"Не хватает разворотов: {len(canonical) - n_bore}.")
                + " Исправьте список или файл.")
        n_list = [str(x) for x in canonical]
    elif canonical is not None:                 # подмножество страниц (отладка)
        n_list = [str(canonical[f // 2]) if (f // 2) < len(canonical) else str(f // 2 + 1)
                  for f in starts]
    else:
        n_list = [str(bi) for bi in range(1, n_bore + 1)]   # ключ = позиция
    all_sheets, all_meta, report_boreholes = [], [], []

    def _stat_str():
        y, q = yandex.stats, qwen.stats
        return (f"Yandex {y['call']} (кэш {y['cache']}, сбой {y['fail']})  "
                f"Qwen {q['call']} (кэш {q['cache']})")

    def _emit(done, msg):
        line = f"{msg} · {_stat_str()}"
        logger.info(line)
        if progress:
            progress(done, n_bore + 1, line)

    for bi, front in enumerate(starts, 1):
        N = n_list[bi - 1]                       # авторитетный номер по позиции/канону
        _emit(bi - 1, f"Скважина {N} (позиция {bi}/{n_bore}, стр.{front+1}-{front+2})…")
        halves = inventory_page(pdf_path, front, cfg)
        if front + 1 < n_pages:
            halves += inventory_page(pdf_path, front + 1, cfg)
        sheets, meta, summary, ocr_hint = _process_borehole(halves, cfg, engine, qwen, yandex, N)
        if ocr_hint and ocr_hint != N:
            logger.warning(f"  позиция {bi}: OCR прочитал скв «{ocr_hint}», "
                           f"канон/позиция = «{N}» -> needs_review")
        all_sheets.extend(sheets)
        all_meta.extend(meta)
        report_boreholes.append({"position": bi, "N": N, "ocr_hint": ocr_hint,
                                 "mismatch": bool(ocr_hint and ocr_hint != N),
                                 "pages": [front + 1, front + 2], "objects": len(sheets)})
        if save_debug:
            _dump_debug(halves, cfg, stem)
        # УСТОЙЧИВОСТЬ: досохраняем частичный результат после каждой скважины
        export_document(all_sheets, all_meta, out_base, source=str(pdf_path),
                        engine=engine, write_meta=cfg.get("export.write_meta", True))
        _emit(bi, f"Готова скважина {N} ({len(sheets)} объектов)")

    paths = export_document(all_sheets, all_meta, out_base, source=str(pdf_path),
                            engine=engine, write_meta=cfg.get("export.write_meta", True))
    report = {"pdf": str(pdf_path), "engine": engine, "boreholes": n_bore,
              "sheets": len(all_sheets), "yandex": yandex.stats, "qwen": qwen.stats,
              "detail": report_boreholes}
    logger.info(f"[ГОТОВО] скважин={n_bore} листов={len(all_sheets)} · {_stat_str()} -> {paths['xlsx']}")
    if progress:
        progress(n_bore + 1, n_bore + 1, f"Готово: {n_bore} скважин, {len(all_sheets)} листов")
    return {"xlsx": paths["xlsx"], "json": paths["json"], "report": report}


def _dump_debug(halves, cfg, stem) -> None:
    """Сохранить enhanced/overlay в data/interim (по галочке «debug» в GUI)."""
    import cv2
    from .inventory import _COLORS, _imwrite
    interim = cfg.path_of("paths.interim_dir") / stem
    for hp in halves:
        tag = f"p{hp['objects'][0].pdf_page:02d}_{hp['side']}" if hp["objects"] else hp["side"]
        _imwrite(interim / "halves" / f"{tag}.png", hp["enh"])
        ov = cv2.cvtColor(hp["enh"], cv2.COLOR_GRAY2BGR) if hp["enh"].ndim == 2 else hp["enh"].copy()
        for o in hp["objects"]:
            b = o.bbox
            cv2.rectangle(ov, (b.x0, b.y0), (b.x1, b.y1),
                          _COLORS.get(o.obj_type.value, (60, 60, 60)), 4)
        _imwrite(interim / "overlays" / f"{tag}_overlay.png", ov)


def process_first_borehole(pdf_path, cfg: Config, *, engine: str = "hybrid",
                           dry_run: bool = False) -> dict:
    """Тонкая обёртка: только первая скважина (стр. 1-2). Для отладки/сравнения."""
    return process_pdf(pdf_path, cfg, engine=engine, dry_run=dry_run, pages=[0])


def _divergence(tab: dict) -> dict:
    """Доля расхождений Qwen↔Yandex среди ячеек, где ОБА непусты."""
    both = dis = 0
    for c in tab.get("cells", []):
        q, y = c.get("qwen_text"), c.get("yandex_text")
        if q and y and q.strip() and y.strip():
            both += 1
            if not hy._agree(q, y):
                dis += 1
    return {"both_nonempty": both, "disagree": dis,
            "rate": (dis / both) if both else None}


def _print_summary(engine, N, summary, forms_out, dry_run) -> None:
    logger.info("=" * 66)
    logger.info(f"[СВОДКА] скважина {N}, движок={engine}"
                f"{'  (dry-run)' if dry_run else ''}")
    for s in summary:
        rc = f"{s['rows_qwen']}q/{s['rows_geom']}h"
        mm = " ⚠строки" if s["rows_mismatch"] else ""
        d = s["divergence"]
        drate = f"{d['disagree']}/{d['both_nonempty']}" if d["both_nonempty"] else "—"
        logger.info(f"  {s['sheet']:<20} {s['kind']:<15} строк {rc}{mm} "
                    f"ячеек={s['cells']} расхожд(Qwen↔Yandex)={drate}")
    for e, form in forms_out:
        filled = sum(1 for f in form["fields"] if f["value"].strip())
        logger.info(f"  {form['title'][:32]:<34} полей заполнено={filled}/{len(form['fields'])}")
    logger.info("=" * 66)
