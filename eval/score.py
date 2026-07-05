"""Метрики Milestone против eval/groundtruth/<скв>.json (разметка заказчика, 6 таблиц).

(a) точность ТЕКСТА (strict/fuzzy) по непустым НЕ-graphic ячейкам — на объект и суммарно;
(b) row_count vs эталон;
(c) merges_match: слияния листа (.xlsx) vs GT;
(d) graph_numbers: ряд номеров граф (spec) vs GT;
(e) hybrid — доля расхождений Qwen↔Yandex.

fuzzy: trim, lower, collapse spaces, ditto->,<ditto>, числа ','<->'.' и без пробелов, ё->е.

Запуск:
  python -m eval.score out_yandex.json out_qwen.json out_hybrid.json --gt eval/groundtruth/скв2.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GT_KEY_TO_KIND = {"гр1-8": "journal_left", "гр9-18": "sampling_right",
                  "самородки": "samorodki", "подсчёт": "podschet",
                  "показатели": "pokazateli", "контроль": "kontrol"}

_DITTO_RE = re.compile(r'^[\s\-‐-―_lі/\\«»"\'’“”·.]{1,6}$')


def _is_ditto(t: str) -> bool:
    t = (t or "").strip()
    return bool(t) and (t == "-ll-" or bool(_DITTO_RE.match(t)) or t in (">>", "—''—", "—«—"))


def _fuzzy(s) -> str:
    t = ("" if s is None else str(s)).strip().lower().replace("ё", "е")
    if _is_ditto(t):
        return "<ditto>"
    t = re.sub(r"\s+", " ", t).replace(",", ".")
    t = re.sub(r"(?<=\d)\s+(?=\d)", "", t)
    return t.strip()


def _strict(s) -> str:
    return ("" if s is None else str(s)).strip()


def _rows_of(obj: dict) -> list:
    return obj.get("data", obj.get("rows", []))


# фикс-колонки (метки/единицы) в labelled-таблицах — это by-construction
_FIXED_COLS = {"podschet": {0, 1}, "pokazateli": {0}, "kontrol": {0}, "samorodki": {0}}


def _prefix(gt: dict) -> str:
    return (gt.get("borehole", "") + "_") if gt.get("borehole") else ""


def text_accuracy(result: dict, gt: dict) -> dict:
    pre = _prefix(gt)
    by_kind = {s["kind"]: s for s in result["sheets"]
               if s.get("table") and s["name"].startswith(pre)}
    per_kind, tot, s_ok, f_ok = {}, 0, 0, 0
    # bucket: live (реальные данные) vs const (ditto/фикс-метки/единицы)
    live = {"total": 0, "strict": 0, "fuzzy": 0}
    const = {"total": 0, "strict": 0, "fuzzy": 0}
    for gt_key, obj in gt.get("objects", {}).items():
        kind = GT_KEY_TO_KIND.get(gt_key, gt_key)
        grid = (by_kind[kind]["table"]["grid"] if kind in by_kind and by_kind[kind].get("table") else [])
        fixed = _FIXED_COLS.get(kind, set())
        k_tot = k_s = k_f = 0
        kl = {"total": 0, "fuzzy": 0}
        for r, row in enumerate(_rows_of(obj)):
            for c, cell in enumerate(row):
                if cell.get("empty") or cell.get("graphic"):
                    continue
                gt_text = cell.get("text", "")
                if not gt_text.strip():
                    continue
                pred = grid[r][c] if (r < len(grid) and c < len(grid[r])) else ""
                s = _strict(pred) == _strict(gt_text)
                f = _fuzzy(pred) == _fuzzy(gt_text)
                k_tot += 1; k_s += s; k_f += f
                is_const = cell.get("ditto") or c in fixed
                b = const if is_const else live
                b["total"] += 1; b["strict"] += s; b["fuzzy"] += f
                if not is_const:
                    kl["total"] += 1; kl["fuzzy"] += f
        per_kind[kind] = {"total": k_tot, "strict": k_s, "fuzzy": k_f, "live": kl}
        tot += k_tot; s_ok += k_s; f_ok += k_f
    return {"total": tot, "strict": s_ok, "fuzzy": f_ok,
            "strict_acc": s_ok / tot if tot else None,
            "fuzzy_acc": f_ok / tot if tot else None, "per_kind": per_kind,
            "live": live, "const": const}


def merges_match(result_path: Path, result: dict, gt: dict) -> dict:
    """Слияния из .xlsx листа vs GT (по каждому объекту)."""
    import openpyxl
    xlsx = result_path.with_suffix(".xlsx")
    out = {}
    if not xlsx.exists():
        return out
    wb = openpyxl.load_workbook(xlsx)
    pre = _prefix(gt)
    kind_to_sheet = {s["kind"]: s["name"] for s in result["sheets"]
                     if s["name"].startswith(pre)}
    for gt_key, obj in gt.get("objects", {}).items():
        kind = GT_KEY_TO_KIND.get(gt_key, gt_key)
        exp = set(obj.get("merges", []))
        sheet = kind_to_sheet.get(kind)
        got = set()
        if sheet and sheet in wb.sheetnames:
            got = {str(m) for m in wb[sheet].merged_cells.ranges}
        out[kind] = {"ok": got == exp, "exp": sorted(exp), "got": sorted(got)}
    return out


def graph_numbers_match(gt: dict) -> dict:
    from src.pipeline.forms_layout import SPECS
    out = {}
    for gt_key, obj in gt.get("objects", {}).items():
        if "graph_numbers" not in obj:
            continue
        kind = GT_KEY_TO_KIND.get(gt_key, gt_key)
        spec = SPECS.get(kind)
        out[kind] = {"ok": bool(spec) and spec.graph_nums == obj["graph_numbers"],
                     "exp": obj["graph_numbers"], "got": spec.graph_nums if spec else None}
    return out


def row_counts(result: dict, gt: dict | None = None) -> list[dict]:
    pre = _prefix(gt) if gt else ""
    return [{"kind": s["kind"], "rows": len(s["table"].get("grid", []))}
            for s in result["sheets"] if s.get("table") and s["name"].startswith(pre)]


def divergence(result: dict) -> dict:
    both = dis = 0
    for m in result.get("meta", []):
        q, y = m.get("qwen_text"), m.get("yandex_text")
        if q and y and str(q).strip() and str(y).strip():
            both += 1
            dis += (_fuzzy(q) != _fuzzy(y))
    return {"both": both, "disagree": dis, "rate": dis / both if both else None}


def score_one(path: Path, gt: dict) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    eng = result.get("engine")
    print(f"\n================  {path.name}  движок={eng}  ================")
    acc = text_accuracy(result, gt)
    if acc["total"]:
        print(f"(a) ТЕКСТ: strict {acc['strict']}/{acc['total']}={acc['strict_acc']:.1%}   "
              f"fuzzy {acc['fuzzy']}/{acc['total']}={acc['fuzzy_acc']:.1%}")
        lv, cn = acc["live"], acc["const"]
        lf = f"{lv['fuzzy']}/{lv['total']}={lv['fuzzy']/lv['total']:.1%}" if lv["total"] else "-"
        cf = f"{cn['fuzzy']}/{cn['total']}={cn['fuzzy']/cn['total']:.1%}" if cn["total"] else "-"
        print(f"    (i) ЖИВЫЕ данные fuzzy: {lf}    (ii) ditto/фикс fuzzy: {cf}")
        for k, v in acc["per_kind"].items():
            if v["total"]:
                lv = v.get("live", {"total": 0, "fuzzy": 0})
                lstr = (f"  живые {lv['fuzzy']}/{lv['total']}" if lv["total"] else "")
                print(f"      {k:<15} strict {v['strict']}/{v['total']}  "
                      f"fuzzy {v['fuzzy']}/{v['total']}{lstr}")
    gtrc = {GT_KEY_TO_KIND.get(k, k): len(_rows_of(o)) for k, o in gt.get("objects", {}).items()}
    print("(b) строк vs эталон:", ", ".join(
        f"{r['kind']}={r['rows']}/{gtrc.get(r['kind'],'?')}" for r in row_counts(result, gt)))
    mm = merges_match(path, result, gt)
    bad = [k for k, v in mm.items() if not v["ok"]]
    print(f"(c) merges_match: {'ВСЕ OK' if not bad else 'расхождения: ' + ', '.join(bad)}")
    for k in bad:
        print(f"      {k}: ждали {mm[k]['exp']} получили {mm[k]['got']}")
    gn = graph_numbers_match(gt)
    gbad = [k for k, v in gn.items() if not v["ok"]]
    print(f"(d) graph_numbers: {'OK' if not gbad else 'расхождения: ' + ', '.join(gbad)}")
    if eng == "hybrid":
        d = divergence(result)
        rate = f"{d['rate']:.1%}" if d["rate"] is not None else "-"
        print(f"(e) расхождения Qwen-Yandex: {d['disagree']}/{d['both']} = {rate}")


def main():
    ap = argparse.ArgumentParser(description="Метрики Milestone vs GT")
    ap.add_argument("result_json", nargs="+")
    ap.add_argument("--gt", required=True)
    a = ap.parse_args()
    gt = json.loads(Path(a.gt).read_text(encoding="utf-8"))
    print(f"GT: {Path(a.gt).name}")
    for rp in a.result_json:
        score_one(Path(rp), gt)


if __name__ == "__main__":
    main()
