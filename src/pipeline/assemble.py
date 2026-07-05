

from __future__ import annotations

import re
from typing import Any

_DITTO_RE = re.compile(r'^[\s\-\u2010-\u2015_lі/\\«»"\'’“”·.]{1,6}$')


def is_ditto(text: str | None) -> bool:
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    return bool(_DITTO_RE.match(t))


def order_lines(lines: list[dict[str, Any]], y_tol: int = 14) -> list[dict[str, Any]]:
    def key(l):
        bb = l.get("bbox")
        if not bb:
            return (1 << 30, 0)
        return (bb[1], bb[0])
    ordered = sorted(lines, key=key)
    result: list[dict[str, Any]] = []
    band: list[dict[str, Any]] = []
    band_y = None
    for l in ordered:
        bb = l.get("bbox")
        y = bb[1] if bb else None
        if band_y is None or y is None or abs((y or 0) - band_y) <= y_tol:
            band.append(l)
            band_y = y if band_y is None else band_y
        else:
            result.extend(sorted(band, key=lambda x: (x["bbox"][0] if x.get("bbox") else 0)))
            band = [l]
            band_y = y
    if band:
        result.extend(sorted(band, key=lambda x: (x["bbox"][0] if x.get("bbox") else 0)))
    return result


def build_table(table: dict[str, Any]) -> dict[str, Any]:

    cells = table.get("cells") or []
    rows = table.get("rows") or 0
    cols = table.get("cols") or 0
    if not rows:
        rows = 1 + max((c["r"] + c.get("rs", 1) - 1 for c in cells), default=-1)
    if not cols:
        cols = 1 + max((c["c"] + c.get("cs", 1) - 1 for c in cells), default=-1)
    rows = max(rows, 0)
    cols = max(cols, 0)

    grid = [["" for _ in range(cols)] for _ in range(rows)]
    conf = [[None for _ in range(cols)] for _ in range(rows)]
    merges: list[tuple[int, int, int, int]] = []

    for cell in cells:
        r, c = cell["r"], cell["c"]
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        grid[r][c] = cell.get("text", "")
        conf[r][c] = cell.get("conf")
        rs, cs = cell.get("rs", 1), cell.get("cs", 1)
        if rs > 1 or cs > 1:
            merges.append((r, c, rs, cs))

    return {"rows": rows, "cols": cols, "grid": grid, "conf": conf, "merges": merges}


def expand_ditto(grid: list[list[str]], columns: list[int] | None = None) -> list[list[bool]]:
    if not grid:
        return []
    n_rows = len(grid)
    n_cols = len(grid[0])
    cols = columns if columns is not None else list(range(n_cols))
    expanded = [[False] * n_cols for _ in range(n_rows)]
    for c in cols:
        last = ""
        for r in range(n_rows):
            val = grid[r][c]
            if is_ditto(val):
                if last:
                    grid[r][c] = last
                    expanded[r][c] = True
            elif val and val.strip():
                last = val
    return expanded
