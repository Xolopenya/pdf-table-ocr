"""
prep.py — подготовка сканов буровых журналов к распознаванию.

Что делает:
  1) рендерит PDF в изображения с учётом поворота страницы (развороты альбомные);
  2) режет разворот на левую (геология) и правую (опробование) половины по шву;
  3) выравнивает перекос (deskew по проекционному профилю — Hough на этих бланках врёт);
  4) усиливает контраст (CLAHE + unsharp) — печатная сетка и рукопись читаются лучше;
  5) извлекает сетку: ГОРИЗОНТАЛЬНЫЕ линии строк надёжны (годятся для проверки
     «сколько проходок»); ВЕРТИКАЛЬНЫЕ на этих бланках бледные и почти не тянутся —
     поэтому КОЛОНКИ берём из фикс-шаблона (*_SPEC), а НЕ отсюда.

Запуск:
    python -m src.pipeline.prep data/raw/Дуга_БЛ_4-2021.pdf --dpi 300
    python -m src.pipeline.prep data/raw --dpi 300            # вся папка
Результат: data/interim/<pdf_stem>/<page>_<side>_{half,enhanced,gridmask,overlay}.png
           + <pdf_stem>/manifest.json (углы deskew, число линий, размеры).
"""
from __future__ import annotations
import argparse, json, os, subprocess, tempfile, glob
from pathlib import Path
import cv2
import numpy as np


# ---------- рендер PDF -> изображения (BGR), с учётом поворота ----------
def render_pdf(path: str, dpi: int = 300, first: int | None = None,
               last: int | None = None) -> list[np.ndarray]:
    try:  # предпочтительно pdf2image (см. requirements), poppler обязателен
        from pdf2image import convert_from_path
        pil = convert_from_path(path, dpi=dpi, first_page=first, last_page=last)
        return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pil]
    except Exception:  # запасной путь: poppler pdftoppm напрямую
        with tempfile.TemporaryDirectory() as td:
            cmd = ["pdftoppm", "-r", str(dpi), "-png"]
            if first: cmd += ["-f", str(first)]
            if last:  cmd += ["-l", str(last)]
            cmd += [path, os.path.join(td, "p")]
            subprocess.run(cmd, check=True)
            return [cv2.imread(f) for f in sorted(glob.glob(os.path.join(td, "*.png")))]


# ---------- препроцессинг ----------
def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def split_spread(img: np.ndarray, search=(0.42, 0.58), min_ratio=0.35):
    """Режет разворот по шву (самая светлая колонка в центральной полосе)."""
    g = to_gray(img); h, w = g.shape
    x0, x1 = int(w * search[0]), int(w * search[1])
    gutter = x0 + int(np.argmax(g[:, x0:x1].mean(axis=0)))
    if min(gutter, w - gutter) < w * min_ratio:
        return [("full", img)]
    return [("L", img[:, :gutter].copy()), ("R", img[:, gutter:].copy())]


def deskew_projection(img: np.ndarray, rng=5.0, step=0.25):
    """Deskew: перебор углов, максимизируем резкость строкового профиля."""
    g = to_gray(img)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    small = cv2.resize(bw, (bw.shape[1] // 3, bw.shape[0] // 3),
                       interpolation=cv2.INTER_AREA)
    best_a, best_s = 0.0, -1.0
    cx, cy = small.shape[1] / 2, small.shape[0] / 2
    for a in np.arange(-rng, rng + 1e-6, step):
        M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
        rot = cv2.warpAffine(small, M, (small.shape[1], small.shape[0]),
                             flags=cv2.INTER_NEAREST)
        proj = rot.sum(axis=1, dtype=np.float64)
        s = float(((proj[1:] - proj[:-1]) ** 2).sum())
        if s > best_s:
            best_s, best_a = s, float(a)
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), best_a, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, best_a


def enhance(img: np.ndarray) -> np.ndarray:
    """Локальный контраст (CLAHE) + unsharp. Возвращает серый."""
    g = to_gray(img)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    blur = cv2.GaussianBlur(g, (0, 0), 3)
    return cv2.addWeighted(g, 1.6, blur, -0.6, 0)


def extract_grid(gray_enh: np.ndarray):
    """Маска сетки. Возвращает (grid, n_horiz, n_vert)."""
    bw = cv2.adaptiveThreshold(gray_enh, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)
    h, w = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 40), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 55)))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (w // 12, 1)))
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 12)))

    def count(mask, axis, frac):
        n, _, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        dim = cv2.CC_STAT_WIDTH if axis == 0 else cv2.CC_STAT_HEIGHT
        L = mask.shape[1] if axis == 0 else mask.shape[0]
        return sum(1 for i in range(1, n) if st[i, dim] > frac * L)

    grid = cv2.bitwise_or(horiz, vert)
    return grid, count(horiz, 0, 0.35), count(vert, 1, 0.30)


# ---------- обработка файла ----------
def process_pdf(path: str, out_root: str = "data/interim", dpi: int = 300,
                first=None, last=None, save_masks: bool = True) -> dict:
    stem = Path(path).stem
    out = Path(out_root) / stem
    out.mkdir(parents=True, exist_ok=True)
    pages = render_pdf(path, dpi=dpi, first=first, last=last)
    manifest = {"pdf": path, "dpi": dpi, "pages": len(pages), "sheets": []}
    for pi, spread in enumerate(pages, (first or 1)):
        for side, half in split_spread(spread):
            des, ang = deskew_projection(half)
            enh = enhance(des)
            grid, nh, nv = extract_grid(enh)
            base = f"p{pi:02d}_{side}"
            cv2.imwrite(str(out / f"{base}_half.png"), des)
            cv2.imwrite(str(out / f"{base}_enhanced.png"), enh)
            if save_masks:
                ov = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR); ov[grid > 0] = (0, 0, 255)
                cv2.imwrite(str(out / f"{base}_gridmask.png"), grid)
                cv2.imwrite(str(out / f"{base}_overlay.png"), ov)
            manifest["sheets"].append({
                "page": pi, "side": side, "deskew_deg": round(ang, 2),
                "h_lines": nh, "v_lines": nv,
                "w": int(half.shape[1]), "h": int(half.shape[0]),
                "enhanced": f"{base}_enhanced.png",
            })
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Подготовка сканов буровых журналов")
    ap.add_argument("path", help="PDF-файл или папка с PDF")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="data/interim")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--no-masks", action="store_true")
    a = ap.parse_args()
    targets = ([a.path] if a.path.lower().endswith(".pdf")
               else sorted(glob.glob(os.path.join(a.path, "*.pdf")) +
                           glob.glob(os.path.join(a.path, "*.PDF"))))
    for t in targets:
        m = process_pdf(t, a.out, a.dpi, a.first, a.last, not a.no_masks)
        print(f"{Path(t).name}: страниц {m['pages']}, полулистов {len(m['sheets'])} "
              f"-> {a.out}/{Path(t).stem}/")


if __name__ == "__main__":
    main()
