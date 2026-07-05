"""CLI: программа сама обрабатывает PDF.

  python -m src.cli inventory [pdf]                 # инвентаризация объектов
  python -m src.cli process  [pdf] [--engine hybrid] [--dry-run]
  python -m src.cli process  data/raw/              # вся папка -> .xlsx на файл
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import bootstrap_poppler, load_config
from .logging_setup import logger, setup_logging

app = typer.Typer(add_completion=False, help="Рукописные буровые журналы (PDF) -> Excel")


def _resolve(pdf: Optional[str], cfg):
    if pdf:
        p = Path(pdf)
        return p if p.is_absolute() else cfg.abs_path(pdf)
    return cfg.path_of("reference_pdf")


@app.command()
def inventory(pdf: Optional[str] = typer.Argument(None),
              config: Optional[str] = typer.Option(None, "--config")):
    """Инвентаризация объектов всех страниц PDF -> data/interim (растры, overlay, JSON)."""
    from .pipeline.inventory import inventory_pdf
    cfg = load_config(config)
    setup_logging(cfg.get("logging.level", "INFO"))
    bootstrap_poppler(cfg.get("poppler_path") or None)
    path = _resolve(pdf, cfg)
    if not path.exists():
        raise typer.BadParameter(f"Файл не найден: {path}")
    inventory_pdf(path, cfg, dump=True)


@app.command()
def process(pdf: Optional[str] = typer.Argument(None),
            engine: Optional[str] = typer.Option(None, "--engine",
                                                  help="hybrid | yandex | qwen"),
            dry_run: bool = typer.Option(False, "--dry-run", help="Без сети (только кэш)"),
            first_only: bool = typer.Option(False, "--first-only",
                                            help="только первая скважина (отладка)"),
            config: Optional[str] = typer.Option(None, "--config")):
    """PDF (или папку) -> один .xlsx + .json на файл (все скважины автоматически)."""
    from .pipeline.process import process_pdf
    cfg = load_config(config)
    setup_logging(cfg.get("logging.level", "INFO"))
    bootstrap_poppler(cfg.get("poppler_path") or None)
    eng = engine or cfg.get("engine", "hybrid")

    path = _resolve(pdf, cfg)
    if path.is_dir():
        targets = sorted(list(path.glob("*.pdf")) + list(path.glob("*.PDF")))
    elif path.exists():
        targets = [path]
    else:
        raise typer.BadParameter(f"Не найдено: {path}")

    for t in targets:
        logger.info(f"Обрабатываю: {t.name} (движок={eng})")
        res = process_pdf(t, cfg, engine=eng, dry_run=dry_run,
                          pages=[0] if first_only else None)
        logger.info(f"Готово: {res['xlsx']}  (скважин {res['report']['boreholes']}, "
                    f"листов {res['report']['sheets']})")


if __name__ == "__main__":
    app()
