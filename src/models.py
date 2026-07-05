"""Модели данных.

Таблицы внутри конвейера остаются dict-ами в конвенции reference/dedmos_ocr
(`{rows, cols, grid, conf, expanded, merges, cells}`, cell=`{r,c,rs,cs,text,bbox,conf}`),
чтобы переиспользовать assemble/forms_layout/export без переписывания. Pydantic —
для объектов инвентаризации, паспорта спеки и строк служебного листа `_meta`.
Все bbox — в пикселях УСИЛЕННОЙ логической половины (единое пиксельное пространство).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ObjectType(str, Enum):
    JOURNAL_LEFT = "journal_left"     # главная таблица, графы 1-8
    SAMPLING_RIGHT = "sampling_right" # главная таблица, графы 9-18
    SAMORODKI = "samorodki"           # «Характеристика самородков»
    PODSCHET = "podschet"             # «Результат подсчёта»
    POKAZATELI = "pokazateli"         # мини-таблица «ПОКАЗАТЕЛИ»
    KONTROL = "kontrol"               # мини «Дата контроля / … / Вес металла»
    FORM_JOURNAL = "form_journal"     # титульный лист «ЖУРНАЛ» (key-value)
    FORM_ACT = "form_act"             # «АКТ на завершённую скважину» (key-value)
    UNKNOWN = "unknown"


TABLE_TYPES = {
    ObjectType.JOURNAL_LEFT, ObjectType.SAMPLING_RIGHT, ObjectType.SAMORODKI,
    ObjectType.PODSCHET, ObjectType.POKAZATELI, ObjectType.KONTROL,
}
FORM_TYPES = {ObjectType.FORM_JOURNAL, ObjectType.FORM_ACT}


class BBox(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


class PageObject(BaseModel):
    """Объект, найденный инвентаризацией на логической половине."""

    obj_type: ObjectType
    bbox: BBox
    pdf_page: int            # 1-based страница PDF
    side: str                # "L" | "R" | "full"
    borehole: Optional[int] = None
    confidence: float = 0.0
    note: Optional[str] = None


class MetaRow(BaseModel):
    """Строка служебного листа _meta."""

    sheet: str
    obj_type: str
    r: int
    c: int
    rs: int = 1
    cs: int = 1
    bbox: Optional[list[int]] = None
    text_source: Optional[str] = None   # qwen | yandex | hybrid | spec | ""
    qwen_text: Optional[str] = None
    yandex_text: Optional[str] = None
    conf: Optional[float] = None
    needs_review: bool = False


class InventoryReport(BaseModel):
    pdf_path: str
    pdf_stem: str
    n_pdf_pages: int
    objects: list[PageObject] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.objects:
            out[o.obj_type.value] = out.get(o.obj_type.value, 0) + 1
        return out
