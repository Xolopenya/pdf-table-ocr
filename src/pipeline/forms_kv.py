"""Формы key-value: титульный «ЖУРНАЛ» (поля 1–16 + шапка) и «АКТ».

Раскладку пар «поле → значение» делает Qwen по фикс-списку полей; рукописные
значения при engine=hybrid уточняет Yandex handwritten (пока — пометка низкой
уверенности от Qwen; покро-сверку по полю подключим, когда будут bbox полей).

Подписи полей титула местами бледные — список предварительный, уточнить по
эталону. Структура (набор полей) фиксирована.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .ocr_qwen import QwenVL
from .ocr_yandex import YandexOCR

JOURNAL_FIELDS = [
    "Отряд", "Участок", "Операция",
    "1. Драгир / линия",
    "2. Линия №",
    "3. Азимут бурения линии",
    "4. Скважина №",
    "5. Ствол (наклонный/вертикальный)",
    "6. Отметка устья скважины",
    "7. Глубина скважины",
    "8. Характер коренных пород",
    "9. Пройдено в талом грунте (от–до)",
    "10. Пройдено в мерзлоте",
    "11. Скважина забурена / заброшена",
    "12. Уровень воды и его появление",
    "13. Дебит воды",
    "14. Диаметр обсадных труб",
    "15. Диаметр скважины (начальный/по пласту)",
    "16. Категория грунтов",
]

ACT_FIELDS = [
    "Скважина №",
    "Глубина, м",
    "Заключение акта (текст)",
    "Начальник участка, отряда",
    "Геолог",
    "Буровой мастер",
    "Дата",
]

FORM_FIELDS = {
    "form_journal": ("Титульный лист «ЖУРНАЛ»", JOURNAL_FIELDS),
    "form_act": ("АКТ на завершённую скважину", ACT_FIELDS),
}


def recognize_form(enh_full: np.ndarray, bbox, kind: str, *, engine: str,
                   qwen: QwenVL, yandex: YandexOCR, cfg) -> dict[str, Any]:
    """Вернуть {title, fields:[{key,value,source,conf,needs_review}], kind}."""
    title, field_names = FORM_FIELDS[kind]
    x0, y0, x1, y1 = bbox
    obj = enh_full[y0:y1, x0:x1]

    if engine in ("hybrid", "qwen"):
        kv = qwen.key_value(obj, field_names, title)
        fields = []
        for name in field_names:
            f = kv["fields"].get(name, {"value": "", "low_conf": False})
            fields.append({
                "key": name, "value": f["value"], "source": "qwen",
                "conf": None, "needs_review": bool(f.get("low_conf")),
            })
    else:  # engine=yandex: сырой дамп строк как значения не по полям
        norm = yandex.recognize_cached(obj, model="handwritten")
        text = norm.get("full_text", "")
        fields = [{"key": "(сырой текст Yandex)", "value": text,
                   "source": "yandex", "conf": None, "needs_review": True}]
    return {"title": title, "kind": kind, "fields": fields}
