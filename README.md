# Распознавание рукописных буровых журналов (PDF → Excel)

Программа сама обрабатывает PDF-сканы заполненных от руки бланков «ДОКУМЕНТАЦИЯ
СКВАЖИН УДАРНО-КАНАТНОГО (КОЛОНКОВОГО) БУРЕНИЯ» (АО ЗДП «Коболдо») и выгружает
ВСЕ таблицы и формы в Excel **структурированно** (шапка со слияниями «как в
бланке»), а не плоским текстом. Каждый объект → отдельный лист.

## Архитектура

**КОЛОНКИ — из фикс-спек**, не детектируются (вертикальные рули бланка бледные и
ненадёжны). **СТРОКИ** сверяются с горизонталями (надёжны). **Текст** — гибрид:
Qwen-VL раскладывает строки/колонки и размечает «-ll-»/пустые, Yandex handwritten
читает текст каждой непустой ячейки; расхождение движков → `needs_review`.

Поток: `render_pdf → split_spread → deskew_projection → enhance → extract_grid`
(всё из готового `preprocess.py`) → `inventory` (локатор объектов) →
`hybrid`/`ocr_qwen`/`ocr_yandex` → `forms_layout`/`forms_kv` → `export` (.xlsx+.json+_meta).

| Модуль | Роль |
|---|---|
| `pipeline/preprocess.py` | готовый модуль: рендер, разрез разворота, deskew, CLAHE, сетка |
| `pipeline/inventory.py` | локатор 8 типов объектов на каждой половине (тип + область) |
| `pipeline/forms_layout.py` | 6 фикс-спек + `write_spec` (шапка/слияния/фикс-строки) + `detect_kind` |
| `pipeline/ocr_yandex.py` | Yandex OCR (handwritten/table) + кэш/ретраи/dry-run |
| `pipeline/ocr_qwen.py` | Qwen-VL (OpenRouter): раскладка таблиц + key-value форм |
| `pipeline/hybrid.py` | основной движок «спека×строки» + сверка Qwen↔Yandex |
| `pipeline/forms_kv.py` | титул «ЖУРНАЛ» и «АКТ» как key-value |
| `pipeline/assemble.py` | `expand_ditto`, порядок строк (из reference) |
| `pipeline/export.py` | листы Excel + служебный `_meta` |
| `pipeline/process.py` | оркестратор: PDF → объекты → движок → .xlsx |
| `src/models.py`, `src/cli.py` | pydantic-модели, CLI |
| `eval/score.py` | метрики (a) текст vs GT, (b) число строк, (c) расхождения гибрида |
| `reference/dedmos_ocr/` | проект-референс (база forms_layout/assemble/ocr_yandex) |

Опорные ассеты (`reference/dedmos_ocr/`, `preprocess.py`) переиспользованы, не
переписаны. Прежний геометрический детект колонок удалён (по ТЗ).

## Установка
```bash
python -m pip install -r requirements.txt   # + внешний poppler (для pdf2image)
cp .env.example .env                         # вписать ключи
```
Poppler ищется автоматически (`src.config.bootstrap_poppler`); путь можно задать
в `configs/default.yaml: poppler_path`. Python 3.11+ (проверено на 3.14).

## Ключи (только `.env`, не коммитить, не логировать)
`YANDEX_OCR_API_KEY` (или `YANDEX_API_KEY`), `YANDEX_FOLDER_ID`, `OPENROUTER_API_KEY`.
Логи маскируют ключи; ответы движков кэшируются на диск (`.cache/`) — перезапуск не платит.

## Запуск

**Ядро** — единая функция `src.pipeline.process.process_pdf(path, engine="hybrid")
-> {"xlsx", "json", "report"}`. CLI и GUI — тонкие обёртки над ней. Один PDF →
один `.xlsx` (листы по всем скважинам/объектам) + один `.json`, автоматически.

```bash
# GUI: вставил PDF -> Excel + JSON (обработка в потоке, прогресс, кнопки «Открыть»)
python run_gui.py

# CLI: весь PDF (все скважины) -> data/output/<stem>_<engine>.{xlsx,json}
python -m src.cli process --engine hybrid
python -m src.cli process data/raw/                  # вся папка -> файл на PDF
python -m src.cli process --first-only               # только 1-я скважина (отладка)
python -m src.cli process --dry-run                  # без сети (только кэш)
python -m src.cli inventory                          # инвентаризация (растры/overlay)

# Метрики vs эталон
python -m eval.score data/output/*_hybrid.json --gt eval/groundtruth/скв2.json
```

Статус и следующий шаг — в [PROGRESS.md](PROGRESS.md).
