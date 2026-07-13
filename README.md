# Parsing_AffFace

Локальный пайплайн разбора PDF со **списком аффилированных лиц** (Россия) в JSON.

Используется локальная модель **Ollama** (по умолчанию `qwen2.5vl:7b`) — без облачных API.

## Состав проекта

| Файл | Назначение |
|------|------------|
| `affiliate_table_parse.py` | Детерминированный разбор pipe-таблиц: основания, даты, доли, колонка 3 (адрес / ИНН) |
| `affiliate_table_parse_Local_LLM.py` | Полный пайплайн: PDF → text-layer / vision → Ollama → JSON + QC |
| `requirements.txt` | Зависимости Python |

## Требования

- Python 3.11+
- [Ollama](https://ollama.com/) с моделью `qwen2.5vl:7b` (для сканов и гибридных PDF)
- Опционально: **Tesseract OCR** + `pytesseract` (вспомогательный текст с титула; без него работает vision LLM)

## Установка

```powershell
cd c:\VS\14_pars_AffFace_Local
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5vl:7b
```

## Запуск

```powershell
# Один файл (авто: text-layer или vision для сканов)
python affiliate_table_parse_Local_LLM.py --file "отчёт.pdf"

# Явно vision для сканов
python affiliate_table_parse_Local_LLM.py --file scan.pdf --mode vision

# Пакетная обработка папки
python affiliate_table_parse_Local_LLM.py --input-dir .\pdfs --output-dir output_local_llm
```

### Основные параметры

| Параметр | Описание |
|----------|----------|
| `--mode auto` | text-layer, если текста достаточно; иначе vision (по умолчанию) |
| `--mode text` | только text-layer |
| `--mode vision` | рендер страниц в PNG + vision LLM |
| `--model` | модель Ollama (по умолчанию `qwen2.5vl:7b`) |
| `--scan-dpi` | DPI рендера сканов (по умолчанию 220) |
| `--no-auto-rotate` | не подбирать поворот 0/90/180/270° |
| `--no-qc-retry` | без повторного шага LLM при FAIL QC |

## Ограничения

- PDF **менее 20 страниц**; иначе — «Рекомендуем ручную обработку»
- Сканы и гибриды (титул-скан + OCR) обрабатываются через vision-ветку
- Промежуточные файлы: `output_local_llm/raw_<модель>/`

## Форматы отчётов (колонка 3)

- **`inn_ogrn`** — только физлица: «Согласие физического лица не получено» / ИНН
- **`address`** — место нахождения / жительства; значение **только из ячейки таблицы**, без подстановки с титула

## Поля JSON (служебные)

- `column3_mode` — вариант колонки 3 (`inn_ogrn` / `address`)
- `mixed_affiliate_types` — в тексте есть подразделы «Физические лица» и «Юридические лица»
- `manual_review` — требуется ручная проверка (расхождение QC)

## Автор

[Dmitry Usov](https://github.com/UsovDmitry)
