"""
affiliate_table_parse_Local_LLM.py
==================================

Независимый локальный пайплайн разбора PDF со списком аффилированных лиц.

Основа табличной логики — модуль affiliate_table_parse.py (детерминированный
разбор pipe-таблиц: основания, доли, даты, склейка переносов страниц).

LLM — локальная модель Ollama (по умолчанию qwen2.5vl:7b), без облачных API.

Ограничения автоматической обработки:
    • PDF должен содержать МЕНЕЕ 20 страниц, иначе — «Рекомендуем ручную обработку».
    • Сканы и гибриды (скан + text-layer/OCR) — vision-режим через qwen2.5vl (см. --mode).

Промежуточные артефакты сохраняются в подкаталог raw_<модель>, например:
    output_local_llm/raw_qwen2_5vl_7b/{stem}_section_i_text.txt
    output_local_llm/raw_qwen2_5vl_7b/{stem}_table_step1.txt
    output_local_llm/raw_qwen2_5vl_7b/{stem}_qc.json
    output_local_llm/raw_qwen2_5vl_7b/{stem}_meta.json
    output_local_llm/raw_qwen2_5vl_7b/{stem}_scan_pages/   (PNG сканов + rotation_meta.json)
    output_local_llm/raw_qwen2_5vl_7b/{stem}_status.json   (для пропущенных файлов)

Запуск:
    python affiliate_table_parse_Local_LLM.py --file report.pdf
    python affiliate_table_parse_Local_LLM.py --file scan.pdf --mode vision
    python affiliate_table_parse_Local_LLM.py --input-dir ./pdfs --mode auto
    python affiliate_table_parse_Local_LLM.py --file scan.pdf --mode vision
    python affiliate_table_parse_Local_LLM.py --input-dir ./pdfs --mode auto --model qwen2.5vl:7b
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import time
from enum import Enum
from pathlib import Path
from typing import Any

import fitz
import ollama
from PIL import Image

# ---------------------------------------------------------------------------
# Импорт детерминированного разбора таблиц из единственного базового модуля
# проекта — affiliate_table_parse.py
# ---------------------------------------------------------------------------
from affiliate_table_parse import (
    align_basis_dates_to_bases,
    apply_table_records_to_affiliates,
    count_affiliate_data_rows,
    detect_column3_mode,
    expand_affiliation_basis_list,
    extract_basis_dates,
    extract_basis_dates_for_table_row,
    is_valid_basis_phrase,
    looks_like_postal_address,
    normalize_sequential_row_numbers,
    parse_affiliate_table_records,
    table_has_mixed_affiliate_types,
)

# Опциональные движки извлечения text-layer (устанавливаются отдельно).
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# =============================================================================
# Блок 0. Константы и сообщения пайплайна
# =============================================================================

# Локальная vision-модель Ollama по умолчанию (для сканов и text-layer).
DEFAULT_MODEL = "qwen2.5vl:7b"

# DPI рендера страниц PDF → PNG для vision-ветки.
DEFAULT_SCAN_DPI = 220

# Жёсткий лимит: 20 страниц и более — не обрабатываем автоматически.
MAX_PAGES = 20

# Тексты для оператора при отказе от автоматической обработки.
MSG_TOO_MANY_PAGES = "Рекомендуем ручную обработку"
MSG_ALL_SCAN = "Требуется ручной контроль, файл-сканы"
MSG_MANUAL_REVIEW = "Требуется ручная проверка"
MSG_NEED_VISION_MODEL = (
    "Для сканов нужна vision-модель Ollama (например qwen2.5vl:7b). "
    "Текстовые модели (qwen2.5:14b) не читают изображения страниц."
)

# Минимум символов на странице, чтобы считать её текстовой (не сканом).
MIN_TEXT_CHARS_PER_PAGE = 20

# Контекстное окно для запросов к Ollama (длинные таблицы).
OLLAMA_NUM_CTX = 32768


class SkipReason(str, Enum):
    """Причины пропуска файла без полного разбора."""

    TOO_MANY_PAGES = "too_many_pages"
    ALL_PAGES_SCAN = "all_pages_scan"


def raw_subdir_for_model(model: str) -> str:
    """
    Имя подкаталога для промежуточных файлов: raw_<модель>.
    Двоеточия и точки заменяются на подчёркивания для совместимости с ФС.
    """
    slug = re.sub(r"[^\w\-]+", "_", model.replace(":", "_").replace(".", "_"))
    slug = slug.strip("_") or "local_llm"
    return f"raw_{slug}"


# =============================================================================
# Блок 1. Промпты для локальной LLM (двухшаговый пайплайн: таблица → JSON)
# =============================================================================

SYSTEM_PROMPT = """Извлеки список аффилированных лиц и верни только ровно ОДИН JSON по схеме:
ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА ФОРМАТА (нарушать нельзя):
- Верни ТОЛЬКО JSON. Никакого markdown, никаких тройных кавычек, никаких комментариев.
- Ответ ДОЛЖЕН начинаться с '{' и заканчиваться '}'.
- Если поле невозможно надёжно определить — ставь null (или [] для массивов).
СТРУКТУРА:
{
  "company": {"name": "полное наименование эмитента (ОПФ + «название в кавычках») или null", "ogrn": "13 или 15 цифр или null"},
  "report_date": "ДД.ММ.ГГГГ или null",
  "affiliates": [
    {
      "row_number": 1,
      "full_name": "ФИО или наименование",
      "address": "колонка 3: адрес ИЛИ «Согласие физического лица не получено» / ИНН / ОГРН (зависит от формы отчёта) или null",
      "affiliation_basis": ["основание 1", "основание 2"],
      "basis_date": ["ДД.ММ.ГГГГ"],
      "share_authorized_capital_pct": "строка как в источнике или null",
      "share_ordinary_stocks_pct": "строка как в источнике или null"
    }
  ]
}
- affiliates: ВСЕ строки таблицы аффилированных лиц, без заголовков разделов.
- affiliation_basis — массив полных формулировок (каждое основание — отдельный элемент);
  basis_date — массив дат ДД.ММ.ГГГГ (столько же, сколько оснований, или одна дата на все).
- Доли (колонки 6–7) переноси дословно: "100", "95.999999", "-", "---"."""

TABLE_EXTRACTION_PROMPT = """Извлеки ТОЛЬКО таблицу аффилированных лиц из Раздела I отчёта.
Верни plain text (НЕ JSON), по одной строке таблицы на запись.

Формат каждой строки (колонки через " | "):
N | full_name | col3_identity | affiliation_basis | basis_date | share_auth_pct | share_ord_pct

col3_identity: для отчётов только с физлицами — «Согласие физического лица не получено» или ИНН;
для отчётов с колонкой «место нахождения/жительства» — только адрес ИЗ ЯЧЕЙКИ таблицы (даже если совпадает с адресом эмитента).
ЗАПРЕЩЕНО подставлять адрес эмитента с титула, если его нет в ячейке строки.

Правила:
1) Включи ВСЕ строки таблицы, без заголовков колонок.
2) Склеивай переносы строк внутри ячейки в одну строку.
3) affiliation_basis — полная формулировка; несколько оснований — отдельные элементы массива
   (НЕ одна строка через «;»).
4) basis_date — все даты из колонки 5: если оснований три и даты «08.10.2020, 08.10.2020, 21.06.2024»,
   верни три даты; если одна дата при трёх основаниях — одну дату.
5) Только текст таблицы, без комментариев."""

TABLE_TO_JSON_PROMPT_TEMPLATE = """Ниже извлечённая таблица аффилированных лиц (шаг 1).
Преобразуй её в JSON строго по схеме из system prompt.
Каждая строка таблицы = один объект в affiliates.
company.name — полное наименование эмитента с титульной страницы (ОПФ + название в «кавычках»),
не только организационно-правовая форма («Публичное акционерное общество» без имени — ошибка).
address — только значение колонки 3 из таблицы; если в ячейке «-» или пусто — null. Адрес с титула в affiliates не переносить.
Только JSON.

===== TABLE START =====
{table_text}
===== TABLE END ====="""

COMPANY_COVER_VISION_PROMPT = """На изображении — титульная страница отчёта «Список аффилированных лиц».
Найди полное официальное наименование эмитента (организации, по которой составлен список).
Верни ОДНУ строку: организационно-правовая форма + название в «кавычках», если оно указано на странице.
Пример: Публичное акционерное общество «Ковровский мехонический завод»
Только наименование, без пояснений, кавычек JSON и markdown."""

VISION_USER_PROMPT = """
На изображениях — страницы PDF-отчёта об аффилированных лицах (возможен скан).
Извлеки данные строго по схеме из system prompt.

Важно:
1) Приоритет: «Раздел I. Состав аффилированных лиц ...».
2) Все строки таблицы аффилированных лиц, без заголовков разделов.
3) Не выдумывай записи. Если поле не видно — null.
4) affiliation_basis — отдельные элементы массива; basis_date — все даты из колонки 5.
5) Только JSON-объект.
""".strip()

VISION_TABLE_STEP1_PROMPT = """
На изображениях страниц PDF (скан) извлеки ТОЛЬКО таблицу аффилированных лиц из Раздела I.
Верни plain text (НЕ JSON), по одной строке на запись:

N | full_name | col3_identity | affiliation_basis | basis_date | share_auth_pct | share_ord_pct

col3_identity: для отчётов только с физлицами — «Согласие физического лица не получено» или ИНН;
для отчётов с колонкой «место нахождения/жительства» — только адрес ИЗ ЯЧЕЙКИ таблицы (даже если совпадает с адресом эмитента).
ЗАПРЕЩЕНО подставлять адрес эмитента с титула, если его нет в ячейке строки.

Включи ВСЕ строки. Несколько оснований — через «; » в колонке affiliation_basis.
Все даты из колонки «Дата наступления основания» — в basis_date (через пробел, если несколько).
""".strip()


# =============================================================================
# Блок 2. Предварительный анализ PDF (число страниц, сканы vs text-layer)
# =============================================================================


def get_pdf_page_count(pdf_path: Path) -> int:
    """Возвращает число страниц PDF через PyMuPDF (fitz)."""
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def get_scan_page_indices(
    pdf_path: Path,
    min_chars: int = MIN_TEXT_CHARS_PER_PAGE,
) -> list[int]:
    """Номера страниц (0-based) без достаточного text-layer — считаем сканом."""
    doc = fitz.open(pdf_path)
    try:
        return [
            i
            for i, page in enumerate(doc)
            if len((page.get_text("text") or "").strip()) < min_chars
        ]
    finally:
        doc.close()


def count_pages_with_text_layer(
    pdf_path: Path,
    min_chars: int = MIN_TEXT_CHARS_PER_PAGE,
) -> tuple[int, int]:
    """
    Считает страницы с читаемым text-layer.

    Возвращает (страниц_с_текстом, всего_страниц).
    Страница без достаточного текста считается «картинкой» (скан).
    """
    doc = fitz.open(pdf_path)
    try:
        total = len(doc)
        with_text = 0
        for page in doc:
            text = (page.get_text("text") or "").strip()
            if len(text) >= min_chars:
                with_text += 1
        return with_text, total
    finally:
        doc.close()


def is_all_pages_scan(pdf_path: Path) -> bool:
    """
    True, если ВСЕ страницы PDF — изображения без text-layer.
    Гибридные PDF (часть страниц текстовые) проходят дальше.
    """
    pages_with_text, total = count_pages_with_text_layer(pdf_path)
    if total == 0:
        return False
    return pages_with_text == 0


def check_pdf_eligibility(
    pdf_path: Path,
    mode: str = "auto",
) -> tuple[bool, str | None, SkipReason | None]:
    """
    Проверяет, можно ли автоматически обрабатывать PDF.

    Сканы больше не отсекаются при --mode auto/vision — для них включается vision-ветка.
    При --mode text полный скан по-прежнему пропускается.
    """
    pages = get_pdf_page_count(pdf_path)

    if pages >= MAX_PAGES:
        return False, MSG_TOO_MANY_PAGES, SkipReason.TOO_MANY_PAGES

    if mode == "text" and is_all_pages_scan(pdf_path):
        return False, MSG_ALL_SCAN, SkipReason.ALL_PAGES_SCAN

    return True, None, None


def decide_processing_plan(
    pdf_path: Path,
    extraction: dict[str, Any],
    full_text: str,
    mode: str,
) -> dict[str, Any]:
    """
    Выбор пайплайна: text-layer, полный vision или гибрид (скан + OCR/text).

    Гибрид: первая (или часть) страниц — скан, далее text-layer/OCR.
    Для гибрида в vision передаём только скан-страницы, таблицу берём из текста.
    """
    mode = (mode or "auto").lower().strip()
    total = get_pdf_page_count(pdf_path)
    scan_indices = get_scan_page_indices(pdf_path)
    text_page_count = total - len(scan_indices)
    all_scan = total > 0 and text_page_count == 0
    is_hybrid = bool(scan_indices) and text_page_count > 0

    if mode == "text":
        return {
            "use_vision": False,
            "is_hybrid": is_hybrid,
            "scan_page_indices": scan_indices,
            "pages_to_render": [],
            "pipeline": "text-layer",
        }

    if mode == "vision":
        pages = list(range(total)) if all_scan else scan_indices or list(range(total))
        return {
            "use_vision": True,
            "is_hybrid": is_hybrid and not all_scan,
            "scan_page_indices": scan_indices,
            "pages_to_render": pages,
            "pipeline": "vision-scan" if all_scan else "hybrid-scan-text",
        }

    # auto
    if all_scan:
        return {
            "use_vision": True,
            "is_hybrid": False,
            "scan_page_indices": scan_indices,
            "pages_to_render": list(range(total)),
            "pipeline": "vision-scan",
        }
    if is_hybrid:
        return {
            "use_vision": True,
            "is_hybrid": True,
            "scan_page_indices": scan_indices,
            "pages_to_render": scan_indices,
            "pipeline": "hybrid-scan-text",
        }
    if len((full_text or "").strip()) < 50:
        return {
            "use_vision": True,
            "is_hybrid": False,
            "scan_page_indices": scan_indices,
            "pages_to_render": list(range(total)),
            "pipeline": "vision-scan",
        }
    return {
        "use_vision": False,
        "is_hybrid": False,
        "scan_page_indices": scan_indices,
        "pages_to_render": [],
        "pipeline": "text-layer",
    }


def merge_scan_text_with_extracted(scan_text: str, extracted_text: str) -> str:
    """Склеивает OCR со скан-страниц (титул, компания) и text-layer остальных страниц."""
    scan = (scan_text or "").strip()
    extracted = (extracted_text or "").strip()
    if not scan:
        return extracted
    if not extracted:
        return scan
    if scan[:200] in extracted or scan in extracted:
        return extracted
    return f"{scan}\n\n{extracted}"


def is_vision_capable_model(model: str) -> bool:
    """Vision-модели Ollama (qwen2.5vl, llava, moondream и т.п.)."""
    name = (model or "").lower()
    return any(token in name for token in ("vl", "vision", "llava", "moondream", "minicpm-v"))


# =============================================================================
# Блок 2b. Vision-ветка: рендер сканов, авто-поворот страниц
# =============================================================================


def _horizontal_projection_score(img: Image.Image) -> float:
    """
    Чем выше score, тем вероятнее текстовые строки горизонтальны.
    Используется для выбора угла поворота 0/90/180/270 без Tesseract.
    """
    gray = img.convert("L")
    max_w = 900
    if gray.width > max_w:
        ratio = max_w / gray.width
        gray = gray.resize((max_w, max(1, int(gray.height * ratio))))

    width, height = gray.size
    pixels = gray.load()
    step = 3
    projection: list[float] = []
    for y in range(0, height, step):
        projection.append(sum(255 - pixels[x, y] for x in range(0, width, step)))
    if not projection:
        return 0.0
    mean = sum(projection) / len(projection)
    return sum((v - mean) ** 2 for v in projection) / len(projection)


def detect_best_rotation(img: Image.Image) -> int:
    """Возвращает лучший угол поворота PIL (0/90/180/270, против часовой)."""
    best_angle = 0
    best_score = -1.0
    for angle in (0, 90, 180, 270):
        rotated = img.rotate(angle, expand=True)
        score = _horizontal_projection_score(rotated)
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def rotate_pil_image(img: Image.Image, angle: int) -> Image.Image:
    if angle == 0:
        return img
    return img.rotate(angle, expand=True)


def pdf_to_images(
    pdf_path: Path,
    image_dir: Path,
    dpi: int = DEFAULT_SCAN_DPI,
    auto_rotate: bool = True,
    page_indices: list[int] | None = None,
) -> tuple[list[Path], list[int], list[int]]:
    """
    Рендер страниц PDF в PNG для vision-модели.

    page_indices: какие страницы рендерить (0-based); None — все страницы.
    Возвращает (пути к PNG, углы поворота, номера страниц PDF 1-based).
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        image_paths: list[Path] = []
        rotation_angles: list[int] = []
        source_pages: list[int] = []
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        indices = page_indices if page_indices is not None else list(range(len(doc)))
        for i in indices:
            if i < 0 or i >= len(doc):
                continue
            page = doc[i]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            angle = detect_best_rotation(img) if auto_rotate else 0
            if angle:
                img = rotate_pil_image(img, angle)
            rotation_angles.append(angle)
            source_pages.append(i + 1)

            out_path = image_dir / f"page_{i + 1:03d}.png"
            img.save(out_path)
            image_paths.append(out_path)
        return image_paths, rotation_angles, source_pages
    finally:
        doc.close()


def ocr_page_image_best_psm(path: Path) -> str:
    """OCR одной страницы: несколько PSM, выбираем самый полный результат."""
    try:
        import pytesseract
    except ImportError:
        return ""

    img = Image.open(path)
    candidates: list[str] = []
    for psm in (3, 6, 11):
        try:
            text = pytesseract.image_to_string(
                img, lang="rus+eng", config=f"--psm {psm}"
            )
        except Exception:
            text = ""
        if text.strip():
            candidates.append(text.strip())
    if not candidates:
        return ""
    return max(candidates, key=len)


def ocr_images_to_text(image_paths: list[Path]) -> str:
    """
    Опциональный OCR (pytesseract + Tesseract) для вспомогательного section_text.
    Не обязателен: основной разбор идёт через vision LLM.
    """
    parts: list[str] = []
    for path in image_paths:
        text = ocr_page_image_best_psm(path)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def write_scan_pages_meta(
    scan_dir: Path,
    image_paths: list[Path],
    rotation_angles: list[int],
    source_page_numbers: list[int],
) -> None:
    """Метаданные PNG скан-страниц (углы поворота, номера страниц PDF)."""
    meta = {
        "rotation_angles": rotation_angles,
        "source_page_numbers": source_page_numbers,
        "page_count": len(image_paths),
        "pages": [p.name for p in image_paths],
    }
    (scan_dir / "rotation_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# Блок 3. Извлечение text-layer из PDF (несколько движков, выбор лучшего)
# =============================================================================


def extract_with_pdfplumber(pdf_path: Path) -> str:
    """pdfplumber: текст + таблицы (строки через ' | ')."""
    if pdfplumber is None:
        return ""
    try:
        parts: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                if len(page_text.strip()) > 20:
                    parts.append(f"--- Страница {page_num} ---\n{page_text.strip()}")

                for table_idx, table in enumerate(page.extract_tables() or []):
                    if not table or len(table) <= 1:
                        continue
                    rows: list[str] = []
                    for row in table:
                        if row and any(cell for cell in row if cell):
                            rows.append(" | ".join(str(cell or "").strip() for cell in row))
                    if rows:
                        parts.append(
                            f"--- Таблица {table_idx + 1} на странице {page_num} ---\n"
                            + "\n".join(rows)
                        )
        full_text = "\n\n".join(parts)
        return full_text if len(full_text.strip()) > 100 else ""
    except Exception as exc:
        print(f"    pdfplumber: {exc}")
        return ""


def extract_with_fitz(pdf_path: Path) -> str:
    """PyMuPDF: быстрый fallback для text-layer."""
    try:
        doc = fitz.open(pdf_path)
        try:
            chunks: list[str] = []
            for i, page in enumerate(doc):
                page_text = page.get_text("text") or ""
                if len(page_text.strip()) > 20:
                    chunks.append(f"--- Страница {i + 1} ---\n{page_text.strip()}")
            full_text = "\n\n".join(chunks)
            return full_text if len(full_text.strip()) > 100 else ""
        finally:
            doc.close()
    except Exception as exc:
        print(f"    pymupdf: {exc}")
        return ""


def extract_with_pypdf(pdf_path: Path) -> str:
    """pypdf: запасной вариант извлечения текста."""
    if PdfReader is None:
        return ""
    try:
        parts: list[str] = []
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if len(text.strip()) > 20:
                parts.append(f"--- Страница {page_num} ---\n{text.strip()}")
        full_text = "\n\n".join(parts)
        return full_text if len(full_text.strip()) > 100 else ""
    except Exception as exc:
        print(f"    pypdf: {exc}")
        return ""


def pdf_has_text_layer(pdf_path: Path) -> bool:
    """Есть ли читаемый text-layer хотя бы на первой странице."""
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                first_text = (pdf.pages[0].extract_text() or "").strip()
                return len(first_text) > 20
        except Exception:
            pass
    try:
        doc = fitz.open(pdf_path)
        try:
            first_text = (doc[0].get_text("text") or "").strip()
            return len(first_text) > 20
        finally:
            doc.close()
    except Exception:
        return False


def smart_pdf_extract(pdf_path: Path, verbose: bool = True) -> dict[str, Any]:
    """
    Умное извлечение текста: пробуем pdfplumber / pymupdf / pypdf,
    выбираем самый полный результат (приоритет pipe-таблицам).
    """
    if verbose:
        print(f"  [extract] Анализ: {pdf_path.name}")

    has_text_layer = pdf_has_text_layer(pdf_path)
    methods: list[tuple[str, Any]] = (
        [
            ("pdfplumber", extract_with_pdfplumber),
            ("pymupdf", extract_with_fitz),
            ("pypdf", extract_with_pypdf),
        ]
        if has_text_layer
        else [
            ("pymupdf", extract_with_fitz),
            ("pdfplumber", extract_with_pdfplumber),
            ("pypdf", extract_with_pypdf),
        ]
    )

    best_text = ""
    best_method: str | None = None
    for method_name, method_func in methods:
        if verbose:
            print(f"    пробуем {method_name}...")
        text = method_func(pdf_path)
        if text and len(text.strip()) > len(best_text):
            best_text = text
            best_method = method_name
            if len(best_text.strip()) > 500 and " | " in best_text:
                break

    return {
        "text": best_text,
        "method": best_method,
        "has_text_layer": has_text_layer,
        "text_length": len(best_text),
    }


# =============================================================================
# Блок 4. Выделение Раздела I и обогащение pipe-таблицами
# =============================================================================


def enrich_section_with_tables(full_text: str, section_text: str) -> str:
    """
    Если pdfplumber извлёк таблицы со строками ' | ', добавляем их в конец текста.
    Детерминированный разбор в affiliate_table_parse.py использует блок ===== ТАБЛИЦЫ.
    """
    table_blocks = re.findall(
        r"--- Таблица \d+ на странице \d+ ---[\s\S]*?(?=\n--- |\Z)",
        full_text,
    )
    if not table_blocks:
        return section_text
    tables_part = "\n\n".join(block.strip() for block in table_blocks)
    return (
        section_text
        + "\n\n===== ТАБЛИЦЫ (структурированный вид, колонки через |) =====\n"
        + tables_part
    )


def extract_section_i_text(full_text: str) -> str:
    """
    Извлекает фрагмент вокруг «Раздел I» / «Состав аффилированных лиц».
    Если границы не найдены — возвращает весь текст с таблицами.
    """
    start_patterns = [
        r"раздел\s*i\b",
        r"состав\s+аффилированных\s+лиц\s+на",
    ]
    end_patterns = [
        r"раздел\s*ii\b",
        r"изменения,\s*произошедшие\s+в\s+списке",
        r"сведения\s+о\s+списке\s+аффилированных",
    ]

    text_lower = full_text.lower()
    starts = [re.search(p, text_lower, flags=re.IGNORECASE) for p in start_patterns]
    starts = [m.start() for m in starts if m]
    if not starts:
        return enrich_section_with_tables(full_text, full_text)

    start_idx = min(starts)
    tail = text_lower[start_idx:]
    ends = [re.search(p, tail, flags=re.IGNORECASE) for p in end_patterns]
    ends = [m.start() for m in ends if m]
    if not ends:
        section = full_text[start_idx:].strip()
        return enrich_section_with_tables(full_text, section)

    end_idx = start_idx + min(ends)
    section = full_text[start_idx:end_idx].strip()
    return enrich_section_with_tables(full_text, section)


# =============================================================================
# Блок 5. Вызовы локальной Ollama (шаг 1: таблица, шаг 2: JSON)
# =============================================================================


def extract_json_block(text: str) -> str:
    """Вырезает первый JSON-объект из ответа модели."""
    text = text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("LLM не вернула JSON")
    return match.group(0)


def ollama_chat_json(
    model: str,
    messages: list[dict[str, Any]],
    num_ctx: int = OLLAMA_NUM_CTX,
) -> dict:
    """Запрос к Ollama с format=json и повтором при невалидном JSON."""
    response = ollama.chat(
        model=model,
        options={"temperature": 0.0, "num_ctx": num_ctx},
        format="json",
        messages=messages,
    )
    raw = response["message"]["content"]
    try:
        return json.loads(extract_json_block(raw))
    except (json.JSONDecodeError, ValueError):
        retry = ollama.chat(
            model=model,
            options={"temperature": 0.0, "num_ctx": num_ctx},
            format="json",
            messages=messages
            + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Верни строго валидный JSON. Только JSON-объект."},
            ],
        )
        return json.loads(extract_json_block(retry["message"]["content"]))


def extract_table_step1(
    section_text: str,
    model: str,
    image_paths: list[Path] | None = None,
) -> str:
    """
    Шаг 1 двухшагового пайплайна: извлечь таблицу аффилиатов как plain text.

    image_paths: при передаче PNG страниц — vision-режим (сканы, перевёрнутые страницы).
    """
    if image_paths:
        user_parts = [VISION_TABLE_STEP1_PROMPT]
        if section_text.strip():
            user_parts.append(
                "\nТекст таблицы и Раздела I (text-layer / OCR страниц с текстом):\n"
                f"{section_text}\n\n"
                "Изображения — скан-страницы (титул, шапка, наименование компании). "
                "Объедини данные с картинок и текста."
            )
        user_content = "\n".join(user_parts)
        response = ollama.chat(
            model=model,
            options={"temperature": 0.0, "num_ctx": OLLAMA_NUM_CTX},
            messages=[
                {"role": "system", "content": TABLE_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": user_content,
                    "images": [str(p) for p in image_paths],
                },
            ],
        )
    else:
        user_content = (
            f"Исходный текст документа (Раздел I):\n\n{section_text}\n\n"
            "Верни только таблицу в формате из инструкции."
        )
        response = ollama.chat(
            model=model,
            options={"temperature": 0.0, "num_ctx": OLLAMA_NUM_CTX},
            messages=[
                {"role": "system", "content": TABLE_EXTRACTION_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
    return (response["message"]["content"] or "").strip()


def structure_table_step2_json(
    table_text: str,
    model: str,
    extra_user_prompt: str | None = None,
) -> dict:
    """Шаг 2: структурированная таблица → JSON по схеме проекта."""
    user_prompt = TABLE_TO_JSON_PROMPT_TEMPLATE.format(table_text=table_text)
    if extra_user_prompt:
        user_prompt += f"\n\n{extra_user_prompt}"
    return ollama_chat_json(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )


def extract_affiliates_two_step(
    section_text: str,
    model: str,
    image_paths: list[Path] | None = None,
) -> tuple[dict, str]:
    """
    Двухшаговый пайплайн: таблица (plain text) → JSON.
    Возвращает (result_json, table_text).

    image_paths: PNG страниц для vision-шага 1 (сканы).
    """
    if image_paths:
        print(
            f"[STEP 1] Vision: извлечение таблицы со {len(image_paths)} стр. (Ollama)..."
        )
    else:
        print("[STEP 1] Извлечение таблицы аффилированных лиц (Ollama)...")
    table_text = extract_table_step1(
        section_text=section_text,
        model=model,
        image_paths=image_paths,
    )
    print(f"[STEP 1] Таблица: {len(table_text)} символов")
    print("[STEP 2] Структурирование таблицы в JSON (Ollama)...")
    result = structure_table_step2_json(table_text=table_text, model=model)
    return result, table_text


def extract_affiliates_with_qc_retry(
    section_text: str,
    model: str,
    qc_report: dict,
    table_text: str,
) -> dict:
    """
    Повтор шага 2 при FAIL mini-QC: недостача строк или обрезанные основания.
    """
    expected_rows = qc_report.get("expected_rows")
    actual_rows = qc_report.get("actual_rows") or 0
    basis_issues = qc_report.get("basis_issues") or []

    fix_parts: list[str] = []
    if expected_rows is not None and actual_rows < expected_rows:
        fix_parts.append(
            f"В таблице должно быть {expected_rows} записей (сейчас {actual_rows}). "
            "Верни все строки без пропусков."
        )
    if basis_issues:
        examples = []
        for issue in basis_issues[:5]:
            examples.append(
                f'- "{issue.get("full_name")}": основание неполное '
                f'("{issue.get("basis")}"), верни полную формулировку.'
            )
        fix_parts.append(
            "Исправь affiliation_basis (полные формулировки):\n" + "\n".join(examples)
        )
    fix_parts.append("Верни ПОЛНЫЙ JSON. Только JSON.")
    return structure_table_step2_json(
        table_text=table_text,
        model=model,
        extra_user_prompt="\n".join(fix_parts),
    )


# =============================================================================
# Блок 6. Нормализация ответа LLM и доводка дат оснований
# =============================================================================


def normalize_date(value: Any) -> str | None:
    """Нормализация даты в формат ДД.ММ.ГГГГ."""
    if not value:
        return None
    match = re.search(r"(\d{2})[./-](\d{2})[./-](\d{4})", str(value))
    if not match:
        return None
    day, month, year = match.groups()
    return f"{day}.{month}.{year}"


def normalize_ogrn(value: Any) -> str | None:
    """ОГРН юрлица — 13 цифр, ОГРНИП — 15 цифр."""
    if value is None:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) in (13, 15):
        return digits
    return None


def extract_ogrn_from_text(text: str) -> str | None:
    """ОГРН из блока «Коды эмитента»."""
    patterns = [
        r"коды\s+эмитента[\s\S]{0,400}?огрн\s*[:\s]*(\d{13})",
        r"огрн\s*[:\s/]*(\d{13})",
        r"\b(\d{13})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            ogrn = normalize_ogrn(match.group(1))
            if ogrn:
                return ogrn
    return None


def is_incomplete_company_name(name: str | None) -> bool:
    """
    True, если наименование похоже на обрезанное (только ОПФ без «имени в кавычках»).
    """
    if not name or not str(name).strip():
        return True
    n = re.sub(r"\s+", " ", str(name).strip())
    lower = n.lower()
    if re.search(r"[«\"]([^»\"]{2,})[»\"]", n):
        return False
    generic_only = (
        r"^публичное акционерное общество$",
        r"^непубличное акционерное общество$",
        r"^акционерное общество$",
        r"^общество с ограниченной ответственностью$",
        r"^открытое акционерное общество$",
        r"^закрытое акционерное общество$",
    )
    return any(re.match(p, lower) for p in generic_only)


def extract_company_name_from_text(text: str) -> str | None:
    """
    Наименование эмитента после «СПИСОК АФФИЛИРОВАННЫХ ЛИЦ».

    Собирает несколько строк: ОПФ + название в «кавычках» на следующей строке.
    """
    if not text:
        return None

    search_areas = [text, text[:3000]]
    block: str | None = None
    for area in search_areas:
        match = re.search(
            r"список\s+аффилированных\s+лиц\s*\n+(.+)",
            area,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            block = match.group(1)
            break
    if not block:
        return None

    stop_re = re.compile(
        r"^(огрн|инн|номер протокола|адрес|место нахождения|телефон|факс|"
        r"на \d|раздел|состав|сведения|п/п|№\s|---|"
        r"\(полное|полное фирменное|наименование \(для)",
        re.IGNORECASE,
    )

    name_parts: list[str] = []
    for raw_line in block.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if len(line) < 3:
            continue
        lower = line.lower()
        if stop_re.search(lower):
            break
        if "|" in line or re.match(r"^\d[\d\s.,]*$", line):
            break
        if line.startswith("(") and not name_parts:
            continue
        name_parts.append(line)
        if re.search(r"[«»\"]", line):
            break
        if len(name_parts) >= 4:
            break

    if not name_parts:
        return None
    name = " ".join(name_parts)
    return name if len(name) >= 5 else None


def extract_company_name_via_vision(
    image_paths: list[Path],
    model: str,
) -> str | None:
    """Точечный vision-запрос: полное наименование эмитента с титульной страницы-скана."""
    if not image_paths:
        return None
    response = ollama.chat(
        model=model,
        options={"temperature": 0.0, "num_ctx": OLLAMA_NUM_CTX},
        messages=[
            {
                "role": "user",
                "content": COMPANY_COVER_VISION_PROMPT,
                "images": [str(image_paths[0])],
            }
        ],
    )
    name = (response["message"]["content"] or "").strip()
    name = name.strip("\"'`*")
    name = re.sub(r"\s+", " ", name)
    if len(name) < 8:
        return None
    return name


def resolve_company_name(
    company: dict,
    full_text: str,
    cover_images: list[Path] | None = None,
    model: str | None = None,
) -> None:
    """
    Выбирает лучшее company.name: LLM → OCR/текст → vision с титульной страницы.
    """
    current = (company.get("name") or "").strip()
    from_text = extract_company_name_from_text(full_text)

    best = current
    if from_text:
        if (
            not current
            or is_incomplete_company_name(current)
            or len(from_text) > len(current)
        ):
            best = from_text

    if (
        is_incomplete_company_name(best)
        and cover_images
        and model
        and is_vision_capable_model(model)
    ):
        print("[COMPANY] Уточнение наименования с титульной страницы (vision)...")
        vision_name = extract_company_name_via_vision(cover_images, model)
        if vision_name and (
            not best
            or is_incomplete_company_name(best)
            or len(vision_name) > len(best)
        ):
            best = vision_name

    if best:
        company["name"] = best


def extract_report_date_from_section(section_text: str, pdf_path: Path) -> str | None:
    """Дата отчёта из заголовка раздела или имени файла."""
    header_match = re.search(
        r"состав\s+аффилированных\s+лиц\s+на\s*(\d{2}[./-]\d{2}[./-]\d{4})",
        section_text,
        flags=re.IGNORECASE,
    )
    if header_match:
        return normalize_date(header_match.group(1))

    file_match = re.search(r"(\d{2}\.\d{2}\.\d{4})", pdf_path.name)
    if file_match:
        return file_match.group(1)
    return None


def normalize_address(value: Any, column3_mode: str = "address") -> str | None:
    """
    Нормализация колонки 3 таблицы (в JSON — поле address).

    inn_ogrn: сохраняем «Согласие...» и ИНН; address: «согласие» → null.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("-", "---"):
        return None
    if column3_mode == "inn_ogrn":
        if "согласие" in text.lower():
            if "не получено" in text.lower():
                return "Согласие физического лица не получено"
            return text
        digits = re.sub(r"\D", "", text)
        if len(digits) in (10, 12, 13, 15):
            return digits
        if looks_like_postal_address(text):
            return None
        return text
    if "согласие" in text.lower():
        return None
    return text


def normalize_share_pct(value: Any) -> str | None:
    """Доли участия — строка как в источнике."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    if re.fullmatch(r"[-—–]+", text):
        return text
    cleaned = text.replace("%", "").replace(",", ".").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return cleaned
    return text


def normalize_result(
    result: dict,
    pdf_path: Path,
    section_text: str,
    full_text: str,
    cover_images: list[Path] | None = None,
    model: str | None = None,
) -> dict:
    """
    Нормализация ответа модели: report_date, company, адреса, даты, доли.
    Детерминированный разбор таблицы выполняется отдельно через affiliate_table_parse.
    """
    if not isinstance(result, dict):
        return result

    report_date = extract_report_date_from_section(section_text, pdf_path)
    if not report_date:
        report_date = normalize_date(result.get("report_date"))
    result["report_date"] = report_date

    column3_mode = detect_column3_mode(section_text)
    result["column3_mode"] = column3_mode
    result["mixed_affiliate_types"] = table_has_mixed_affiliate_types(section_text)

    company = result.get("company")
    if not isinstance(company, dict):
        company = {}
        result["company"] = company

    resolve_company_name(company, full_text, cover_images=cover_images, model=model)

    ogrn = normalize_ogrn(company.get("ogrn")) or extract_ogrn_from_text(full_text)
    company["ogrn"] = ogrn

    affiliates = result.get("affiliates")
    if isinstance(affiliates, list):
        for row in affiliates:
            if not isinstance(row, dict):
                continue
            if column3_mode == "address":
                row["address"] = None
            else:
                row["address"] = normalize_address(
                    row.get("address"), column3_mode=column3_mode
                )
            row["share_authorized_capital_pct"] = normalize_share_pct(
                row.get("share_authorized_capital_pct")
            )
            row["share_ordinary_stocks_pct"] = normalize_share_pct(
                row.get("share_ordinary_stocks_pct")
            )
            bases = row.get("affiliation_basis") or []
            if isinstance(bases, list):
                bases = expand_affiliation_basis_list(bases)
                row["affiliation_basis"] = bases
                dates = row.get("basis_date") or []
                if isinstance(dates, list):
                    row["basis_date"] = align_basis_dates_to_bases(
                        bases,
                        [d for d in (normalize_date(v) for v in dates) if d],
                    )

    return result


def enforce_basis_dates_for_affiliates(
    affiliates: list[dict],
    section_text: str,
    table_text: str | None = None,
) -> list[dict]:
    """
    Финальная привязка basis_date к affiliation_basis.

    Правила:
        • N оснований и N дат — сохраняем все даты (в т.ч. повторы);
        • N оснований и 1 дата — одна дата относится ко всем основаниям;
        • дат меньше оснований (но >1) — дополняем последней датой.

    Источники дат (берём самый полный набор):
        1. Детерминированная pipe-таблица из section_text;
        2. Окно строки таблицы в section_text (многострочные даты в колонке 5);
        3. Уже разобранный basis_date в записи;
        4. Ячейка дат из table_step1.txt (шаг 1 LLM).
    """
    table_records = parse_affiliate_table_records(section_text)
    step1_dates_by_name: dict[str, list[str]] = {}
    if table_text:
        for line in table_text.splitlines():
            if " | " not in line:
                continue
            parts = [p.strip() for p in line.split(" | ")]
            if len(parts) < 5:
                continue
            name_key = parts[1][:40].strip().lower()
            if name_key:
                step1_dates_by_name[name_key] = extract_basis_dates(parts[4])

    for row in affiliates:
        if not isinstance(row, dict):
            continue

        bases = expand_affiliation_basis_list(row.get("affiliation_basis") or [])
        row["affiliation_basis"] = bases

        existing = [str(d).strip() for d in (row.get("basis_date") or []) if str(d).strip()]

        table_dates: list[str] = []
        row_number = row.get("row_number")
        if isinstance(row_number, int) and row_number in table_records:
            table_dates = table_records[row_number].get("basis_date") or []
        if not table_dates:
            name_key = str(row.get("full_name") or "")[:40].strip().lower()
            for trec in table_records.values():
                tname = str(trec.get("full_name") or "")[:40].strip().lower()
                if name_key and (name_key in tname or tname in name_key):
                    table_dates = trec.get("basis_date") or []
                    break

        section_dates: list[str] = []
        if isinstance(row_number, int):
            section_dates = extract_basis_dates_for_table_row(section_text, row_number)

        step1_dates = step1_dates_by_name.get(
            str(row.get("full_name") or "")[:40].strip().lower(), []
        )

        # Приоритет у источника с наибольшим числом дат (часто section_text > step1 LLM).
        raw_dates = max(
            [existing, table_dates, section_dates, step1_dates],
            key=len,
        )
        row["basis_date"] = align_basis_dates_to_bases(bases, raw_dates)

    return affiliates


# =============================================================================
# Блок 7. Mini-QC без второй модели (сверка числа строк и полноты оснований)
# =============================================================================

TABLE_ROW_PATTERN = re.compile(r"(?m)^(\d{1,3})\.\s+(?=[А-ЯA-ZЁ«\"])")


def count_step1_table_rows(table_text: str | None) -> int | None:
    """Число строк в pipe-таблице шага 1."""
    if not table_text:
        return None
    count = sum(
        1 for line in table_text.splitlines() if re.match(r"^\d+\s*\|", line.strip())
    )
    return count or None


def looks_like_truncated_basis(basis: str) -> bool:
    """Эвристика: основание похоже на обрезанный фрагмент."""
    text = basis.strip()
    if not text or len(text) < 12:
        return True
    if re.search(r";\s*Л\.?$", text):
        return True
    if not is_valid_basis_phrase(text):
        return True
    return False


def check_affiliation_basis_completeness(result: dict) -> tuple[list[dict], bool]:
    """QC полноты affiliation_basis по каждой записи."""
    issues: list[dict] = []
    affiliates = result.get("affiliates") if isinstance(result, dict) else None
    if not isinstance(affiliates, list):
        return issues, False

    for idx, row in enumerate(affiliates):
        if not isinstance(row, dict):
            continue
        full_name = str(row.get("full_name") or "")
        basis_list = row.get("affiliation_basis") or []
        if not basis_list:
            issues.append(
                {
                    "record_index": idx,
                    "full_name": full_name,
                    "basis": None,
                    "reason": "empty_affiliation_basis",
                }
            )
            continue
        for basis in basis_list:
            text = str(basis).strip()
            if looks_like_truncated_basis(text):
                issues.append(
                    {
                        "record_index": idx,
                        "full_name": full_name,
                        "basis": text,
                        "reason": "truncated_or_fragment",
                    }
                )
    return issues, len(issues) > 0


def count_affiliate_rows_in_section(
    section_text: str,
    table_text: str | None = None,
) -> int | None:
    """
    Ожидаемое число записей в таблице PDF.

    Приоритет: детерминированный разбор pipe-таблицы (источник PDF),
    затем step1 LLM, затем эвристика по нумерации в тексте.
    """
    from_table = count_affiliate_data_rows(section_text)
    if from_table:
        return from_table

    from_step1 = count_step1_table_rows(table_text)
    if from_step1:
        return from_step1

    numbers = [int(m.group(1)) for m in TABLE_ROW_PATTERN.finditer(section_text)]
    return len(numbers) if numbers else None


def finalize_affiliates_list(
    affiliates: list[dict],
    section_text: str,
    table_text: str | None = None,
    full_text: str | None = None,
) -> list[dict]:
    """
    Финальная доводка affiliates: таблица → даты → последовательные номера п/п.
    """
    column3_mode = detect_column3_mode(section_text)
    affiliates = apply_table_records_to_affiliates(
        affiliates,
        section_text,
        full_text=full_text,
    )
    affiliates = enforce_basis_dates_for_affiliates(
        affiliates,
        section_text=section_text,
        table_text=table_text,
    )
    for row in affiliates:
        if isinstance(row, dict):
            row["address"] = normalize_address(row.get("address"), column3_mode=column3_mode)
    return normalize_sequential_row_numbers(affiliates)


def attach_review_status_to_result(result: dict, qc_report: dict) -> dict:
    """
    Добавляет в итоговый JSON явный статус для оператора.

    При expected ≠ actual — «Требуется ручная проверка».
    """
    if not isinstance(result, dict):
        return result

    expected = qc_report.get("expected_rows")
    actual = qc_report.get("actual_rows")
    row_mismatch = (
        expected is not None
        and isinstance(actual, int)
        and expected != actual
    )

    result["manual_review"] = bool(
        row_mismatch or qc_report.get("status") == "FAIL"
    )
    if row_mismatch:
        result["status_message"] = MSG_MANUAL_REVIEW
        result["review_reason"] = (
            f"Строк в таблице PDF: {expected}, записей в JSON: {actual}"
        )
    elif qc_report.get("status") == "FAIL":
        result["status_message"] = MSG_MANUAL_REVIEW
        result["review_reason"] = qc_report.get("final_note")
    elif qc_report.get("status") == "WARN":
        result["status_message"] = qc_report.get("final_note")
        result["review_reason"] = None
    else:
        result["status_message"] = "OK"
        result["review_reason"] = None

    result["qc_status"] = qc_report.get("status")
    result["expected_rows"] = expected
    result["actual_rows"] = actual
    return result


def run_mini_qc(
    section_text: str,
    result: dict,
    table_text: str | None = None,
    is_scan: bool = False,
) -> dict:
    """
    Лёгкий QC: число строк, report_date, company, affiliation_basis.
    Без вызова второй модели.

    is_scan: для сканов не требуем report_date/ОГРН/название компании (часто нечитаемы).
    """
    affiliates = result.get("affiliates") if isinstance(result, dict) else None
    actual_rows = len(affiliates) if isinstance(affiliates, list) else 0
    expected_rows = count_affiliate_rows_in_section(section_text, table_text=table_text)

    checks: list[dict] = []
    hard_fail = False

    if expected_rows is not None:
        row_ok = actual_rows == expected_rows
        checks.append(
            {
                "code": "row_count",
                "ok": row_ok,
                "expected_rows": expected_rows,
                "actual_rows": actual_rows,
                "message": (
                    f"Строк в таблице PDF: {expected_rows}, "
                    f"записей в JSON: {actual_rows}"
                ),
            }
        )
        if not row_ok:
            hard_fail = True
            qc_report_row_mismatch = True
        else:
            qc_report_row_mismatch = False
    else:
        qc_report_row_mismatch = False
        checks.append(
            {
                "code": "row_count",
                "ok": True,
                "expected_rows": None,
                "actual_rows": actual_rows,
                "message": "Нумерация не найдена, проверка пропущена",
            }
        )

    report_date = result.get("report_date") if isinstance(result, dict) else None
    date_ok = isinstance(report_date, str) and bool(
        re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", report_date.strip())
    )
    checks.append(
        {
            "code": "report_date_format",
            "ok": date_ok,
            "message": "report_date OK" if date_ok else "Некорректный report_date",
        }
    )
    if not date_ok and not is_scan:
        hard_fail = True

    company = result.get("company") if isinstance(result, dict) else {}
    company = company if isinstance(company, dict) else {}
    ogrn = company.get("ogrn")
    ogrn_ok = isinstance(ogrn, str) and len(ogrn) in (13, 15) and ogrn.isdigit()
    checks.append(
        {
            "code": "company_ogrn",
            "ok": ogrn_ok,
            "message": f"company.ogrn: {ogrn}" if ogrn_ok else "company.ogrn не найден",
        }
    )
    if not ogrn_ok and not is_scan:
        hard_fail = True

    company_name = company.get("name")
    name_ok = isinstance(company_name, str) and len(company_name.strip()) >= 5
    checks.append(
        {
            "code": "company_name",
            "ok": name_ok,
            "message": f"company.name OK" if name_ok else "company.name не найдено",
        }
    )
    if not name_ok and not is_scan:
        hard_fail = True

    basis_issues, basis_fail = check_affiliation_basis_completeness(result)
    checks.append(
        {
            "code": "affiliation_basis_completeness",
            "ok": not basis_fail,
            "issues_count": len(basis_issues),
            "issues": basis_issues,
            "message": (
                "affiliation_basis полные"
                if not basis_fail
                else f"неполные основания: {len(basis_issues)}"
            ),
        }
    )
    if basis_fail:
        hard_fail = True

    if qc_report_row_mismatch:
        status = "FAIL"
        note = (
            f"{MSG_MANUAL_REVIEW}: строк в таблице PDF — {expected_rows}, "
            f"записей в JSON — {actual_rows}."
        )
    elif hard_fail:
        status = "FAIL"
        note = (
            "Расхождения по affiliation_basis (скан)."
            if is_scan
            else "Расхождения: дата, company или affiliation_basis."
        )
    elif expected_rows is None:
        status = "WARN"
        note = "Базовые проверки OK, нумерацию таблицы определить не удалось."
    else:
        status = "PASS"
        note = "Количество записей совпадает с таблицей."

    return {
        "status": status,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "checks": checks,
        "basis_issues": basis_issues,
        "final_note": note,
        "row_count_mismatch": qc_report_row_mismatch,
        "manual_review": qc_report_row_mismatch or hard_fail,
        "manual_review_message": MSG_MANUAL_REVIEW
        if (qc_report_row_mismatch or hard_fail)
        else None,
    }


def should_run_qc_retry(qc_report: dict) -> bool:
    """Нужен ли повтор при FAIL: недостача строк или проблемы с основаниями."""
    if qc_report.get("status") != "FAIL":
        return False
    expected = qc_report.get("expected_rows")
    actual = qc_report.get("actual_rows") or 0
    if expected is not None and actual < expected:
        return True
    if qc_report.get("basis_issues"):
        return True
    return False


def print_qc_summary(qc: dict) -> None:
    """Печать краткого отчёта QC в консоль."""
    print(f"\n===== QC status: {qc.get('status')} =====")
    print(qc.get("final_note", ""))
    if qc.get("manual_review_message"):
        print(f"[MANUAL] {qc.get('manual_review_message')}")
    for check in qc.get("checks", []):
        mark = "OK" if check.get("ok") else "FAIL"
        print(f"  [{mark}] {check.get('code')}: {check.get('message')}")


# =============================================================================
# Блок 8. Сохранение промежуточных файлов в raw_<модель>
# =============================================================================


def ensure_raw_dir(output_dir: Path, model: str) -> Path:
    """Создаёт output_dir/raw_<модель> и возвращает путь."""
    raw_dir = output_dir / raw_subdir_for_model(model)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def save_skip_status(
    raw_dir: Path,
    pdf_path: Path,
    message: str,
    reason: SkipReason,
    pages: int,
    model: str,
) -> Path:
    """JSON-статус для пропущенного файла (слишком много страниц или скан)."""
    status_file = raw_dir / f"{pdf_path.stem}_status.json"
    payload: dict[str, Any] = {
        "source_file": str(pdf_path),
        "status": "skipped",
        "message": message,
        "skip_reason": reason.value,
        "pages": pages,
        "max_pages_allowed": MAX_PAGES,
        "model": model,
        "pipeline": "affiliate_table_parse_Local_LLM",
        "created_at_unix": int(time.time()),
    }
    status_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return status_file


def save_raw_intermediates(
    raw_dir: Path,
    pdf_path: Path,
    *,
    section_text: str,
    table_text: str | None,
    qc_report: dict,
    meta: dict[str, Any],
) -> None:
    """
    Промежуточные артефакты успешного прогона:
        _section_i_text.txt — текст Раздела I;
        _table_step1.txt    — таблица после шага 1 LLM;
        _qc.json            — отчёт mini-QC;
        _meta.json          — метаданные прогона.
    """
    stem = pdf_path.stem

    (raw_dir / f"{stem}_section_i_text.txt").write_text(
        section_text or "(пусто)", encoding="utf-8"
    )

    if table_text is not None:
        (raw_dir / f"{stem}_table_step1.txt").write_text(table_text, encoding="utf-8")

    (raw_dir / f"{stem}_qc.json").write_text(
        json.dumps(qc_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (raw_dir / f"{stem}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# =============================================================================
# Блок 9. Основной цикл разбора одного PDF
# =============================================================================


def parse_single_pdf_local(
    pdf_path: Path,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    qc_retry: bool = True,
    mode: str = "auto",
    auto_rotate: bool = True,
    scan_dpi: int = DEFAULT_SCAN_DPI,
) -> tuple[Path | None, dict[str, Any]]:
    """
    Полный цикл разбора одного PDF (text-layer или vision-сканы).

    mode:
        auto   — text-layer, если текста достаточно; иначе vision (сканы)
        text   — только text-layer; полный скан → skip
        vision — рендер страниц в PNG + vision LLM (нужна модель с VL)
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")

    mode = (mode or "auto").lower().strip()
    if mode not in ("auto", "text", "vision"):
        raise ValueError(f"Неизвестный mode: {mode!r} (допустимо: auto, text, vision)")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = ensure_raw_dir(output_dir, model)

    pages = get_pdf_page_count(pdf_path)
    eligible, skip_message, skip_reason = check_pdf_eligibility(pdf_path, mode=mode)

    if not eligible and skip_message and skip_reason:
        print(f"[SKIP] {skip_message}")
        status_path = save_skip_status(
            raw_dir=raw_dir,
            pdf_path=pdf_path,
            message=skip_message,
            reason=skip_reason,
            pages=pages,
            model=model,
        )
        print(f"[SKIP] Статус сохранён: {status_path}")
        return None, {
            "status": "skipped",
            "message": skip_message,
            "skip_reason": skip_reason.value,
            "pages": pages,
        }

    print("[EXTRACT] Извлечение text-layer (pdfplumber / pymupdf / pypdf)...")
    extraction = smart_pdf_extract(pdf_path, verbose=True)
    extracted_text = extraction.get("text") or ""
    full_text = extracted_text

    plan = decide_processing_plan(pdf_path, extraction, full_text, mode)
    use_vision = plan["use_vision"]
    is_hybrid = plan["is_hybrid"]
    pipeline = plan["pipeline"]
    scan_page_indices = plan["scan_page_indices"]

    if mode == "text" and is_all_pages_scan(pdf_path):
        raise ValueError(
            f"{pdf_path.name}: файл-скан без text-layer. "
            f"Используйте --mode auto или --mode vision --model qwen2.5vl:7b"
        )

    rotation_angles: list[int] = []
    source_page_numbers: list[int] = []
    image_paths: list[Path] | None = None
    scan_pages_dir: Path | None = None

    if use_vision:
        if not is_vision_capable_model(model):
            raise ValueError(MSG_NEED_VISION_MODEL)

        pages_to_render = plan["pages_to_render"]
        if is_hybrid:
            print(
                f"[HYBRID] Скан-страницы PDF: {[i + 1 for i in scan_page_indices]}; "
                f"text-layer на остальных ({pages - len(scan_page_indices)} стр.)"
            )
        else:
            print(
                f"[SCAN] Vision-ветка: dpi={scan_dpi}, "
                f"auto_rotate={'да' if auto_rotate else 'нет'}, страниц: {len(pages_to_render)}"
            )

        scan_pages_dir = raw_dir / f"{pdf_path.stem}_scan_pages"
        if scan_pages_dir.exists():
            shutil.rmtree(scan_pages_dir)

        image_paths, rotation_angles, source_page_numbers = pdf_to_images(
            pdf_path,
            scan_pages_dir,
            dpi=scan_dpi,
            auto_rotate=auto_rotate,
            page_indices=pages_to_render,
        )
        write_scan_pages_meta(
            scan_pages_dir, image_paths, rotation_angles, source_page_numbers
        )
        print(f"[SCAN] PNG страниц: {scan_pages_dir}")

        if rotation_angles and any(a != 0 for a in rotation_angles):
            rotated = [
                f"p{pg}:{ang}°"
                for pg, ang in zip(source_page_numbers, rotation_angles)
                if ang
            ]
            print(f"[SCAN] Авто-поворот страниц: {', '.join(rotated)}")

        scan_ocr = ocr_images_to_text(image_paths)
        if is_hybrid:
            full_text = merge_scan_text_with_extracted(scan_ocr, extracted_text)
            if scan_ocr.strip():
                print(
                    f"[HYBRID] OCR скан-страниц: {len(scan_ocr)} симв.; "
                    f"итого с text-layer: {len(full_text)} симв."
                )
            else:
                print(
                    f"[HYBRID] text-layer: {len(extracted_text)} симв.; "
                    "OCR скан-страниц недоступен — компания с картинок через vision"
                )
        elif scan_ocr.strip():
            full_text = scan_ocr
            print(f"[SCAN] Вспомогательный OCR-текст: {len(full_text)} символов")
        else:
            print("[SCAN] OCR недоступен (pytesseract) — только vision LLM")

        section_text = extract_section_i_text(full_text) if full_text.strip() else ""
    else:
        if len(full_text.strip()) < 50:
            raise ValueError(
                f"{pdf_path.name}: недостаточно текста ({len(full_text.strip())} симв.). "
                "Попробуйте --mode auto для гибридных PDF (скан + text-layer)."
            )
        section_text = extract_section_i_text(full_text)

    print(f"[SECTION] Раздел I: {len(section_text)} символов")

    expected_rows = count_affiliate_data_rows(section_text) if section_text else None
    if expected_rows:
        print(f"[TABLE] Детерминированный подсчёт строк: {expected_rows}")

    print(f"[LLM] Модель: {model} (локальная Ollama), pipeline: {pipeline}")
    result, table_text = extract_affiliates_two_step(
        section_text=section_text,
        model=model,
        image_paths=image_paths,
    )

    result = normalize_result(
        result=result,
        pdf_path=pdf_path,
        section_text=section_text,
        full_text=full_text,
        cover_images=image_paths,
        model=model,
    )

    if isinstance(result.get("affiliates"), list):
        result["affiliates"] = finalize_affiliates_list(
            result["affiliates"],
            section_text=section_text,
            table_text=table_text,
            full_text=full_text,
        )

    qc_report = run_mini_qc(
        section_text=section_text,
        result=result,
        table_text=table_text,
        is_scan=use_vision,
    )

    if qc_retry and should_run_qc_retry(qc_report):
        print("[QC] Повторное извлечение (строки и/или affiliation_basis)...")
        try:
            result = extract_affiliates_with_qc_retry(
                section_text=section_text,
                model=model,
                qc_report=qc_report,
                table_text=table_text,
            )
            result = normalize_result(
                result=result,
                pdf_path=pdf_path,
                section_text=section_text,
                full_text=full_text,
                cover_images=image_paths,
                model=model,
            )
            if isinstance(result.get("affiliates"), list):
                result["affiliates"] = finalize_affiliates_list(
                    result["affiliates"],
                    section_text=section_text,
                    table_text=table_text,
                    full_text=full_text,
                )
            qc_retry_report = run_mini_qc(
                section_text=section_text,
                result=result,
                table_text=table_text,
                is_scan=use_vision,
            )
            qc_report["retry"] = qc_retry_report
            qc_report["status"] = qc_retry_report.get("status")
            qc_report["expected_rows"] = qc_retry_report.get("expected_rows")
            qc_report["actual_rows"] = qc_retry_report.get("actual_rows")
            qc_report["checks"] = qc_retry_report.get("checks")
            qc_report["basis_issues"] = qc_retry_report.get("basis_issues")
            qc_report["final_note"] = (
                "После QC-retry: " + str(qc_retry_report.get("final_note", ""))
            )
        except Exception as exc:
            qc_report["retry_error"] = str(exc)
            qc_report["final_note"] = f"QC-retry не удался: {exc}"

    print_qc_summary(qc_report)

    result = attach_review_status_to_result(result, qc_report)
    if result.get("status_message") == MSG_MANUAL_REVIEW:
        print(f"[MANUAL] {MSG_MANUAL_REVIEW}")
        if result.get("review_reason"):
            print(f"[MANUAL] {result.get('review_reason')}")

    out_file = output_dir / f"{pdf_path.stem}.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    meta: dict[str, Any] = {
        "source_file": str(pdf_path),
        "pages": pages,
        "model": model,
        "mode": mode,
        "pipeline": pipeline,
        "is_hybrid": is_hybrid,
        "is_scan": use_vision,
        "scan_page_indices": [i + 1 for i in scan_page_indices],
        "vision_page_numbers": source_page_numbers if use_vision else None,
        "scan_dpi": scan_dpi if use_vision else None,
        "auto_rotate": auto_rotate if use_vision else None,
        "rotation_angles": rotation_angles if use_vision else None,
        "scan_pages_dir": str(scan_pages_dir) if scan_pages_dir else None,
        "raw_subdir": raw_subdir_for_model(model),
        "created_at_unix": int(time.time()),
        "extraction_method": extraction.get("method"),
        "extraction_text_length": extraction.get("text_length"),
        "has_text_layer": extraction.get("has_text_layer"),
        "deterministic_table_rows": expected_rows,
        "qc_status": qc_report.get("status"),
        "expected_rows": qc_report.get("expected_rows"),
        "actual_rows": qc_report.get("actual_rows"),
        "manual_review": qc_report.get("manual_review"),
        "manual_review_message": qc_report.get("manual_review_message"),
        "row_count_mismatch": qc_report.get("row_count_mismatch"),
        "note": (
            "Локальная Ollama + детерминированный разбор affiliate_table_parse.py"
        ),
    }
    save_raw_intermediates(
        raw_dir=raw_dir,
        pdf_path=pdf_path,
        section_text=section_text,
        table_text=table_text,
        qc_report=qc_report,
        meta=meta,
    )
    print(f"[RAW] Промежуточные файлы: {raw_dir}")

    qc_report["pipeline"] = pipeline
    qc_report["is_scan"] = use_vision
    qc_report["is_hybrid"] = is_hybrid
    return out_file, qc_report


# =============================================================================
# Блок 10. CLI — пакетная обработка PDF
# =============================================================================


def iter_pdf_files(input_dir: Path) -> list[Path]:
    """Список PDF-файлов в каталоге (без рекурсии)."""
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def main() -> None:
    """Точка входа: argparse + цикл по PDF-файлам."""
    parser = argparse.ArgumentParser(
        description=(
            "PDF (<20 стр.) → локальная Ollama → affiliates JSON "
            "+ промежуточные файлы в raw_<модель>. "
            "Сканы: --mode vision (qwen2.5vl)."
        )
    )
    parser.add_argument("--file", required=False, help="Путь к одному PDF")
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Папка с PDF (если --file не задан)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Локальная модель Ollama (по умолчанию: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        default="output_local_llm",
        help="Каталог для JSON и подкаталога raw_<модель>",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "text", "vision"],
        default="auto",
        help=(
            "auto — text-layer или vision для сканов; "
            "text — только text-layer; vision — рендер страниц в PNG"
        ),
    )
    parser.add_argument(
        "--scan-dpi",
        type=int,
        default=DEFAULT_SCAN_DPI,
        help=f"DPI рендера страниц для vision (по умолчанию: {DEFAULT_SCAN_DPI})",
    )
    parser.add_argument(
        "--no-auto-rotate",
        action="store_true",
        help="Не подбирать поворот 0/90/180/270 для сканов",
    )
    parser.add_argument(
        "--no-qc-retry",
        action="store_true",
        help="Не делать автоматический повтор при FAIL mini-QC",
    )
    args = parser.parse_args()

    # --- Список файлов ---
    if args.file:
        files = [Path(args.file).resolve()]
    else:
        input_dir = Path(args.input_dir).resolve()
        if not input_dir.exists() or not input_dir.is_dir():
            raise SystemExit(f"Папка не найдена: {input_dir}")
        files = iter_pdf_files(input_dir)

    if not files:
        raise SystemExit("PDF-файлы не найдены")

    output_dir = Path(args.output_dir).resolve()
    started = time.perf_counter()

    processed = 0
    processed_scans = 0
    skipped_pages = 0
    skipped_scans = 0
    qc_pass = 0
    qc_fail = 0
    manual_review = 0
    errors = 0

    for pdf_path in files:
        print(f"\n========== FILE: {pdf_path.name} ==========")
        try:
            out, report = parse_single_pdf_local(
                pdf_path=pdf_path,
                output_dir=output_dir,
                model=args.model,
                qc_retry=not args.no_qc_retry,
                mode=args.mode,
                auto_rotate=not args.no_auto_rotate,
                scan_dpi=args.scan_dpi,
            )

            if out is None:
                reason = report.get("skip_reason")
                if reason == SkipReason.TOO_MANY_PAGES.value:
                    skipped_pages += 1
                elif reason == SkipReason.ALL_PAGES_SCAN.value:
                    skipped_scans += 1
                continue

            print(f"[OK] JSON сохранён: {out}")
            processed += 1
            if report.get("is_scan") or report.get("pipeline") == "vision-scan":
                processed_scans += 1
            if report.get("status") == "PASS":
                qc_pass += 1
            elif report.get("status") == "FAIL":
                qc_fail += 1
            if report.get("manual_review") or report.get("row_count_mismatch"):
                manual_review += 1

        except Exception as exc:
            errors += 1
            print(f"[ERROR] {pdf_path.name}: {exc}")

    # --- Итоговая статистика ---
    elapsed = int(time.perf_counter() - started)
    minutes, seconds = divmod(elapsed, 60)
    raw_name = raw_subdir_for_model(args.model)
    print(f"\n[STATS] Обработано: {processed} файлов (из них сканов vision: {processed_scans})")
    print(f"[STATS] Пропущено (≥{MAX_PAGES} стр.): {skipped_pages} — «{MSG_TOO_MANY_PAGES}»")
    if args.mode == "text":
        print(f"[STATS] Пропущено (сканы, mode=text): {skipped_scans} — «{MSG_ALL_SCAN}»")
    print(f"[STATS] QC PASS: {qc_pass}, QC FAIL: {qc_fail}")
    print(f"[STATS] {MSG_MANUAL_REVIEW}: {manual_review}")
    print(f"[STATS] Ошибки: {errors}")
    print(f"[STATS] Время: {minutes} мин {seconds} сек")
    print(f"[STATS] Промежуточные файлы: {output_dir / raw_name}")


if __name__ == "__main__":
    main()
