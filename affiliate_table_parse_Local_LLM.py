"""
affiliate_table_parse_Local_LLM.py
==================================

ПОЛНЫЙ ЛОКАЛЬНЫЙ ПАЙПЛАЙН: PDF со списком аффилированных лиц → JSON.

Файл самодостаточный: детерминированный разбор pipe-таблиц (Block A) и
LLM-ветка через Ollama (блоки 0–11). Отдельный import affiliate_table_parse.py
не нужен.

══════════════════════════════════════════════════════════════════════════════
КАК УСТРОЕН ПАЙПЛАЙН (сверху вниз)
══════════════════════════════════════════════════════════════════════════════

1) ВХОД (CLI main)
   --file / --input-dir, --model (minicpm-v4.5 / qwen2.5vl / auto),
   --mode auto|text|vision, --scan-dpi, флаги QC.
   PDF должен быть < 20 страниц.

2) КЛАССИФИКАЦИЯ (decide_processing_plan)
   text-layer | scan | hybrid (часто: 1-я страница скан титула, дальше текст).

3) ИЗВЛЕЧЕНИЕ ТЕКСТА РАЗДЕЛА I
   section_i_text + таблицы; для сканов — PNG + vision LLM.

4) LLM ДВА ШАГА
   Step1 — pipe-таблица строк (главный качественный источник строк).
   Step2 — каркас JSON (company, report_date, affiliates).

5) ДЕТЕРМИНИРОВАННЫЙ POST-PROCESS
   parse step1 + parse section → choose_merged (чистый step1 > грязный PDF) →
   orphan-склейка обрывов страниц → normalize → QC → JSON.
   Даты и основания — КАК В ИСТОЧНИКЕ (повторы дат не схлопывать).

6) ВЫХОД
   output_local_llm/{stem}.json и raw_<модель>/{stem}_*.

══════════════════════════════════════════════════════════════════════════════
ПРИНЦИПЫ (не нарушать при правках)
══════════════════════════════════════════════════════════════════════════════

• Не выдумывать ФИО, даты, доли, адреса, основания.
• 3 основания + 1 дата — норма; 2 одинаковые даты на 2 основания — норма.
• Step1 — источник истины по строкам, если валиден.
• Section/PDF — orphan-хвосты и чистое дополнение; грязный OCR не затирает step1.
• Универсальные эвристики обрыва, без хардкода под одного эмитента.

══════════════════════════════════════════════════════════════════════════════
КАРТА БЛОКОВ (как в файле)
══════════════════════════════════════════════════════════════════════════════

  Block A  — pipe-таблицы, merge step1⊕PDF, orphan-склейка, даты/основания
  Блок 0   — константы моделей, лимиты, сообщения оператору
  Блок 1   — промпты Ollama Step1/Step2
  Блок 2   — классификация PDF (text/scan/hybrid), eligibility
  Блок 2b  — рендер PNG, автоповорот, OCR-хелперы
  Блок 3   — text-layer: pdfplumber → fitz → pypdf
  Блок 4   — выделение Раздела I
  Блок 5   — вызовы Ollama (JSON chat, step1, step2, QC-retry)
  Блок 6   — нормализация даты/ОГРН/компании/адресов
  Блок 7   — QC и finalize_affiliates_list
  Блок 8   — сохранение raw-артефактов
  Блок 9   — parse_single_pdf_local (оркестратор файла)
  Блок 10  — CLI main()

Запуск:
    python affiliate_table_parse_Local_LLM.py --file report.pdf
    python affiliate_table_parse_Local_LLM.py --file scan.pdf --mode vision --model minicpm-v4.5:latest
    python affiliate_table_parse_Local_LLM.py --input-dir ./pdfs --mode auto
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
# Block A. Детерминированный разбор pipe-таблиц (step1 / section / PDF)
# =============================================================================
# Цель: pipe/plain текст таблицы → записи (ФИО, col3, основания[], даты[], доли).
# Включает: нарезку строк, page-break склейку, orphan-хвосты, merge step1⊕PDF.
# Даты — как в источнике (повторы не схлопывать на align/pick).
# =============================================================================


BASIS_PHRASE_STARTERS = (
    "Лицо является",
    "Лицо имеет право",
    "Общество имеет право",
    "Лицо принадлежит",
    "Общество, осуществляет",
    "Общество осуществляет",
)

# Начало строки таблицы: «1. |» (текст PDF) или «1 |» (pdfplumber), но не заголовок «1 | 2 |».
ROW_PIPE_START_RE = re.compile(
    r"^(?P<num>\d{1,2})(?:\.\s*\||\s*\|\s*(?!\d\s*\|))"
)
ROW_PLAIN_START_RE = re.compile(r"^(?P<num>\d{1,2})\.\s+(?=[А-ЯA-ZЁ«\"])")
ROW_NUMBER_ONLY_RE = re.compile(r"^(?P<num>\d{1,2})\.\s*$")
# Нумерация колонок: «1 | 2 | …» и «1. | 2 | …» (pdfplumber часто ставит точку после № п/п).
TABLE_COLUMN_HEADER_RE = re.compile(r"^\d{1,2}\.?\s*\|\s*\d+\s*\|")
DATE_SHARE_TAIL_RE = re.compile(
    r"\|\s*(?P<date>\d{2}\.\d{2}\.\d{4}[^|]*)\s*\|\s*(?P<share_auth>[^|]*)\s*\|\s*(?P<share_ord>[^|]*)\s*$"
)
ORPHAN_BASIS_CONT_RE = re.compile(
    r"(?ms)^\s*\|\s*\|\s*\|\s*(?P<basis>(?:Совета\s+директоров|исполнительным|органом)[^|\n]*(?:\n[^|\n]+)*)\s*\|"
)
PLAIN_BASIS_CONT_RE = re.compile(
    r"(?ms)Совета\s+директоров[\s\S]{0,60}?Общества\."
)
SECTION_HEADER_RE = re.compile(
    r"^(?:Физические|Юридические)\s+лица",
    re.IGNORECASE,
)
BASIS_GARBAGE_RE = re.compile(
    r"---\s*Таблица|Коды эмитента|№\s*п/п|на странице\s*\d|"
    r"\|\s*ИНН\s*\||\|\s*ОГРН\s*\||\d\s*\|\s*2\s*\|\s*3\s*\|\s*4\s*\|\s*5\s*\|\s*6\s*\|\s*7|"
    r"Полное фирменное наименование|Доля участия аффилированного|"
    r"фамилия,?\s*имя|отчество\s+аффилированного|для\s+некоммерческой|"
    r"организации\)|наименование\s+\(наименование|место\s+нахождения\s+юридического|"
    r"дата\s+наступления\s+основания|доля\s+принадлежащих",
    re.IGNORECASE,
)
COUNCIL_CONTINUATION_RE = re.compile(
    r"Совета\s+директоров(?:\s+Общества)?\.?",
    re.IGNORECASE,
)
INCOMPLETE_BASIS_END_RE = re.compile(
    r"(?:членом|органом|директором|группе)\s*\.?\s*$",
    re.IGNORECASE,
)
# Склейки оснований при кривом pipe/OCR (два статуса в одной фразе).
BASIS_MALFORMED_RE = (
    re.compile(r"членом\s+единоличным", re.IGNORECASE),
    re.compile(r"является\s+членом\s+единоличным", re.IGNORECASE),
    re.compile(r"членом\s+исполнительным", re.IGNORECASE),
    re.compile(r"^принадлежит\s+общество\s*\.?$", re.IGNORECASE),
)
# Все даты «ДД.ММ.ГГГГ» в ячейке (порядок и повторы сохраняются).
INN_COL3_RE = re.compile(r"\b\d{12}\b")
OGRN_COL3_RE = re.compile(r"\b\d{13}(?:\d{2})?\b")
CONSENT_COL3_RE = re.compile(
    r"Согласие\s+физического\s+лица\s+не\s+получено",
    re.IGNORECASE,
)
COLUMN3_INN_OGRN_HEADER_RE = re.compile(
    r"инн\s+физического|огрн\s+юридического",
    re.IGNORECASE,
)
COLUMN3_ADDRESS_HEADER_RE = re.compile(
    r"место\s+нахождения|место\s+жительства",
    re.IGNORECASE,
)
POSTAL_ADDRESS_RE = re.compile(
    r"\b\d{6}\b.*(?:область|край|город|г\.|ул\.|улица|проспект|дом)",
    re.IGNORECASE,
)
BASIS_DATE_PATTERN = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")
PAGE_OR_TABLE_BOUNDARY_RE = re.compile(
    r"^---\s*(?:Страница|Таблица)\b|===== ТАБЛИЦЫ",
    re.IGNORECASE,
)
TABLE_HEADER_LABEL_RE = re.compile(
    r"^(?:№\s*п/п|полное\s+фирменное|место\s+нахождения|основание\s*\(|"
    r"дата\s+наступления|доля\s+участия|доля\s+принадлежащих)",
    re.IGNORECASE,
)
CROSS_PAGE_CONTINUATION_START_RE = re.compile(
    r"^(?:составляющие\s+устав|лицо\s+|общество[,]?\s+осуществляет|"
    r"общество\s+имеет|является\s+членом|принадлежит\s+к\s+той)",
    re.IGNORECASE,
)
# Orphan-хвост после шапки: «составляющие … данного лица» + доп. основания + дата.
# Стоп только на реальном начале следующей строки п/п — НЕ на датах вида «01.12.2012».
ORPHAN_SHARES_TAIL_BLOCK_RE = re.compile(
    r"составляющие\s+уставный\s+капитал\s+данного\s+лица;?"
    r"(?P<body>[\s\S]{0,900}?)"
    r"(?=\n\s*\d{1,2}\.\s+(?=[А-ЯA-ZЁ«\"])"
    r"|\n\s*\d{1,2}\.\s*\|\s*"
    r"|\n\s*\d{1,2}\s*\|\s*"
    r"|\n\s*---\s*(?:Страница|Таблица)"
    r"|\Z)",
    re.IGNORECASE,
)
# Orphan «Совета директоров Общества» (дата может быть до или после «Общества»).
ORPHAN_COUNCIL_TAIL_RE = re.compile(
    r"Совета\s+директоров\s+"
    r"(?:(?P<date1>\d{2}\.\d{2}\.\d{4})[\s\S]{0,80}?)?"
    r"Общества\.?\s*"
    r"(?:[^\d\n]{0,40})?"
    r"(?P<date2>\d{2}\.\d{2}\.\d{4})?",
    re.IGNORECASE,
)
# Конец следующей строки п/п (не дата DD.MM.YYYY).
# Учитываем «12. Название», «12. | Название», «12 | Название».
_NEXT_ROW_START_RE = re.compile(
    r"\n\s*\d{1,2}\.\s+(?=[А-ЯA-ZЁ«\"])"
    r"|\n\s*\d{1,2}\.\s*\|\s*"
    r"|\n\s*\d{1,2}\s*\|\s*",
)


def _flatten(text: str) -> str:
    """
    Сжать пробелы/переносы в одну строку и обрезать края.

    Используется везде, где сравниваем ФИО, основания, адреса.
    """
    return re.sub(r"\s+", " ", text or "").strip()


def detect_column3_mode(section_text: str) -> str:
    """
    Определить тип колонки 3 таблицы аффилированных лиц.

    Возвращает:
      • «inn_ogrn» — ОГРН/ИНН или «Согласие физического лица не получено»
        (типично отчёты только с физлицами);
      • «address» — место нахождения / жительства.

    Смотрит шапку section_text и наличие маркеров согласия.
    """
    head = (section_text or "").lower()
    has_inn_col = "инн физического" in head or "огрн юридического" in head
    has_address_col = "место нахождения" in head or "место жительства" in head
    has_consent = "согласие физического лица" in head

    if has_inn_col and not has_address_col:
        return "inn_ogrn"
    if has_consent and not has_address_col:
        return "inn_ogrn"
    if has_address_col and not has_inn_col:
        return "address"
    if has_inn_col and has_address_col:
        inn_pos = head.find("инн физического")
        addr_pos = head.find("место нахождения")
        if inn_pos >= 0 and (addr_pos < 0 or inn_pos < addr_pos):
            return "inn_ogrn"
    if CONSENT_COL3_RE.search(section_text or ""):
        return "inn_ogrn"
    return "address"


def table_has_mixed_affiliate_types(section_text: str) -> bool:
    """True, если в тексте есть подразделы и физлиц, и юрлиц."""
    head = (section_text or "").lower()
    return "физические лица" in head and "юридические лица" in head


def addresses_are_similar(left: str | None, right: str | None) -> bool:
    """Сравнение адресов с допуском на обрезку и разный пробел."""
    a = _flatten(left or "").lower()
    b = _flatten(right or "").lower()
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 18 and shorter in longer


def _title_page_text(text: str) -> str:
    """Текст титула до «Раздел I» / «Состав аффилированных лиц» (не таблица)."""
    if not text:
        return ""
    lower = text.lower()
    markers = (
        "раздел i",
        "состав аффилированных лиц на",
        "состав аффилированных лиц",
    )
    cut = len(text)
    for marker in markers:
        pos = lower.find(marker)
        if pos > 0:
            cut = min(cut, pos)
    return text[:cut] if cut < len(text) else text[:4000]


def extract_emitter_address_candidates(text: str) -> list[str]:
    """Кандидаты адреса эмитента с титула — чтобы не подставить их в строки таблицы."""
    title = _title_page_text(text)
    if not title:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(raw: str) -> None:
        value = _flatten(raw)
        if len(value) < 12:
            return
        key = value.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(value)

    patterns = (
        r"(?:место\s+нахождения(?:\s+юридического\s+лица)?|"
        r"адрес(?:\s+юридического\s+лица)?(?:\s+эмитента)?)"
        r"[^\n:]{0,40}:?\s*([^\n(]{12,220})",
        r"(?:место\s+нахождения|адрес)[^\n]{0,80}\n\s*(\d{6}[^\n(]{10,220})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, title, flags=re.IGNORECASE):
            add_candidate(match.group(1))

    return candidates


def is_emitter_address_substitution(
    address: str | None,
    emitter_candidates: list[str],
) -> bool:
    """True, если address строки совпадает с адресом эмитента (подстановка LLM)."""
    if not address:
        return False
    return any(addresses_are_similar(address, candidate) for candidate in emitter_candidates)


def looks_like_postal_address(text: str | None) -> bool:
    """Грубая эвристика: строка похожа на почтовый адрес."""
    if not text:
        return False
    stripped = _flatten(text)
    if CONSENT_COL3_RE.search(stripped):
        return False
    digits = re.sub(r"\D", "", stripped)
    if len(digits) in (10, 12):
        return False
    return bool(POSTAL_ADDRESS_RE.search(stripped))


def extract_column3_from_row_blob(blob: str, mode: str = "address") -> str:
    """Вытащить колонку 3 (адрес или ИНН/ОГРН/согласие) из сырого blob строки."""
    text = _flatten(blob)
    if not text:
        return ""

    consent = CONSENT_COL3_RE.search(text)
    if consent:
        return _flatten(consent.group(0))

    if mode == "inn_ogrn" or "согласие" in text.lower():
        if "физического" in text.lower() and "согласие" in text.lower():
            return "Согласие физического лица не получено"

    inn = INN_COL3_RE.search(text)
    if inn:
        return inn.group(0)

    ogrn = OGRN_COL3_RE.search(text)
    if ogrn and len(ogrn.group(0)) in (13, 15):
        return ogrn.group(0)

    return ""


def _flatten_multiline_row(lines: list[str]) -> str:
    """Склеить многострочный chunk pipe-строки в одну плоскую строку."""
    return _flatten("\n".join(ln.strip() for ln in lines if ln.strip()))


def _split_pipe_row_columns(flat_row: str) -> tuple[int | None, list[str]]:
    """Разобрать pipe-строку на номер п/п и список ячеек."""
    match = re.match(r"^(?P<num>\d{1,2})\s*\.?\s*\|\s*(?P<rest>.+)$", flat_row.strip())
    if not match:
        return None, []
    parts = [p.strip() for p in match.group("rest").split("|")]
    return int(match.group("num")), parts


def _is_column_numbering_pipe_parts(pipe_parts: list[str]) -> bool:
    """True, если ячейки — нумерация колонок «1|2|3|…», а не данные."""
    cells = [p.strip() for p in pipe_parts if p.strip()]
    if len(cells) < 4:
        return False
    if not all(re.fullmatch(r"\d{1,2}", c) for c in cells):
        return False
    nums = [int(c) for c in cells]
    if nums[0] > 3:
        return False
    return all(nums[i] + 1 == nums[i + 1] for i in range(len(nums) - 1))


def is_table_column_numbering_row(
    flat_row: str,
    pipe_parts: list[str] | None = None,
) -> bool:
    """Строка-заголовок нумерации колонок (пропускаем при разборе данных)."""
    stripped = (flat_row or "").strip()
    if TABLE_COLUMN_HEADER_RE.match(stripped):
        return True
    if pipe_parts is None:
        _, pipe_parts = _split_pipe_row_columns(stripped)
    return _is_column_numbering_pipe_parts(pipe_parts)


def _normalize_basis_ocr(text: str) -> str:
    """Лёгкая нормализация OCR в тексте основания (ё/е и т.п.)."""
    return re.sub(r"\bЛифо\b", "Лицо", text or "", flags=re.IGNORECASE)


def is_malformed_basis_phrase(phrase: str) -> bool:
    """Битая склейка двух статусов в одной фразе («членом единоличным…»)."""
    text = _flatten(_normalize_basis_ocr(phrase))
    if not text:
        return True
    lower = text.lower()
    for pat in BASIS_MALFORMED_RE:
        if pat.search(lower):
            return True
    if "единоличным исполнительным" in lower and "совета директоров" in lower:
        return True
    if len(re.findall(r"лицо\s+является", lower)) > 1:
        return True
    return False


def _basis_phrase_looks_polluted(phrase: str) -> bool:
    """
    Основание похоже на склейку соседних строк OCR.

    Признаки: «Согласие/не получено», чужие ФИО, ОГРН/ИНН, «0 0», слишком длинный текст.
    Такие фразы нельзя подмешивать в чистый step1.
    """
    text = _flatten(_normalize_basis_ocr(phrase))
    if not text:
        return True
    if is_malformed_basis_phrase(text) or BASIS_GARBAGE_RE.search(text):
        return True
    lower = text.lower()
    if re.search(r"согласие|не\s+получено", lower):
        return True
    # В основание вклеены ФИО / номер п/п / доли.
    if re.search(
        r"членом\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+|"
        r"групп[еы]\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+|"
        r"\b\d{1,2}\s+\d{2}\.\d{2}\.\d{4}\b|"
        r"\b0\s+0\b",
        text,
    ):
        return True
    if re.search(r"\b\d{12,15}\b", text):
        return True
    if len(text) > 280:
        return True
    return False


def _clean_basis_list(phrases: list[str], *, keep_truncated: bool = True) -> list[str]:
    """Отфильтровать список оснований: убрать polluted, оставить валидные/truncated."""
    out: list[str] = []
    for p in phrases:
        if _basis_phrase_looks_polluted(p):
            continue
        if is_valid_basis_phrase(p) or (keep_truncated and is_truncated_basis_phrase(p)):
            out.append(p)
    return out


def _basis_needs_council_continuation(last: str) -> bool:
    """True, если фраза оборвана на «…является членом» и ждёт «Совета директоров…»."""
    t = _flatten(_normalize_basis_ocr(last)).lower()
    if "совета" in t or "директор" in t:
        return False
    if "единоличным" in t or "исполнительным органом" in t:
        return False
    if "принадлежит" in t and "группе" in t:
        return False
    return bool(re.search(r"(?:является\s+)?членом\s*\.?\s*$", t))


def _basis_needs_society_suffix(last: str) -> bool:
    """True, если есть «Совета директоров» без «Общества»/«эмитента» (обрыв строки)."""
    t = _flatten(_normalize_basis_ocr(last)).lower()
    if not t:
        return False
    if "общества" in t or "эмитента" in t:
        return False
    return bool(re.search(r"совета\s+директоров\s*\.?\s*$", t))


def _name_missing_closing_quote(name: str | None) -> bool:
    """True, если в наименовании открыта «кавычка» без закрывающей."""
    n = str(name or "")
    return ("«" in n and "»" not in n) or bool(
        re.search(r'["«][^»"]+$', n)
    )


def _name_looks_polluted(name: str | None) -> bool:
    """Наименование содержит чужие колонки (даты, доли, текст оснований) — отбраковать."""
    n = _flatten(name or "")
    if not n:
        return True
    lower = n.lower()
    if BASIS_DATE_PATTERN.search(n):
        return True
    if re.search(r"\d+\s*%\s*-?\s*\d*\s*акци", lower):
        return True
    if re.search(
        r"лицо\s+имеет\s+право|общество\s+имеет\s+право|"
        r"принадлежит\s+к\s+той|является\s+членом|"
        r"распоряжаться\s+более",
        lower,
    ):
        return True
    # Слишком много фрагментов «как из нескольких колонок».
    if len(n) > 120 and n.count(",") >= 2 and ("г." in lower or "%" in n):
        return True
    return False


def _prefer_full_name(primary: str | None, secondary: str | None) -> str | None:
    """Выбрать более полное чистое наименование из двух источников (step1 vs PDF)."""
    a = _flatten(primary or "")
    b = _flatten(secondary or "")
    a_bad = _name_looks_polluted(a)
    b_bad = _name_looks_polluted(b)
    if a_bad and not b_bad:
        return b or None
    if b_bad and not a_bad:
        return a or None
    if a_bad and b_bad:
        # Оба плохие — более короткое обычно меньше мусора.
        if not a:
            return b or None
        if not b:
            return a or None
        return a if len(a) <= len(b) else b
    if not a:
        return b or None
    if not b:
        return a or None
    a_open = _name_missing_closing_quote(a)
    b_open = _name_missing_closing_quote(b)
    if a_open and not b_open and len(b) >= len(a) - 2:
        return b
    if b_open and not a_open and len(a) >= len(b) - 2:
        return a
    if "«" in b and "«" not in a and len(b) >= len(a):
        return b
    if "«" in a and "«" not in b and len(a) >= len(b):
        return a
    return a if len(a) >= len(b) else b


def _union_basis_phrases(
    primary: list[str],
    secondary: list[str],
) -> list[str]:
    """
    Объединить основания двух источников без дублей.

    Primary важнее: грязная более длинная фраза НЕ затирает чистую короткую.
    """
    out: list[str] = []
    seen: list[str] = []

    def _add(phrase: str, *, allow_polluted: bool = False) -> None:
        if _basis_phrase_looks_polluted(phrase) and not allow_polluted:
            return
        if not (is_valid_basis_phrase(phrase) or is_truncated_basis_phrase(phrase)):
            return
        key = _flatten(phrase).lower()
        for i, prev in enumerate(seen):
            if key == prev:
                return
            if key in prev or prev in key:
                prev_polluted = _basis_phrase_looks_polluted(out[i])
                cur_polluted = _basis_phrase_looks_polluted(phrase)
                # Чистая формулировка побеждает грязное «расширение».
                if prev_polluted and not cur_polluted:
                    out[i] = phrase
                    seen[i] = key
                elif not prev_polluted and cur_polluted:
                    return
                elif len(key) > len(prev) and not cur_polluted:
                    out[i] = phrase
                    seen[i] = key
                return
        out.append(phrase)
        seen.append(key)

    for p in primary:
        _add(p)
    for p in secondary:
        _add(p)
    return out


def is_share_placeholder(value: Any) -> bool:
    """True для пустой доли или «—» / «---»."""
    if value is None:
        return True
    text = str(value).strip()
    if not text or text.lower() == "null":
        return True
    return bool(re.fullmatch(r"[-—–]+", text))


def is_meaningful_share(value: Any) -> bool:
    """True, если доля выглядит как реальный процент/акции, а не плейсхолдер."""
    if is_share_placeholder(value):
        return False
    text = str(value).strip().lower()
    if re.search(r"%|акци", text):
        return True
    cleaned = text.replace(",", ".")
    if re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        try:
            return float(cleaned) > 7
        except ValueError:
            return False
    return len(text) > 2


def is_valid_basis_phrase(phrase: str) -> bool:
    """Юридически похожая полная фраза основания (старт с «Лицо/Общество…»)."""
    text = _normalize_basis_ocr(_flatten(phrase))
    if len(text) < 12 or len(text) > 420:
        return False
    if is_malformed_basis_phrase(text):
        return False
    lower = text.lower()
    if re.search(r"является\s+членом\s*\.?\s*$", lower) and "совета" not in lower:
        return False
    if BASIS_GARBAGE_RE.search(text):
        return False
    if text.count("|") >= 2:
        return False
    # Хвост шапки таблицы / обрывки колонок.
    if re.search(
        r"фамилия|отчество|некоммерческ|организации\)|наименован\s*\(|"
        r"доля\s+юридического|место\s+жительства|указывается\s+только|"
        r"аффилированным\s*\(|основания\s*\(оснований\)|акционерного\s+общества,\s*%|"
        r"приморский\s+край|г\.\s*владивосток",
        lower,
    ):
        # «г. Владивосток» в основании — мусор склейки адреса, не основание.
        if not lower.startswith(("лицо принадлежит к той группе", "общество")):
            return False
        if "доля" in lower or "место жительства" in lower or "указывается" in lower:
            return False
        if "владивосток" in lower or "приморский" in lower:
            return False
    if not lower.startswith(
        ("лицо", "лифо", "общество", "юридическое лицо", "на основании")
    ):
        return False
    return True


def is_truncated_basis_phrase(phrase: str) -> bool:
    """Основание оборвано и ждёт хвост со следующей страницы."""
    text = _normalize_basis_ocr(_flatten(phrase))
    if not text:
        return False
    lower = text.lower()
    if text.rstrip().endswith(","):
        return True
    if _basis_needs_council_continuation(text):
        return True
    if re.search(r"приходящихся\s+на\s+акци[\w]*\s*,?\s*$", lower):
        return True
    return False


def extract_basis_dates(cell_text: str) -> list[str]:
    """Все даты ДД.ММ.ГГГГ из текста ячейки; порядок и ПОВТОРЫ сохраняются."""
    if not cell_text:
        return []
    cleaned = str(cell_text).replace("г.", "").replace("Г.", "")
    return BASIS_DATE_PATTERN.findall(cleaned)


def _is_date_only_cell(text: str) -> bool:
    """Ячейка содержит только даты и разделители."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    without_dates = BASIS_DATE_PATTERN.sub("", stripped)
    without_dates = re.sub(r"[\s;,\-—]+", "", without_dates)
    return not without_dates


def normalize_share_pct(value: Any) -> str | None:
    """Доля как строка числа без округления; мусор шапки → None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    if BASIS_GARBAGE_RE.search(text) or re.search(
        r"[А-Яа-яA-Za-z]{4,}", text
    ) and not re.search(r"акци", text, re.IGNORECASE):
        if not re.fullmatch(r"[-—–]+", text) and not re.match(r"\d", text):
            return None
        if re.search(r"фамилия|наименован|организац|отчество|доля", text, re.I):
            return None
    if re.fullmatch(r"[-—–]+", text):
        return text
    match = re.match(r"(\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", ".")
    return text


def _unique_dates_ordered(dates: list[str]) -> list[str]:
    """Уникальные даты с сохранением порядка (для дедупа при склейке источников)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for d in dates:
        key = str(d).strip()
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _dates_as_found(dates: list[str]) -> list[str]:
    """Даты как в источнике: порядок и повторы сохраняются (без unique)."""
    return [str(d).strip() for d in dates if d and str(d).strip()]


def align_basis_dates_to_bases(
    affiliation_basis: list[str],
    basis_dates: list[str],
) -> list[str]:
    """
    Финальный список дат «как есть»: без растягивания и без схлопывания повторов.

    affiliation_basis не используется — не домысливаем даты по числу оснований.
    """
    del affiliation_basis  # не домысливаем даты по числу оснований
    return _dates_as_found(basis_dates)


def pick_basis_dates_for_row(
    affiliation_basis: list[str],
    *,
    existing: list[str] | None = None,
    table_dates: list[str] | None = None,
    section_dates: list[str] | None = None,
    step1_dates: list[str] | None = None,
) -> list[str]:
    """
    Взять даты из первого непустого источника: step1 → table → JSON → section.

    Источник берётся ЦЕЛИКОМ (с повторами), источники не склеиваются.
    """
    del affiliation_basis
    for dates in (
        list(step1_dates or []),
        list(table_dates or []),
        list(existing or []),
        list(section_dates or []),
    ):
        found = _dates_as_found(dates)
        if found:
            return found
    return []


def _clean_basis_text_for_split(text: str) -> str:
    """Убрать из текста оснований хвосты колонок дат/долей, попавшие при pipe-разборе."""
    cleaned_lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Строка целиком «21.06.2024 | 0 | 0» (даты и доли без текста основания).
        if re.fullmatch(
            r"\d{2}\.\d{2}\.\d{4}(?:\s*\|\s*[^|]*){0,2}",
            line,
        ):
            continue
        # «... общества | 08.10.2020» или «... | 08.10.2020 | 0 | 0»
        line = re.sub(r"\|\s*\d{2}\.\d{2}\.\d{4}(?:\s*\|\s*[^|]*)?$", "", line)
        # «... общества 21.06.2024 | 0 | 0» (дата без ведущего |)
        line = re.sub(r"\s+\d{2}\.\d{2}\.\d{4}\s*\|\s*[^|]*(?:\s*\|\s*[^|]*)?$", "", line)
        line = line.strip().strip("|").strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _normalize_basis_fragment(phrase: str) -> str:
    """Нормализация фрагмента перед split_basis_cell."""
    p = _normalize_basis_ocr(_flatten(phrase)).strip().rstrip(";.").strip()
    if not p:
        return ""
    lower = p.lower()
    if lower.startswith(
        ("принадлежит", "является", "имеет ", "осуществляет")
    ) and not lower.startswith(("лицо", "общество", "юридическое")):
        return f"Лицо {p}"
    return p


def split_basis_cell(text: str) -> list[str]:
    """Разрезать ячейку оснований на отдельные фразы (по «;» / типичным стартам)."""
    flat = _normalize_basis_ocr(_flatten(_clean_basis_text_for_split(text)))
    if not flat:
        return []

    # Отрезаем мусор из колонок ФИО/адреса до первого «Лицо/Общество».
    first_basis = re.search(r"(?:Лицо|Общество|Юридическое)", flat, flags=re.IGNORECASE)
    if first_basis:
        flat = flat[first_basis.start() :]

    # Разделение по началу каждой типовой формулировки основания.
    parts = re.split(
        r"(?=(?:(?:Лицо|Общество|Юридическое\s+лицо),?\s+"
        r"(?:является|имеет|принадлежит|осуществляет)|"
        r",?\s*является\s+членом))",
        flat,
        flags=re.IGNORECASE,
    )

    # LLM/склейка иногда объединяют несколько оснований через «; Лицо ...».
    expanded: list[str] = []
    for part in parts:
        subparts = re.split(
            r"\s*;\s*(?=(?:Лицо|Общество|Юридическое))",
            part.strip(),
            flags=re.IGNORECASE,
        )
        for sub in subparts:
            expanded.extend(
                re.split(
                    r",\s*(?=(?:Лицо|Общество|принадлежит|является)\s)",
                    sub.strip(),
                    flags=re.IGNORECASE,
                )
            )

    phrases: list[str] = []
    seen: set[str] = set()
    for part in expanded:
        phrase = _normalize_basis_fragment(part)
        phrase = re.sub(r";\s*Л\.?$", "", phrase).strip()
        if not phrase:
            continue
        # Оборванный хвост «…является членом» сохраняем для доклейки со след. страницы.
        if not is_valid_basis_phrase(phrase) and not is_truncated_basis_phrase(phrase):
            continue
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            phrases.append(phrase)
    return phrases


def expand_affiliation_basis_list(basis_list: list[Any]) -> list[str]:
    """Развернуть массив оснований: склеенные элементы → отдельные фразы."""
    phrases: list[str] = []
    seen: set[str] = set()
    for item in basis_list:
        for phrase in split_basis_cell(str(item)):
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                phrases.append(phrase)
    return phrases


def _parse_row_lines(
    row_num: int,
    lines: list[str],
    column3_mode: str = "address",
) -> dict[str, Any] | None:
    """
    Разобрать chunk строк одной записи таблицы в dict полей.

    Извлекает ФИО, col3, основания, даты (как есть), доли.
    """
    if not lines:
        return None

    flat_row = _flatten_multiline_row(lines)
    _, pipe_parts = _split_pipe_row_columns(flat_row)

    if is_table_column_numbering_row(flat_row, pipe_parts):
        return None

    full_name = ""
    address = ""
    basis_chunks: list[str] = []
    date_parts: list[str] = []
    share_auth: str | None = None
    share_ord: str | None = None

    if len(pipe_parts) >= 1:
        full_name = pipe_parts[0]
    if len(pipe_parts) >= 2:
        address = pipe_parts[1]
    if len(pipe_parts) >= 3:
        basis_chunks.append(pipe_parts[2])
        if len(pipe_parts) >= 4 and _is_date_only_cell(pipe_parts[3]):
            date_parts.extend(extract_basis_dates(pipe_parts[3]))
        if len(pipe_parts) >= 5:
            share_auth = pipe_parts[4].strip() or None
        if len(pipe_parts) >= 6:
            share_ord = pipe_parts[5].strip() or None

    if not address or address in ("-", "---"):
        address = extract_column3_from_row_blob(flat_row, mode=column3_mode)

    # Fallback: старый разбор по первой строке, если склейка не дала колонок.
    if not full_name:
        first_parts = [p.strip() for p in lines[0].split("|")]
        if len(first_parts) >= 2:
            full_name = first_parts[1]
        if len(first_parts) >= 4:
            address = address or first_parts[2]
            basis_chunks.append(first_parts[3])
            if len(first_parts) >= 5 and _is_date_only_cell(first_parts[4]):
                date_parts.extend(extract_basis_dates(first_parts[4]))

    used_plain_layout = False
    if not full_name and not any("|" in ln for ln in lines):
        plain = _parse_plain_multiline_row(lines, column3_mode=column3_mode)
        if plain:
            full_name, address, plain_bases, plain_dates, p_auth, p_ord = plain
            basis_chunks = plain_bases
            date_parts = plain_dates
            share_auth = share_auth or p_auth
            share_ord = share_ord or p_ord
            used_plain_layout = True

    if not used_plain_layout:
        for line in lines[1:]:
            tail = DATE_SHARE_TAIL_RE.search(line)
            if tail:
                before = line[: tail.start()].strip().strip("|").strip()
                if before:
                    basis_chunks.append(before)
                date_parts.extend(extract_basis_dates(tail.group("date")))
                share_auth = tail.group("share_auth").strip()
                share_ord = tail.group("share_ord").strip()
            else:
                cleaned = line.strip()
                if not cleaned:
                    continue
                if _is_repeatable_table_header_line(cleaned):
                    continue
                # Строка целиком «21.06.2024 | 0 | 0».
                if re.fullmatch(
                    r"\d{2}\.\d{2}\.\d{4}(?:\s*\|\s*[^|]*){0,2}",
                    cleaned,
                ):
                    date_parts.extend(extract_basis_dates(cleaned))
                    share_tail = DATE_SHARE_TAIL_RE.search(cleaned)
                    if share_tail:
                        share_auth = share_auth or share_tail.group("share_auth").strip()
                        share_ord = share_ord or share_tail.group("share_ord").strip()
                    continue
                # Даты «Дата наступления основания» часто идут отдельными строками без «|».
                if "|" not in cleaned and _is_date_only_cell(cleaned):
                    date_parts.extend(extract_basis_dates(cleaned))
                    continue
                pipe_parts = [p.strip() for p in cleaned.split("|")]
                # Строка вида «... основание ... | 08.10.2020» (только дата, без долей).
                if len(pipe_parts) == 2 and _is_date_only_cell(pipe_parts[1]):
                    if pipe_parts[0].strip():
                        basis_chunks.append(pipe_parts[0])
                    date_parts.extend(extract_basis_dates(pipe_parts[1]))
                    continue
                # Продолжение ячейки «Дата наступления основания» (колонка 5) на отдельной строке.
                if len(pipe_parts) >= 5 and _is_date_only_cell(pipe_parts[4]):
                    date_parts.extend(extract_basis_dates(pipe_parts[4]))
                    if len(pipe_parts) > 3 and pipe_parts[3].strip():
                        basis_chunks.append(pipe_parts[3])
                    if len(pipe_parts) > 5 and pipe_parts[5].strip():
                        share_auth = share_auth or pipe_parts[5].strip()
                    if len(pipe_parts) > 6 and pipe_parts[6].strip():
                        share_ord = share_ord or pipe_parts[6].strip()
                    continue
                basis_chunks.append(cleaned)

    basis_text = _merge_split_basis_chunks(basis_chunks)
    affiliation_basis = split_basis_cell(basis_text)

    # Даты могут быть разнесены по нескольким строкам строки таблицы.
    row_context = "\n".join(lines)
    row_dates = extract_basis_dates(row_context)
    if len(row_dates) > len(date_parts):
        date_parts = row_dates

    basis_dates = align_basis_dates_to_bases(affiliation_basis, date_parts)

    if not full_name:
        return None

    if address in ("-", "---"):
        address = ""

    return {
        "row_number": row_num,
        "full_name": _flatten(full_name),
        "address": address if address else None,
        "column3_mode": column3_mode,
        "affiliation_basis": affiliation_basis,
        "basis_date": basis_dates,
        "share_authorized_capital_pct": share_auth,
        "share_ordinary_stocks_pct": share_ord,
        "_basis_text": basis_text,
    }


def _merge_split_basis_chunks(chunks: list[str]) -> str:
    """Склеить куски основания, разрезанные переносами строк pipe."""
    merged: list[str] = []
    for raw in chunks:
        chunk = (raw or "").strip()
        if not chunk or _is_repeatable_table_header_line(chunk):
            continue
        if (
            merged
            and merged[-1].rstrip().endswith(",")
            and not chunk.lower().startswith(("лицо ", "общество"))
        ):
            merged[-1] = merged[-1].rstrip() + " " + chunk
            continue
        if (
            merged
            and chunk[0].islower()
            and not CROSS_PAGE_CONTINUATION_START_RE.match(chunk)
            and not chunk.lower().startswith(("лицо ", "общество"))
        ):
            merged[-1] = merged[-1].rstrip() + " " + chunk
            continue
        merged.append(chunk)
    return "\n".join(merged)


def _is_page_or_table_boundary_line(line: str) -> bool:
    """Маркер «--- Страница/Таблица ---» или блок ТАБЛИЦЫ."""
    return bool(PAGE_OR_TABLE_BOUNDARY_RE.search((line or "").strip()))


def _is_repeatable_table_header_line(line: str) -> bool:
    """Повтор шапки таблицы после разрыва страницы (пропускаем)."""
    stripped = (line or "").strip()
    if not stripped:
        return True
    if TABLE_COLUMN_HEADER_RE.match(stripped) or is_table_column_numbering_row(stripped):
        return True
    if TABLE_HEADER_LABEL_RE.match(stripped):
        return True
    if BASIS_GARBAGE_RE.search(stripped) and not CROSS_PAGE_CONTINUATION_START_RE.match(
        stripped
    ):
        return True
    return False


def _match_data_row_start(line: str) -> int | None:
    """Номер п/п в начале строки данных или None."""
    stripped = (line or "").strip()
    match = ROW_PIPE_START_RE.match(stripped)
    if match:
        return int(match.group("num"))
    match = ROW_PLAIN_START_RE.match(stripped)
    if match:
        return int(match.group("num"))
    match = ROW_NUMBER_ONLY_RE.match(stripped)
    if match:
        return int(match.group("num"))
    return None


def _parse_plain_multiline_row(
    lines: list[str],
    column3_mode: str = "address",
) -> tuple[str, str, list[str], list[str], str | None, str | None] | None:
    """Разбор строки без pipe (plain text PDF) в поля записи."""
    work = [
        ln.strip()
        for ln in lines
        if ln.strip() and not _is_repeatable_table_header_line(ln.strip())
    ]
    if not work:
        return None

    idx = 0
    if ROW_NUMBER_ONLY_RE.match(work[0]) or ROW_PLAIN_START_RE.match(work[0]):
        plain_match = ROW_PLAIN_START_RE.match(work[0])
        if plain_match:
            tail = work[0][plain_match.end() :].strip()
            if tail:
                work[0] = tail
            else:
                idx = 1
        else:
            idx = 1

    if idx >= len(work):
        return None

    full_name = work[idx]
    idx += 1
    address = work[idx] if idx < len(work) else ""
    idx += 1

    basis_chunks: list[str] = []
    date_parts: list[str] = []
    share_auth: str | None = None
    share_ord: str | None = None

    while idx < len(work):
        part = work[idx]
        idx += 1
        if _is_date_only_cell(part):
            date_parts.extend(extract_basis_dates(part))
            continue
        if re.fullmatch(r"[-—–]+", part):
            if share_auth is None:
                share_auth = part
            elif share_ord is None:
                share_ord = part
            continue
        basis_chunks.append(part)

    if not full_name:
        return None
    if not address or address in ("-", "---"):
        address = extract_column3_from_row_blob(
            "\n".join(lines), mode=column3_mode
        )
    return full_name, address, basis_chunks, date_parts, share_auth, share_ord


def _row_chunk_looks_incomplete(lines: list[str]) -> bool:
    """Chunk строки выглядит оборванным (нужен хвост со след. страницы)."""
    if not lines:
        return False
    text = _flatten("\n".join(lines))
    if not text:
        return False
    if re.search(
        r"(?:акци[\s,]*|голосов[\s,]*|органом[\s,]*|членом[\s,]*|группе[\s,]*),\s*$",
        text,
        re.IGNORECASE,
    ):
        return True
    last = lines[-1].strip().rstrip("|").strip()
    if last.endswith(","):
        return True
    if re.search(r"приходящихся\s+на\s+акци[\s,]*,?\s*$", text, re.IGNORECASE):
        return True
    # Одно основание без завершающей точки при наличии даты — возможен хвост на след. стр.
    if re.search(r"\d{2}\.\d{2}\.\d{4}", text) and not text.rstrip().endswith((".", ";")):
        tail = text.split("\n")[-1].strip()
        if tail and not tail.endswith((".", ";")) and not _is_date_only_cell(tail):
            if not is_valid_basis_phrase(tail):
                return True
    return False


def _is_cross_page_continuation_line(line: str) -> bool:
    """Строка без номера п/п — продолжение предыдущей записи после page-break."""
    stripped = (line or "").strip()
    if not stripped or _is_page_or_table_boundary_line(stripped):
        return False
    if _match_data_row_start(stripped) is not None:
        return False
    if SECTION_HEADER_RE.match(stripped):
        return False
    if _is_repeatable_table_header_line(stripped):
        return False
    if _is_date_only_cell(stripped):
        return True
    if re.fullmatch(r"[-—–]+(?:\s*\|\s*[-—–]+)*", stripped):
        return True
    if ORPHAN_BASIS_CONT_RE.search(stripped):
        return True
    if CROSS_PAGE_CONTINUATION_START_RE.match(stripped):
        return True
    if stripped.startswith("|"):
        return True
    lower = stripped.lower()
    if lower.startswith(("лицо ", "общество", "составляющие", "является ", "принадлежит")):
        return True
    if stripped.endswith(",") or stripped.endswith(";"):
        return True
    return False


def _iter_pipe_row_chunks(block: str) -> list[tuple[int, list[str]]]:
    """
    Разбить текст таблицы на логические строки (номер → список текстовых линий).

    Учитывает межстраничный перенос: хвост без п/п клеится к предыдущему chunk.
    """
    chunks: list[tuple[int, list[str]]] = []
    current_num: int | None = None
    current_lines: list[str] = []
    after_page_break = False

    def flush() -> None:
        nonlocal current_num, current_lines, after_page_break
        if current_num is not None and current_lines:
            chunks.append((current_num, current_lines))
        current_num = None
        current_lines = []
        after_page_break = False

    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if SECTION_HEADER_RE.match(stripped):
            flush()
            continue

        if _is_page_or_table_boundary_line(stripped):
            after_page_break = True
            continue

        if _is_repeatable_table_header_line(stripped):
            continue

        row_num = _match_data_row_start(stripped)
        if row_num is not None:
            flush()
            current_num = row_num
            current_lines = [line]
            after_page_break = False
            continue

        if current_num is not None:
            current_lines.append(line)
            after_page_break = False
            continue

        # Нет открытой строки: «висячие» строки после разрыва страницы → к предыдущему chunk.
        if after_page_break and chunks and _is_cross_page_continuation_line(stripped):
            prev_num, prev_lines = chunks[-1]
            chunks[-1] = (prev_num, prev_lines + [line])
            after_page_break = False
            continue

        if chunks and _is_cross_page_continuation_line(stripped):
            prev_num, prev_lines = chunks[-1]
            chunks[-1] = (prev_num, prev_lines + [line])
            continue

    flush()
    return chunks


def _row_quality(record: dict[str, Any]) -> int:
    """Эвристический score полноты записи (основания, даты, доли, col3)."""
    bases = record.get("affiliation_basis") or []
    basis_text = record.get("_basis_text") or ""
    score = len(bases) * 100 + len(basis_text)
    if record.get("share_authorized_capital_pct"):
        score += 5
    if is_meaningful_share(record.get("share_authorized_capital_pct")):
        score += 12
    if is_meaningful_share(record.get("share_ordinary_stocks_pct")):
        score += 8
    basis_dates = record.get("basis_date") or []
    if basis_dates:
        score += 5 + len(basis_dates)
    col3 = record.get("address")
    if col3 and col3 not in ("-", "---"):
        score += 20
        if CONSENT_COL3_RE.search(str(col3)):
            score += 15
    return score


def _block_rows_score(rows: list[dict[str, Any]]) -> int:
    """Суммарный score списка записей блока таблицы."""
    return sum(_row_quality(row) for row in rows)


def _merge_record(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Слить две версии одной строки (дубликаты таблиц в section)."""
    if _row_quality(new) > _row_quality(existing):
        winner, loser = new, existing
    else:
        winner, loser = existing, new

    merged = dict(winner)
    if not merged.get("affiliation_basis") and loser.get("affiliation_basis"):
        merged["affiliation_basis"] = loser["affiliation_basis"]
    bases = merged.get("affiliation_basis") or loser.get("affiliation_basis") or []
    winner_dates = merged.get("basis_date") or []
    loser_dates = loser.get("basis_date") or []
    chosen_dates = winner_dates if len(winner_dates) >= len(loser_dates) else loser_dates
    if not chosen_dates:
        chosen_dates = winner_dates or loser_dates
    merged["basis_date"] = align_basis_dates_to_bases(bases, chosen_dates)
    if merged.get("share_authorized_capital_pct") in (None, "", "-", "—") and loser.get(
        "share_authorized_capital_pct"
    ):
        merged["share_authorized_capital_pct"] = loser["share_authorized_capital_pct"]
    if merged.get("share_ordinary_stocks_pct") in (None, "", "-", "—") and loser.get(
        "share_ordinary_stocks_pct"
    ):
        merged["share_ordinary_stocks_pct"] = loser["share_ordinary_stocks_pct"]
    return merged


def _table_blocks(section_text: str) -> list[str]:
    """Вырезать блоки таблиц из section_text."""
    blocks: list[str] = []
    tables_match = re.search(r"===== ТАБЛИЦЫ.*", section_text, flags=re.DOTALL)
    if tables_match:
        blocks.append(tables_match.group(0))

    for match in re.finditer(
        r"--- Таблица \d+.*?(?=--- Таблица|--- Страница|=====|\Z)",
        section_text,
        flags=re.DOTALL,
    ):
        blocks.append(match.group(0))

    if not blocks:
        blocks = [section_text]
    return blocks


def _parse_block_rows_ordered(block: str, column3_mode: str = "address") -> list[dict[str, Any]]:
    """Разобрать один блок таблицы в упорядоченный список записей."""
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row_num, lines in _iter_pipe_row_chunks(block):
        parsed = _parse_row_lines(row_num, lines, column3_mode=column3_mode)
        if not parsed:
            continue
        name_key = str(parsed.get("full_name") or "")[:40].strip().lower()
        if not name_key or len(name_key) < 4:
            continue
        if name_key.startswith(("физические лица", "юридические лица")):
            continue
        fingerprint = f"{row_num}:{name_key}"
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        parsed["row_number_pdf"] = row_num
        ordered.append(parsed)

    return ordered


def parse_affiliate_table_rows_ordered(section_text: str) -> list[dict[str, Any]]:
    """Все записи таблицы из section_text в порядке появления."""
    column3_mode = detect_column3_mode(section_text)
    best: list[dict[str, Any]] = []
    best_score = -1

    for block in _table_blocks(section_text):
        block_rows = _parse_block_rows_ordered(block, column3_mode=column3_mode)
        score = _block_rows_score(block_rows)
        if score > best_score or (score == best_score and len(block_rows) > len(best)):
            best = block_rows
            best_score = score

    for idx, record in enumerate(best):
        row_num = int(record.get("row_number_pdf") or record.get("row_number") or 0)
        next_num = (
            int(
                best[idx + 1].get("row_number_pdf")
                or best[idx + 1].get("row_number")
                or row_num + 1
            )
            if idx + 1 < len(best)
            else row_num + 1
        )
        _attach_orphan_basis_to_record(record, section_text, row_num, next_num)
        record.pop("_basis_text", None)
        record["affiliation_basis"] = [
            p for p in (record.get("affiliation_basis") or []) if is_valid_basis_phrase(p)
        ]
        record["basis_date"] = align_basis_dates_to_bases(
            record["affiliation_basis"],
            record.get("basis_date") or [],
        )

    return best


def count_affiliate_data_rows(section_text: str) -> int | None:
    """Детерминированный подсчёт числа строк данных в таблице."""
    ordered = parse_affiliate_table_rows_ordered(section_text)
    if not ordered:
        return None

    seen: set[int] = set()
    for record in ordered:
        row_num = normalize_row_number(
            record.get("row_number_pdf") or record.get("row_number")
        )
        if row_num is not None and row_num > 0:
            seen.add(row_num)
    return len(seen) if seen else None


def _find_council_continuation(window: str) -> tuple[str | None, list[str]]:
    """Устаревающий поиск хвоста «Совета директоров» в узком окне."""
    if not window:
        return None, []
    match = COUNCIL_CONTINUATION_RE.search(window)
    if not match:
        return None, []
    # Текст до Совета не должен быть новой строкой данных с ФИО.
    before = window[: match.start()]
    after = window[match.start() : match.end() + 80]
    # Не брать, если это шапка («членам Совета» редко, но защитимся).
    if BASIS_GARBAGE_RE.search(match.group(0)):
        return None, []
    cont = _flatten(match.group(0))
    if not cont.lower().startswith("совета"):
        return None, []
    dates = extract_basis_dates(after)
    if not dates:
        dates = extract_basis_dates(window[match.start() : match.start() + 200])
    return cont, dates


def _row_anchor_pos(section_text: str, full_name: str | None, truncated_hint: str) -> int:
    """Позиция якоря ФИО/названия в section для orphan-поиска (первое вхождение)."""
    text = section_text or ""
    name_hits: list[int] = []
    name = _flatten(full_name or "")
    if name:
        tokens = [t for t in re.split(r"\s+", name) if len(t) >= 5]
        # Для юрлиц предпочтительнее уникальный фрагмент названия.
        key = tokens[-1] if tokens else name[:24]
        if "«" in name:
            q = re.search(r"[«\"]([^»\"]{4,})", name)
            if q:
                key = q.group(1)
        if key:
            start = 0
            lower = text.lower()
            needle = key.lower()
            while True:
                i = lower.find(needle, start)
                if i < 0:
                    break
                name_hits.append(i)
                start = i + len(needle)
    if name_hits:
        return name_hits[0]
    # Fallback: truncated hint
    hint = _flatten(truncated_hint or "")
    if len(hint) >= 24:
        frag = hint[:48].lower()
        i = text.lower().find(frag)
        if i >= 0:
            return i
    return 0


def _extract_orphan_share_tail(section_text: str, after_pos: int) -> tuple[list[str], list[str]]:
    """Orphan-блок «составляющие уставный капитал данного лица; …» + доп.основания/даты."""
    m = ORPHAN_SHARES_TAIL_BLOCK_RE.search(section_text or "", after_pos)
    if not m:
        return [], []
    full = _flatten(m.group(0))
    if "данного лица" not in full.lower():
        return [], []
    dates = extract_basis_dates(full)
    # Убираем даты из текста перед разбором оснований (они ломают split).
    cleaned = BASIS_DATE_PATTERN.sub(" ", full)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cont_m = re.search(
        r"составляющие\s+уставный\s+капитал\s+данного\s+лица;?",
        cleaned,
        re.IGNORECASE,
    )
    if not cont_m:
        return [], dates
    cont_frag = cont_m.group(0).rstrip(";").strip()
    rest = cleaned[cont_m.end() :].strip(" ;.")
    rest_phrases = [
        p for p in split_basis_cell(rest) if is_valid_basis_phrase(p)
    ]
    return [cont_frag] + rest_phrases, dates


def _extract_orphan_council_tail(section_text: str, after_pos: int) -> tuple[str | None, list[str]]:
    """Orphan «Совета директоров Общества» (+ дата) после якоря строки."""
    for m in ORPHAN_COUNCIL_TAIL_RE.finditer(section_text or "", after_pos):
        before = _flatten(section_text[max(0, m.start() - 60) : m.start()]).lower()
        if "является членом совета" in before:
            continue
        dates = [d for d in (m.group("date1"), m.group("date2")) if d]
        if not dates:
            dates = extract_basis_dates(m.group(0))
        return "Совета директоров Общества", dates
    return None, []


def _org_name_open_stem(full_name: str | None) -> str | None:
    """Стем незакрытого наименования в кавычках / после ОПФ (для склейки)."""
    name = _flatten(full_name or "")
    if not name:
        return None
    q = re.search(r"[«\"]([^»\"]+)$", name)
    if q:
        stem = q.group(1).strip()
        return stem if len(stem) >= 4 else None
    # Без кавычек: хвост после типовой ОПФ (для склейки с PDF, где кавычки есть).
    stripped = re.sub(
        r"^(?:публичное\s+)?акционерное\s+общество\s+",
        "",
        name,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(
        r"^общество\s+с\s+ограниченной\s+ответственностью\s+",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    stripped = stripped.strip(" «»\"")
    if stripped and stripped.lower() != name.lower() and len(stripped) >= 8:
        return stripped
    return None


def _extract_orphan_quote_name_tail(
    section_text: str,
    incomplete_name: str | None,
) -> dict[str, Any] | None:
    """
    Закрытие оборванного имени в «кавычках» + хвост адреса/оснований.

    Ищет стем без intervening «чужих» кавычек; не цепляет » соседних строк.
    """
    text = section_text or ""
    stem = _org_name_open_stem(incomplete_name)
    if not stem:
        return None

    stem_re = re.compile(
        r"[«\"]\s*" + re.escape(stem) + r"(?=[^»\"]*)",
        re.IGNORECASE,
    )
    # Fallback: стем без кавычек в тексте (step1 без «»).
    stem_plain_re = re.compile(re.escape(stem), re.IGNORECASE)

    candidates: list[tuple[int, int, str, str]] = []
    for m in list(stem_re.finditer(text)) or list(stem_plain_re.finditer(text)):
        after_stem = m.end()
        region = text[after_stem : after_stem + 2200]
        # Между стемом и » не должно быть новой « — иначе это чужая строка.
        close_m = re.search(
            r"([А-Яа-яёЁA-Za-z0-9\-]+)\s*[»\"]",
            region,
        )
        if not close_m:
            continue
        between = region[: close_m.start()]
        if "«" in between:
            continue
        if _NEXT_ROW_START_RE.search(between):
            continue
        # Хвост после » до следующей строки п/п.
        after_close = region[close_m.end() :]
        end_m = _NEXT_ROW_START_RE.search(after_close)
        if end_m is None:
            # Не берём безграничный хвост: иначе даты/основания соседних строк.
            if re.search(r"\n\s*\d{1,2}\s*[.|]", after_close[:900]):
                continue
            rest = after_close[:500]
        else:
            rest = after_close[: end_m.start()]
        name_word = close_m.group(1)
        if BASIS_GARBAGE_RE.search(name_word) or len(name_word) < 3:
            continue
        candidates.append((m.start(), after_stem + close_m.end(), name_word, rest))

    if not candidates:
        return None
    # Последний валидный стем (обычно строка перед разрывом страницы).
    _start, _end, name_word, rest = candidates[-1]

    dates = extract_basis_dates(rest)
    cleaned = BASIS_DATE_PATTERN.sub(" ", rest)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    addr_parts: list[str] = []
    region_m = re.search(
        r"([А-Яа-яёЁ\-]+(?:\s+[А-Яа-яёЁ\-]+){0,3}\s+"
        r"(?:край|область|республика)(?:\s+[А-Яа-яёЁ\(\)\-]+)*)",
        cleaned,
        re.IGNORECASE,
    )
    if region_m:
        addr_parts.append(_flatten(region_m.group(1)).rstrip(","))
    city_m = re.search(r"(г\.\s*[А-Яа-яёЁ\-]+)", cleaned, re.IGNORECASE)
    if city_m:
        addr_parts.append(_flatten(city_m.group(1)))

    extra_bases = [
        p for p in split_basis_cell(cleaned) if is_valid_basis_phrase(p)
    ]
    if not any("осуществляет" in p.lower() for p in extra_bases):
        exec_m = re.search(
            r"(Общество[,]?\s+осуществляет[\s\S]{10,220}?данного\s+лица\.?)",
            cleaned,
            re.IGNORECASE,
        )
        if exec_m:
            phrase = _flatten(exec_m.group(1))
            if is_valid_basis_phrase(phrase):
                extra_bases.append(phrase)

    # Не тащим чужие «Общество имеет право…» из мусора шапки/соседних кусков:
    # оставляем только продолжения после разрыва (группа / единоличный орган).
    filtered_extra: list[str] = []
    for p in extra_bases:
        pl = p.lower()
        if "осуществляет" in pl or (
            "группе" in pl and "принадлежит" in pl
        ):
            filtered_extra.append(p)

    group_close = bool(re.search(r"принадлежит\s+Общество", cleaned, re.IGNORECASE))

    return {
        "name_tail": f"{name_word}»",
        "address_tail": ", ".join(addr_parts) if addr_parts else None,
        "group_close": group_close,
        "extra_bases": filtered_extra,
        "dates": dates,
    }


def _complete_truncated_basis_in_record(
    record: dict[str, Any],
    section_text: str,
    row_num: int,
    next_row: int,
) -> bool:
    """
    Доклеить обрыв основания/имени/адреса с следующей страницы по orphan-якорю.

    Не опирается на хрупкое окно row→row+1 (таблицы дублируются в text-layer).
    """
    del next_row  # якорь надёжнее окна между номерами
    bases = list(record.get("affiliation_basis") or [])
    raw = str(record.get("_basis_text") or "")
    last = bases[-1] if bases else ""
    flat_raw = _flatten(raw)
    full_name = str(record.get("full_name") or "")
    address = str(record.get("address") or "")

    needs_council = _basis_needs_council_continuation(last) or bool(
        re.search(r"является\s+членом\s*\.?\s*$", flat_raw, re.IGNORECASE)
        and "совета директоров" not in flat_raw.lower()
    )
    needs_society = (not needs_council) and (
        _basis_needs_society_suffix(last)
        or any(_basis_needs_society_suffix(b) for b in bases)
    )
    needs_shares_tail = bool(
        (last and last.rstrip().endswith(","))
        or is_truncated_basis_phrase(last)
        or re.search(r"приходящихся\s+на\s+акци[\w]*\s*,?\s*$", flat_raw, re.I)
    ) and not needs_council and not needs_society
    # Имя/адрес: только явный обрыв (незакрытые кавычки или адрес, оканчивающийся запятой).
    needs_quote_name = _name_missing_closing_quote(full_name) or (
        bool(_org_name_open_stem(full_name)) and address.rstrip().endswith(",")
    )

    if not needs_council and not needs_society and not needs_shares_tail and not needs_quote_name:
        return False

    # Если «членом» отфильтровали из bases — вернуть из raw.
    if needs_council and bases and not _basis_needs_council_continuation(bases[-1]):
        if re.search(r"является\s+членом\s*\.?\s*$", flat_raw, re.I):
            if not any(_basis_needs_council_continuation(b) for b in bases):
                bases.append("Лицо является членом")

    if needs_shares_tail and bases and not (
        bases[-1].rstrip().endswith(",") or is_truncated_basis_phrase(bases[-1])
    ):
        recovered = [
            p
            for p in split_basis_cell(raw)
            if is_valid_basis_phrase(p) or is_truncated_basis_phrase(p)
        ]
        if recovered and is_truncated_basis_phrase(recovered[-1]):
            bases = recovered

    hint = last or flat_raw
    anchor = _row_anchor_pos(section_text, full_name, hint)
    changed = False

    if needs_council or needs_society:
        cont, dates = _extract_orphan_council_tail(section_text, anchor)
        if cont and bases:
            if needs_council and _basis_needs_council_continuation(bases[-1]):
                bases[-1] = _flatten(f"{bases[-1]} {cont}")
                changed = True
            elif needs_society:
                for i, b in enumerate(bases):
                    if _basis_needs_society_suffix(b):
                        bases[i] = _flatten(b.rstrip(". ") + " Общества")
                        changed = True
                        break
            elif needs_council:
                bases.append(_flatten(f"Лицо является членом {cont}"))
                changed = True
            if changed:
                bases = [
                    p
                    for p in bases
                    if is_valid_basis_phrase(p) or is_truncated_basis_phrase(p)
                ]
                record["affiliation_basis"] = bases
                if dates:
                    record["basis_date"] = _unique_dates_ordered(
                        list(record.get("basis_date") or []) + dates
                    )

    if needs_shares_tail:
        tail_parts, dates = _extract_orphan_share_tail(section_text, anchor)
        if tail_parts and bases:
            cont_frag = tail_parts[0]
            extra = tail_parts[1:]
            if cont_frag.lower().startswith("составляющие"):
                bases[-1] = _flatten(f"{bases[-1]} {cont_frag}")
            bases.extend(extra)
            bases = [p for p in bases if is_valid_basis_phrase(p)]
            record["affiliation_basis"] = bases
            if dates:
                record["basis_date"] = _unique_dates_ordered(
                    list(record.get("basis_date") or []) + dates
                )
            changed = True

    if needs_quote_name:
        orphan = _extract_orphan_quote_name_tail(section_text, full_name)
        if orphan:
            name = _flatten(full_name)
            tail_word = orphan["name_tail"].rstrip("»\"").lower()
            if name and tail_word and tail_word not in name.lower():
                if "«" in name and "»" not in name:
                    record["full_name"] = _flatten(f"{name} {orphan['name_tail']}")
                else:
                    record["full_name"] = _flatten(f"{name} {tail_word}")
                changed = True
            addr = _flatten(address)
            addr_tail = orphan.get("address_tail")
            if addr_tail:
                if not addr or addr in ("-", "---", "—"):
                    record["address"] = addr_tail
                    changed = True
                elif addr_tail.lower() not in addr.lower():
                    if addr.rstrip().endswith(","):
                        record["address"] = _flatten(f"{addr} {addr_tail}")
                    else:
                        record["address"] = _flatten(f"{addr}, {addr_tail}")
                    changed = True
            if orphan.get("group_close") and bases:
                for i, b in enumerate(bases):
                    bl = b.lower()
                    if "группе" in bl and "принадлежит общество" not in bl:
                        if re.search(r"к\s+которой\s*$", bl):
                            bases[i] = _flatten(b + " принадлежит Общество")
                            changed = True
            extras = orphan.get("extra_bases") or []
            if extras:
                bases = _union_basis_phrases(bases, extras)
                record["affiliation_basis"] = bases
                changed = True
            # Даты строки уже из step1/PDF; из name-orphan не подмешиваем
            # (на хвосте страницы часто торчит дата следующей строки).

    return changed


def _attach_orphan_basis_to_record(
    record: dict[str, Any],
    section_text: str,
    row_num: int,
    next_row: int,
) -> None:
    """Обёртка: склейка orphan-хвоста для одной записи."""
    _complete_truncated_basis_in_record(record, section_text, row_num, next_row)


def _attach_orphan_basis_continuations(
    records: dict[int, dict[str, Any]],
    section_text: str,
    column3_mode: str = "address",
) -> None:
    """Досклейка orphan-хвостов для всего словаря records."""
    del column3_mode
    sorted_rows = sorted(records)
    for idx, row_num in enumerate(sorted_rows):
        next_row = sorted_rows[idx + 1] if idx + 1 < len(sorted_rows) else row_num + 1
        _complete_truncated_basis_in_record(
            records[row_num], section_text, row_num, next_row
        )


def _row_start_re(row_num: int) -> re.Pattern[str]:
    """Regex начала строки с данным номером п/п."""
    return re.compile(
        rf"(?m)^{row_num}(?:\.\s*\||\s*\|\s*(?!\d\s*\|)|\.\s+(?=[А-ЯA-ZЁ«\"])|\.\s*$)"
    )


def _extract_between_rows(section_text: str, row_num: int, next_row: int) -> str:
    """Текст между началом row_num и началом next_row (хрупко при дублях)."""
    start_match = _row_start_re(row_num).search(section_text)
    if not start_match:
        return ""

    tail = section_text[start_match.end() :]
    end_match = _row_start_re(next_row).search(tail)
    if end_match:
        return tail[: end_match.start()]
    return tail[:1200]


def extract_basis_dates_for_table_row(
    section_text: str,
    row_num: int,
    next_row: int | None = None,
) -> list[str]:
    """Даты колонки 5 в окне одной строки pipe-таблицы."""
    window = _extract_between_rows(section_text, row_num, next_row or row_num + 1)
    if not window:
        return []
    return extract_basis_dates(window)


def parse_affiliate_table_records(section_text: str) -> dict[int, dict[str, Any]]:
    """Словарь row_number → поля из детерминированного разбора section_text."""
    records: dict[int, dict[str, Any]] = {}
    column3_mode = detect_column3_mode(section_text)

    # Единый проход по всему section_text — межстраничные переносы не теряются на границе блоков.
    for row_num, lines in _iter_pipe_row_chunks(section_text):
        parsed = _parse_row_lines(row_num, lines, column3_mode=column3_mode)
        if not parsed:
            continue
        if row_num in records:
            records[row_num] = _merge_record(records[row_num], parsed)
        else:
            parsed["row_number_pdf"] = row_num
            records[row_num] = parsed

    _attach_orphan_basis_continuations(records, section_text, column3_mode=column3_mode)

    for record in records.values():
        record.pop("_basis_text", None)
        record["affiliation_basis"] = expand_affiliation_basis_list(
            record.get("affiliation_basis") or []
        )
        record["affiliation_basis"] = [
            p for p in record["affiliation_basis"] if is_valid_basis_phrase(p)
        ]
        record["basis_date"] = align_basis_dates_to_bases(
            record["affiliation_basis"],
            record.get("basis_date") or [],
        )

    return records


def normalize_row_number(value: Any) -> int | None:
    """Привести row_number к int или None."""
    if value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    text = str(value).strip().rstrip(".")
    return int(text) if text.isdigit() else None


def normalize_sequential_row_numbers(affiliates: list[dict]) -> list[dict]:
    """Проставить подряд идущие row_number; PDF-номер сохранить в row_number_pdf."""
    indexed = [(i, row) for i, row in enumerate(affiliates) if isinstance(row, dict)]
    if not indexed:
        return affiliates

    indexed.sort(
        key=lambda item: (
            item[1].get("row_number") is None,
            item[1].get("row_number") if item[1].get("row_number") is not None else 10**9,
            item[0],
        )
    )

    first_raw = normalize_row_number(indexed[0][1].get("row_number"))
    next_seq = first_raw or 1

    for _idx, row in indexed:
        raw = normalize_row_number(row.get("row_number"))
        if raw is not None and raw != next_seq:
            row["row_number_pdf"] = raw
        elif raw is not None and row.get("row_number_pdf") is None:
            row["row_number_pdf"] = raw
        row["row_number"] = next_seq
        next_seq += 1

    affiliates.sort(
        key=lambda row: row.get("row_number") if isinstance(row, dict) else 10**9
    )
    return affiliates


def match_record_by_name(
    records: dict[int, dict[str, Any]], full_name: str
) -> dict[str, Any] | None:
    """Найти запись в словаре по префиксу ФИО/наименования."""
    probe = full_name.strip().lower()
    if not probe:
        return None

    for record in records.values():
        name = str(record.get("full_name") or "").lower()
        if probe[:30] in name or name[:30] in probe:
            return record
    return None


def _lookup_pipe_record(
    records: dict[int, dict[str, Any]],
    row_number: int | None,
    full_name: str,
) -> dict[str, Any] | None:
    """Найти запись по номеру п/п или по имени."""
    if isinstance(row_number, int) and row_number in records:
        return records[row_number]
    return match_record_by_name(records, full_name)


def parse_pipe_table_text(table_text: str | None) -> dict[int, dict[str, Any]]:
    """Разобрать pipe-таблицу Step1 LLM тем же chunk-парсером, что и PDF."""
    if not (table_text or "").strip():
        return {}

    records: dict[int, dict[str, Any]] = {}
    for row_num, lines in _iter_pipe_row_chunks(table_text):
        flat = _flatten_multiline_row(lines)
        _, pipe_parts = _split_pipe_row_columns(flat)
        if is_table_column_numbering_row(flat, pipe_parts):
            continue
        parsed = _parse_row_lines(row_num, lines, column3_mode="address")
        if not parsed:
            continue
        name = str(parsed.get("full_name") or "").strip()
        if len(name) < 4:
            continue
        parsed["row_number_pdf"] = row_num
        # _basis_text нужен для доклейки «членом» → «Совета директоров» на разрыве страницы.
        if row_num in records:
            records[row_num] = _merge_record(records[row_num], parsed)
        else:
            records[row_num] = parsed
    return records


def _basis_list_quality(bases: list[str]) -> int:
    """Score качества списка оснований (валидные +, мусор −)."""
    if not bases:
        return 0
    score = len(bases) * 40
    for phrase in bases:
        if is_malformed_basis_phrase(phrase) or BASIS_GARBAGE_RE.search(phrase or ""):
            score -= 120
        elif is_valid_basis_phrase(phrase):
            score += 25
        elif is_truncated_basis_phrase(phrase):
            score += 5  # почти валидно, ждёт склейки
        else:
            score -= 40
    return score


def choose_merged_table_record(
    table_row: dict[str, Any] | None,
    step1_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Выбрать/слить лучшую версию строки: step1 LLM vs PDF pipe.

    Чистый step1 — источник истины по основаниям и датам.
    PDF добавляет только чистые доп.фразы; грязный OCR (склейка соседей,
    «Согласие…» в основании) никогда не затирает хороший step1.
    Даты из step1 — как есть, с повторами.
    """
    if not table_row and not step1_row:
        return None
    if not table_row:
        return dict(step1_row)
    if not step1_row:
        return dict(table_row)

    step1_bases_raw = list(step1_row.get("affiliation_basis") or [])
    pdf_bases_raw = list(table_row.get("affiliation_basis") or [])
    step1_bases = _clean_basis_list(step1_bases_raw, keep_truncated=True)
    pdf_bases_clean = _clean_basis_list(pdf_bases_raw, keep_truncated=False)

    pdf_dirty = bool(pdf_bases_raw) and (
        any(_basis_phrase_looks_polluted(p) for p in pdf_bases_raw)
        or any(BASIS_GARBAGE_RE.search(p or "") for p in pdf_bases_raw)
        or (
            step1_bases
            and len(pdf_bases_raw) >= len(step1_bases) + 2
        )
        or (len(pdf_bases_raw) > 0 and len(pdf_bases_clean) == 0)
    )

    step1_truncated = any(is_truncated_basis_phrase(b) for b in step1_bases) or bool(
        re.search(
            r"(?:акции|членом)\s*,?\s*$",
            _flatten(str(step1_row.get("_basis_text") or "")),
            flags=re.IGNORECASE,
        )
    ) or any(_basis_needs_society_suffix(b) for b in step1_bases)

    # Грязный PDF / усечённый step1 → победитель step1 (доклейка orphan позже).
    if step1_bases and (pdf_dirty or step1_truncated or not pdf_bases_clean):
        winner, loser = step1_row, table_row
        prefer_step1_bases = True
    elif _basis_list_quality(step1_bases) >= _basis_list_quality(pdf_bases_clean):
        winner, loser = step1_row, table_row
        prefer_step1_bases = True
    else:
        winner, loser = table_row, step1_row
        prefer_step1_bases = False

    merged = dict(winner)

    if prefer_step1_bases and step1_bases:
        # Чистый PDF может дополнить step1 (др. основание на разрыве страницы).
        # Грязный PDF/OCR — никогда: оставляем только step1, хвосты доклеит orphan.
        if pdf_bases_clean and not pdf_dirty:
            merged["affiliation_basis"] = _union_basis_phrases(
                step1_bases, pdf_bases_clean
            )
        else:
            merged["affiliation_basis"] = list(step1_bases)
        merged["_basis_text"] = step1_row.get("_basis_text") or "\n".join(
            str(b) for b in step1_bases_raw
        )
    elif pdf_bases_clean:
        if step1_bases and not pdf_dirty:
            merged["affiliation_basis"] = _union_basis_phrases(
                pdf_bases_clean, step1_bases
            )
        else:
            merged["affiliation_basis"] = list(pdf_bases_clean)
        merged["_basis_text"] = table_row.get("_basis_text") or "\n".join(
            str(b) for b in pdf_bases_raw
        )
    elif step1_bases:
        merged["affiliation_basis"] = list(step1_bases)
        merged["_basis_text"] = step1_row.get("_basis_text") or "\n".join(
            str(b) for b in step1_bases_raw
        )
    elif loser.get("affiliation_basis"):
        merged["affiliation_basis"] = _clean_basis_list(
            list(loser.get("affiliation_basis") or []), keep_truncated=True
        )

    merged["full_name"] = _prefer_full_name(
        winner.get("full_name"), loser.get("full_name")
    )

    for share_key in ("share_authorized_capital_pct", "share_ordinary_stocks_pct"):
        cur = normalize_share_pct(merged.get(share_key))
        alt = normalize_share_pct(loser.get(share_key))
        if cur is None and alt is not None:
            merged[share_key] = alt
        elif is_share_placeholder(merged.get(share_key)) and is_meaningful_share(
            loser.get(share_key)
        ):
            cleaned = normalize_share_pct(loser.get(share_key))
            if cleaned is not None:
                merged[share_key] = cleaned
        else:
            merged[share_key] = cur if cur is not None else merged.get(share_key)

    # Адрес / col3: более полный фрагмент, без подстановки с титула.
    w_addr = _flatten(str(winner.get("address") or ""))
    l_addr = _flatten(str(loser.get("address") or ""))
    if not w_addr or w_addr in ("-", "---", "—"):
        if l_addr and l_addr not in ("-", "---", "—"):
            merged["address"] = loser.get("address")
    elif l_addr and l_addr not in ("-", "---", "—"):
        # Предпочитаем полное «Согласие … не получено» усечённому «Согласие физического лица».
        if CONSENT_COL3_RE.search(l_addr) and not CONSENT_COL3_RE.search(w_addr):
            merged["address"] = loser.get("address")
        elif w_addr.rstrip().endswith(",") and l_addr.lower() not in w_addr.lower():
            if len(l_addr) > len(w_addr):
                merged["address"] = loser.get("address")
        elif len(l_addr) > len(w_addr) + 5 and w_addr.lower() in l_addr.lower():
            merged["address"] = loser.get("address")

    # Даты: step1 «как есть» (с повторами). Грязный PDF не подмешиваем.
    step1_dates = _dates_as_found(list(step1_row.get("basis_date") or []))
    pdf_dates = _dates_as_found(list(table_row.get("basis_date") or []))
    if step1_dates:
        merged["basis_date"] = step1_dates
    elif pdf_dates and not pdf_dirty:
        merged["basis_date"] = pdf_dates
    else:
        merged["basis_date"] = pdf_dates if pdf_dates and not pdf_dirty else step1_dates
    return merged


def apply_table_records_to_affiliates(
    affiliates: list[dict],
    section_text: str,
    full_text: str | None = None,
    table_text: str | None = None,
) -> list[dict]:
    """Подмешать поля таблицы в affiliates JSON + orphan-доклейка + даты."""
    table_records = parse_affiliate_table_records(section_text)
    step1_records = parse_pipe_table_text(table_text)
    if not table_records and not step1_records:
        return affiliates

    column3_mode = detect_column3_mode(section_text)
    emitter_candidates = extract_emitter_address_candidates(full_text or section_text)

    for row in affiliates:
        if not isinstance(row, dict):
            continue

        full_name = str(row.get("full_name") or "")
        row_number = row.get("row_number")
        pdf_row = _lookup_pipe_record(table_records, row_number, full_name)
        step1_row = _lookup_pipe_record(step1_records, row_number, full_name)
        table_row = choose_merged_table_record(pdf_row, step1_row)

        if not table_row:
            row["affiliation_basis"] = _sanitize_basis_list(row.get("affiliation_basis") or [])
            row["basis_date"] = align_basis_dates_to_bases(
                row["affiliation_basis"],
                row.get("basis_date") or [],
            )
            if column3_mode == "address":
                row["address"] = None
                row["_col3_from_table"] = False
            row["_from_table"] = False
            continue

        row["row_number"] = (
            table_row.get("row_number_pdf")
            or table_row.get("row_number")
            or row.get("row_number")
        )
        row["row_number_pdf"] = (
            table_row.get("row_number_pdf")
            or table_row.get("row_number")
            or row.get("row_number_pdf")
        )

        preferred_name = _prefer_full_name(row.get("full_name"), table_row.get("full_name"))
        if preferred_name:
            row["full_name"] = preferred_name

        table_bases_raw = list(table_row.get("affiliation_basis") or [])
        # Не выбрасываем оборванные «…является членом» до доклейки хвоста со след. страницы.
        table_bases = [
            p
            for p in table_bases_raw
            if is_valid_basis_phrase(p) or is_truncated_basis_phrase(p)
        ]
        if not table_bases:
            table_bases = _sanitize_basis_list(row.get("affiliation_basis") or [])

        row["affiliation_basis"] = table_bases
        row["_basis_text"] = table_row.get("_basis_text") or "\n".join(
            str(b) for b in table_bases_raw
        )
        row["_from_table"] = bool(table_bases)

        table_col3 = table_row.get("address")
        if column3_mode == "address":
            row["address"] = table_col3
            row["_col3_from_table"] = True
        elif table_col3:
            row["address"] = table_col3
            row["_col3_from_table"] = True
        elif column3_mode == "inn_ogrn" and looks_like_postal_address(row.get("address")):
            row_num = int(
                table_row.get("row_number_pdf") or table_row.get("row_number") or 0
            )
            window = _extract_between_rows(section_text, row_num, row_num + 1)
            row["address"] = extract_column3_from_row_blob(window, mode=column3_mode) or None
            row["_col3_from_table"] = bool(row.get("address"))

        row_num = row.get("row_number_pdf") or row.get("row_number")
        next_row = int(row_num) + 1 if isinstance(row_num, int) else 10**9
        # Даты только из таблицы/step1 + orphan-хвоста — не из «грязного» JSON LLM.
        seed_dates = list((step1_row or {}).get("basis_date") or []) or list(
            table_row.get("basis_date") or []
        )
        row["basis_date"] = list(seed_dates)
        if isinstance(row_num, int):
            _complete_truncated_basis_in_record(
                row, section_text, int(row_num), next_row
            )

        # Финальная фильтрация мусора после доклейки.
        row["affiliation_basis"] = [
            p
            for p in (row.get("affiliation_basis") or [])
            if is_valid_basis_phrase(p)
        ]

        final_bases = row.get("affiliation_basis") or []
        raw_dates = pick_basis_dates_for_row(
            final_bases,
            existing=row.get("basis_date") or [],
            table_dates=table_row.get("basis_date") or [],
            section_dates=[],
            step1_dates=(step1_row or {}).get("basis_date") or [],
        )
        row["basis_date"] = align_basis_dates_to_bases(final_bases, raw_dates)

        for share_key in ("share_authorized_capital_pct", "share_ordinary_stocks_pct"):
            table_share = table_row.get(share_key)
            cleaned = normalize_share_pct(table_share) if table_share is not None else None
            if cleaned is not None and str(cleaned).strip():
                if is_meaningful_share(cleaned) or is_share_placeholder(row.get(share_key)):
                    row[share_key] = cleaned
            elif row.get(share_key):
                row[share_key] = normalize_share_pct(row.get(share_key))

        row.pop("_basis_text", None)

    if column3_mode == "address":
        # Только строки без табличного источника: срезаем подстановку LLM с титула.
        # Строки с _col3_from_table не трогаем — в т.ч. если адрес = адресу компании.
        for row in affiliates:
            if not isinstance(row, dict):
                continue
            if row.get("_col3_from_table"):
                continue
            if is_emitter_address_substitution(row.get("address"), emitter_candidates):
                row["address"] = None
            elif row.get("address"):
                row["address"] = None

    return affiliates


def deduplicate_affiliates_by_pdf_row(affiliates: list[dict]) -> list[dict]:
    """Одна запись на каждый row_number_pdf (лучшая по полноте)."""
    by_pdf: dict[int, dict] = {}
    for row in affiliates:
        if not isinstance(row, dict):
            continue
        pdf = row.get("row_number_pdf") or row.get("row_number")
        if not isinstance(pdf, int):
            continue
        prev = by_pdf.get(pdf)
        if prev is None or _row_quality(row) > _row_quality(prev):
            by_pdf[pdf] = row
    return [by_pdf[k] for k in sorted(by_pdf)]


def _sanitize_basis_list(basis_list: list[Any]) -> list[str]:
    """Очистка/разбиение склеенных элементов affiliation_basis."""
    return expand_affiliation_basis_list(basis_list)

# =============================================================================
# Блок 0. Константы и сообщения пайплайна
# =============================================================================
# Модели Ollama, DPI сканов, MAX_PAGES, тексты отказов для оператора.
# =============================================================================

# Автовыбор модели по типу пайплайна (--model auto).
MODEL_AUTO = "auto"
TEXT_LAYER_MODEL = "qwen2.5vl:7b" # qwen2.5:14b-instruct-q4_K_M
VISION_MODEL = "qwen2.5vl:7b"
DEFAULT_MODEL = MODEL_AUTO

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


def resolve_model_for_pipeline(user_model: str, use_vision: bool) -> str:
    """Выбрать модель Ollama: auto → text/vision default, иначе user_model."""
    requested = (user_model or MODEL_AUTO).strip()
    if requested.lower() == MODEL_AUTO:
        return VISION_MODEL if use_vision else TEXT_LAYER_MODEL
    return requested


def raw_subdir_for_model(model: str) -> str:
    """Имя подкаталога raw_<модель> для артефактов."""
    slug = re.sub(r"[^\w\-]+", "_", model.replace(":", "_").replace(".", "_"))
    slug = slug.strip("_") or "local_llm"
    return f"raw_{slug}"


# =============================================================================
# Блок 1. Промпты для локальной LLM (двухшаговый пайплайн: таблица → JSON)
# =============================================================================
# Промпты Step1 (pipe-таблица) и Step2 (JSON-каркас) для Ollama.

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
# Классификация PDF: text-layer / scan / hybrid, eligibility, план обработки.


def get_pdf_page_count(pdf_path: Path) -> int:
    """Число страниц PDF через PyMuPDF."""
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def get_scan_page_indices(
    pdf_path: Path,
    min_chars: int = MIN_TEXT_CHARS_PER_PAGE,
) -> list[int]:
    """Индексы страниц без достаточного text-layer (кандидаты на vision)."""
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
    """(страниц_с_текстом, всего) — для классификации scan/hybrid."""
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
    """True, если все страницы — картинки без text-layer."""
    pages_with_text, total = count_pages_with_text_layer(pdf_path)
    if total == 0:
        return False
    return pages_with_text == 0


def check_pdf_eligibility(
    pdf_path: Path,
    mode: str = "auto",
) -> tuple[bool, str | None, SkipReason | None]:
    """Проверка лимита страниц и полной-скан политики; текст отказа или None."""
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
    """План обработки: text-layer / scan / hybrid + список vision-страниц."""
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
    """Склеить текст vision/OCR скана с text-layer остальных страниц."""
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
    """True для моделей Ollama, умеющих картинки (vl, minicpm-v, llava…)."""
    name = (model or "").lower()
    return any(token in name for token in ("vl", "vision", "llava", "moondream", "minicpm-v"))


# =============================================================================
# Блок 2b. Vision-ветка: рендер сканов, авто-поворот страниц
# =============================================================================
# Рендер выбранных страниц PDF → PNG (DPI), автоповорот 0/90/180/270, OCR-хелперы.


def _horizontal_projection_score(img: Image.Image) -> float:
    """Оценка «горизонтальности» текста на PNG (для выбора поворота)."""
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
    """Подобрать угол 0/90/180/270 для скана."""
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
    """Повернуть PIL-изображение на angle градусов."""
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
    """Рендер выбранных страниц PDF в PNG (DPI, auto-rotate)."""
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
    """Tesseract OCR страницы с подбором PSM (вспомогательно)."""
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
    """OCR набора PNG → сплошной текст."""
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
    """Записать rotation_meta.json в каталог scan_pages."""
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
# Извлечение text-layer: pdfplumber → fitz → pypdf (smart_pdf_extract).


def extract_with_pdfplumber(pdf_path: Path) -> str:
    """Извлечь text-layer через pdfplumber (предпочтительный движок)."""
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
    """Извлечь text-layer через PyMuPDF."""
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
    """Извлечь text-layer через pypdf (fallback)."""
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
    """True, если в PDF есть читаемый text-layer."""
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
    """Умное извлечение: pdfplumber → fitz → pypdf, метаданные метода."""
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
# Вырезание текста «Состав аффилированных лиц» / Раздел I + блоки ===== ТАБЛИЦЫ.


def enrich_section_with_tables(full_text: str, section_text: str) -> str:
    """Добавить к section_text блоки ===== ТАБЛИЦЫ из full_text."""
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
    """Вырезать текст Раздела I / состава аффилированных лиц."""
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
# Ollama chat: Step1 pipe-таблица, Step2 JSON-каркас, QC-retry.


def extract_json_block(text: str) -> str:
    """Достать JSON-объект из ответа LLM (отрезать markdown-обёртки)."""
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
    """Вызов Ollama chat с ожиданием JSON в ответе."""
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
    """Step1: LLM → pipe-таблица аффилированных лиц (текст или vision)."""
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
    """Step2: LLM → каркас JSON affiliates из pipe + контекста."""
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
    """Полный двухшаговый LLM-разбор: step1 + step2."""
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
    """Двухшаговый разбор с повторной попыткой при QC FAIL."""
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
# Нормализация дат, ОГРН, наименования компании, адресов; enforce_basis_dates.


def normalize_date(value: Any) -> str | None:
    """Нормализовать дату к ДД.ММ.ГГГГ или None."""
    if not value:
        return None
    match = re.search(r"(\d{2})[./-](\d{2})[./-](\d{4})", str(value))
    if not match:
        return None
    day, month, year = match.groups()
    return f"{day}.{month}.{year}"


def normalize_ogrn(value: Any) -> str | None:
    """ОГРН: только цифры 13/15 или None."""
    if value is None:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) in (13, 15):
        return digits
    return None


def extract_ogrn_from_text(text: str) -> str | None:
    """Найти ОГРН в произвольном тексте титула/раздела."""
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
    """True, если имя эмитента — только ОПФ без названия."""
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
    """Наименование эмитента после «СПИСОК АФФИЛИРОВАННЫХ ЛИЦ»."""
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
    """Наименование эмитента через vision по PNG титула."""
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
    """Итоговое имя компании: текст / vision / JSON LLM."""
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
    """Дата списка: из текста раздела или из имени файла YYYY.MM.DD."""
    header_match = re.search(
        r"состав\s+аффилированных\s+лиц\s+на\s*(\d{2}[./-]\d{2}[./-]\d{4})",
        section_text,
        flags=re.IGNORECASE,
    )
    if header_match:
        return normalize_date(header_match.group(1))

    # «2019.09.30 …» в имени файла → 30.09.2019
    ymd = re.search(r"(20\d{2})[._-](\d{2})[._-](\d{2})", pdf_path.name)
    if ymd:
        return f"{ymd.group(3)}.{ymd.group(2)}.{ymd.group(1)}"

    file_match = re.search(r"(\d{2}\.\d{2}\.\d{4})", pdf_path.name)
    if file_match:
        return file_match.group(1)
    return None


def normalize_address(value: Any, column3_mode: str = "address") -> str | None:
    """Нормализация адреса/col3 под column3_mode."""
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


def normalize_result(
    result: dict,
    pdf_path: Path,
    section_text: str,
    full_text: str,
    cover_images: list[Path] | None = None,
    model: str | None = None,
) -> dict:
    """Нормализовать верхнеуровневые поля result (даты, ОГРН, company…)."""
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
    """Финально проставить basis_date из step1/table «как есть»."""
    del section_text
    step1_records = parse_pipe_table_text(table_text)

    for row in affiliates:
        if not isinstance(row, dict):
            continue

        bases = expand_affiliation_basis_list(row.get("affiliation_basis") or [])
        bases = [p for p in bases if is_valid_basis_phrase(p)]
        row["affiliation_basis"] = bases

        existing = [str(d).strip() for d in (row.get("basis_date") or []) if str(d).strip()]

        row_number = row.get("row_number_pdf") or row.get("row_number")
        step1_row = step1_records.get(row_number) if isinstance(row_number, int) else None
        if not step1_row:
            name_key = str(row.get("full_name") or "")[:40].strip().lower()
            for srec in step1_records.values():
                sname = str(srec.get("full_name") or "")[:40].strip().lower()
                if name_key and (name_key in sname or sname in name_key):
                    step1_row = srec
                    break

        step1_dates = (step1_row or {}).get("basis_date") or []
        raw_dates = pick_basis_dates_for_row(
            bases,
            existing=existing,
            table_dates=[],
            section_dates=[],
            step1_dates=step1_dates,
        )
        row["basis_date"] = align_basis_dates_to_bases(bases, raw_dates)

    return affiliates


# =============================================================================
# Блок 7. Mini-QC без второй модели (сверка числа строк и полноты оснований)
# =============================================================================
# QC: число строк PDF vs step1, полнота оснований; finalize_affiliates_list.


TABLE_ROW_PATTERN = re.compile(r"(?m)^(\d{1,3})\.\s+(?=[А-ЯA-ZЁ«\"])")


def count_step1_table_rows(table_text: str | None) -> int | None:
    """Число строк данных в pipe step1."""
    if not table_text:
        return None
    count = sum(
        1 for line in table_text.splitlines() if re.match(r"^\d+\s*\|", line.strip())
    )
    return count or None


def looks_like_truncated_basis(basis: str) -> bool:
    """QC-эвристика: основание похоже на обрыв."""
    text = basis.strip()
    if not text or len(text) < 12:
        return True
    if re.search(r";\s*Л\.?$", text):
        return True
    if not is_valid_basis_phrase(text):
        return True
    return False


def check_affiliation_basis_completeness(result: dict) -> tuple[list[dict], bool]:
    """QC: список проблем неполных оснований."""
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
    """Ожидаемое число строк: max(PDF-count, step1-count)."""
    from_table = count_affiliate_data_rows(section_text)
    from_step1 = count_step1_table_rows(table_text)
    if from_table and from_step1:
        return max(from_table, from_step1)
    if from_step1:
        return from_step1
    if from_table:
        return from_table

    numbers = [int(m.group(1)) for m in TABLE_ROW_PATTERN.finditer(section_text)]
    return len(numbers) if numbers else None


def finalize_affiliates_list(
    affiliates: list[dict],
    section_text: str,
    table_text: str | None = None,
    full_text: str | None = None,
) -> list[dict]:
    """Финальная сборка affiliates: merge таблиц, orphan, даты, sanitizer."""
    column3_mode = detect_column3_mode(section_text)
    affiliates = apply_table_records_to_affiliates(
        affiliates,
        section_text,
        full_text=full_text,
        table_text=table_text,
    )
    affiliates = enforce_basis_dates_for_affiliates(
        affiliates,
        section_text=section_text,
        table_text=table_text,
    )
    affiliates = deduplicate_affiliates_by_pdf_row(affiliates)
    for row in affiliates:
        if isinstance(row, dict):
            row["address"] = normalize_address(row.get("address"), column3_mode=column3_mode)
    return normalize_sequential_row_numbers(affiliates)


def attach_review_status_to_result(result: dict, qc_report: dict) -> dict:
    """Проставить qc_status / manual_review / status_message в result."""
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
    """Мини-QC: число строк, даты, ОГРН, company, полнота оснований."""
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
    """Нужен ли повторный LLM-проход по результатам QC."""
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
    """Печать сводки QC в консоль."""
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
# Сохранение section_i, step1, meta, qc, scan_pages в raw_<модель>/.


def ensure_raw_dir(output_dir: Path, model: str) -> Path:
    """Создать/вернуть каталог raw_<модель>."""
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
    """Сохранить status.json для пропущенного файла."""
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
    """Записать section_i, step1, qc, meta в raw_<модель>."""
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
# Оркестратор одного PDF: classify → extract → LLM → finalize → JSON.


def parse_single_pdf_local(
    pdf_path: Path,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    qc_retry: bool = True,
    mode: str = "auto",
    auto_rotate: bool = True,
    scan_dpi: int = DEFAULT_SCAN_DPI,
) -> tuple[Path | None, dict[str, Any]]:
    """Оркестратор одного PDF: classify → extract → LLM → finalize → JSON."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")

    mode = (mode or "auto").lower().strip()
    if mode not in ("auto", "text", "vision"):
        raise ValueError(f"Неизвестный mode: {mode!r} (допустимо: auto, text, vision)")

    output_dir.mkdir(parents=True, exist_ok=True)

    pages = get_pdf_page_count(pdf_path)
    eligible, skip_message, skip_reason = check_pdf_eligibility(pdf_path, mode=mode)

    if not eligible and skip_message and skip_reason:
        skip_use_vision = skip_reason == SkipReason.ALL_PAGES_SCAN
        skip_model = resolve_model_for_pipeline(model, use_vision=skip_use_vision)
        raw_dir = ensure_raw_dir(output_dir, skip_model)
        print(f"[SKIP] {skip_message}")
        status_path = save_skip_status(
            raw_dir=raw_dir,
            pdf_path=pdf_path,
            message=skip_message,
            reason=skip_reason,
            pages=pages,
            model=skip_model,
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

    model_requested = model
    model = resolve_model_for_pipeline(model, use_vision=use_vision)
    raw_dir = ensure_raw_dir(output_dir, model)
    if (model_requested or "").strip().lower() == MODEL_AUTO:
        print(f"[LLM] Auto-модель: {model} (pipeline: {pipeline})")
    elif model != model_requested:
        print(f"[LLM] Модель: {model}")

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

    print(f"[LLM] Ollama: {model}, pipeline: {pipeline}")
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
            "Локальная Ollama + детерминированный разбор таблицы"
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
# Точка входа: argparse, цикл по PDF, выбор модели, итоговая статистика.


def iter_pdf_files(input_dir: Path) -> list[Path]:
    """Список PDF в каталоге."""
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def main() -> None:
    """Точка входа CLI: argparse + цикл по файлам + статистика."""
    parser = argparse.ArgumentParser(
        description=(
            "PDF (<20 стр.) → локальная Ollama → affiliates JSON "
            "+ промежуточные файлы в raw_<модель>. "
            "По умолчанию: text-layer → qwen2.5:14b, сканы/гибриды → qwen2.5vl."
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
        help=(
            f"Модель Ollama (по умолчанию: {MODEL_AUTO} — text-layer → {TEXT_LAYER_MODEL}, "
            f"сканы/гибриды → {VISION_MODEL})"
        ),
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
    print(f"[STATS] Пропущено (>={MAX_PAGES} стр.): {skipped_pages} — «{MSG_TOO_MANY_PAGES}»")
    if args.mode == "text":
        print(f"[STATS] Пропущено (сканы, mode=text): {skipped_scans} — «{MSG_ALL_SCAN}»")
    print(f"[STATS] QC PASS: {qc_pass}, QC FAIL: {qc_fail}")
    print(f"[STATS] {MSG_MANUAL_REVIEW}: {manual_review}")
    print(f"[STATS] Ошибки: {errors}")
    print(f"[STATS] Время: {minutes} мин {seconds} сек")
    print(f"[STATS] Промежуточные файлы: {output_dir / raw_name}")


if __name__ == "__main__":
    main()
