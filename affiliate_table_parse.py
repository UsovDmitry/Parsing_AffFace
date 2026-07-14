"""
Детерминированный разбор таблицы аффилированных лиц из text-layer PDF.

Главная задача: собрать строки таблицы (в т.ч. с переносом страницы) без
«склеивания» заголовков, кодов эмитента и соседних записей.
"""

from __future__ import annotations

import re
from typing import Any

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
    r"Полное фирменное наименование|Доля участия аффилированного",
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
    re.compile(r"принадлежит\s+общество\s*$", re.IGNORECASE),
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


def _flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def detect_column3_mode(section_text: str) -> str:
    """
    Вариант колонки 3 таблицы аффилированных лиц.

    inn_ogrn — ОГРН/ИНН (отчёты только с физлицами, «Согласие не получено»);
    address  — место нахождения / жительства.
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
    """В таблице есть и физические, и юридические лица (отдельные подразделы)."""
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
    """Текст до «Раздел I» / «Состав аффилированных лиц» — титул, не таблица."""
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
    """
    Адреса эмитента только с титульной страницы (до Раздела I).

    Используется для отсечения подстановок LLM, а не для обнуления адресов
    из ячеек таблицы — совпадение с адресом компании в таблице допустимо.
    """
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
    if not address:
        return False
    return any(addresses_are_similar(address, candidate) for candidate in emitter_candidates)


def looks_like_postal_address(text: str | None) -> bool:
    """True, если строка похожа на почтовый адрес, а не на ИНН/согласие."""
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
    """Извлекает значение колонки 3 из склеенного текста строки таблицы."""
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
    """Склеивает переносы строк pipe-таблицы в одну логическую строку."""
    return _flatten("\n".join(ln.strip() for ln in lines if ln.strip()))


def _split_pipe_row_columns(flat_row: str) -> tuple[int | None, list[str]]:
    """Разбивает плоскую строку таблицы на номер п/п и ячейки колонок."""
    match = re.match(r"^(?P<num>\d{1,2})\s*\.?\s*\|\s*(?P<rest>.+)$", flat_row.strip())
    if not match:
        return None, []
    parts = [p.strip() for p in match.group("rest").split("|")]
    return int(match.group("num")), parts


def _is_column_numbering_pipe_parts(pipe_parts: list[str]) -> bool:
    """
    Строка — нумерация колонок таблицы (2|3|4|5|6|7), а не запись аффилиата.

    Срабатывает, если все непустые ячейки — целые числа, идущие подряд,
    начиная с 1–3 (после колонки п/п в pipe остаются номера 2…7).
    """
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
    """Строка pipe-таблицы — заголовок «№ | 1 | 2 | …», не данные аффилиата."""
    stripped = (flat_row or "").strip()
    if TABLE_COLUMN_HEADER_RE.match(stripped):
        return True
    if pipe_parts is None:
        _, pipe_parts = _split_pipe_row_columns(stripped)
    return _is_column_numbering_pipe_parts(pipe_parts)


def _normalize_basis_ocr(text: str) -> str:
    """Исправляет типичные OCR-опечатки в формулировках оснований."""
    return re.sub(r"\bЛифо\b", "Лицо", text or "", flags=re.IGNORECASE)


def is_malformed_basis_phrase(phrase: str) -> bool:
    """Битая склейка оснований (разрыв страницы / соседняя строка таблицы)."""
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


def _basis_needs_council_continuation(last: str) -> bool:
    """
    True только если последнее основание обрывается на «…является членом»
    и ждёт продолжения «Совета директоров Общества».
    """
    t = _flatten(_normalize_basis_ocr(last)).lower()
    if "совета" in t or "директор" in t:
        return False
    if "единоличным" in t or "исполнительным органом" in t:
        return False
    if "принадлежит" in t and "группе" in t:
        return False
    return bool(re.search(r"(?:является\s+)?членом\s*\.?\s*$", t))


def is_share_placeholder(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text or text.lower() == "null":
        return True
    return bool(re.fullmatch(r"[-—–]+", text))


def is_meaningful_share(value: Any) -> bool:
    """Доля из таблицы: процент, «N акций», не пусто и не «—»."""
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
    lower = text.lower()
    if not lower.startswith(
        ("лицо", "лифо", "общество", "юридическое лицо", "на основании")
    ):
        return False
    return True


def extract_basis_dates(cell_text: str) -> list[str]:
    """
    Извлекает все даты «Дата наступления основания» из текста ячейки.

    Сохраняет порядок и дубликаты (08.10.2020 может встречаться дважды
    для разных оснований).
    """
    if not cell_text:
        return []
    cleaned = str(cell_text).replace("г.", "").replace("Г.", "")
    return BASIS_DATE_PATTERN.findall(cleaned)


def _is_date_only_cell(text: str) -> bool:
    """True, если ячейка содержит только даты и разделители (пробелы, ; ,)."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    without_dates = BASIS_DATE_PATTERN.sub("", stripped)
    without_dates = re.sub(r"[\s;,\-—]+", "", without_dates)
    return not without_dates


def normalize_share_pct(value: Any) -> str | None:
    """
    Доля участия как строка: только числовая часть без округления.
    «100%-1 акция» → «100», «95,999999» → «95.999999», «—» без изменений.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    if re.fullmatch(r"[-—–]+", text):
        return text
    match = re.match(r"(\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", ".")
    return text


def _unique_dates_ordered(dates: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for d in dates:
        key = str(d).strip()
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def align_basis_dates_to_bases(
    affiliation_basis: list[str],
    basis_dates: list[str],
) -> list[str]:
    """
    Даты «как в источнике»: без растягивания/дублирования под число оснований.

    affiliation_basis не используется — оставлен для совместимости вызовов.
    """
    del affiliation_basis
    return _unique_dates_ordered(
        [str(d).strip() for d in basis_dates if d and str(d).strip()]
    )


def pick_basis_dates_for_row(
    affiliation_basis: list[str],
    *,
    existing: list[str] | None = None,
    table_dates: list[str] | None = None,
    section_dates: list[str] | None = None,
    step1_dates: list[str] | None = None,
) -> list[str]:
    """
    Собирает уникальные даты по приоритету источников (step1 → table → JSON → section).

    Без выравнивания под число оснований: даты берутся как есть.
    """
    del affiliation_basis
    ordered_sources = [
        list(step1_dates or []),
        list(table_dates or []),
        list(existing or []),
        list(section_dates or []),
    ]
    merged: list[str] = []
    seen: set[str] = set()
    for dates in ordered_sources:
        for d in dates:
            key = str(d).strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(key)
    return merged


def _clean_basis_text_for_split(text: str) -> str:
    """
    Убирает из текста ячейки «основание» хвосты колонок дат и долей,
    случайно попавшие при многострочном pipe-разборе.
    """
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
    """Дополняет оборванный фрагмент основания типовым субъектом «Лицо»."""
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
    """Разбивает текст ячейки «основание» на отдельные формулировки."""
    flat = _normalize_basis_ocr(_flatten(_clean_basis_text_for_split(text)))
    if not flat:
        return []

    # Отрезаем мусор из колонок ФИО/адреса до первого «Лицо/Общество».
    first_basis = re.search(r"(?:Лицо|Общество|Юридическое)", flat, flags=re.IGNORECASE)
    if first_basis:
        flat = flat[first_basis.start() :]

    # Разделение по началу каждой типовой формулировки основания.
    parts = re.split(
        r"(?=(?:(?:Лицо|Общество|Юридическое\s+лицо,?)\s+"
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
        if not is_valid_basis_phrase(phrase):
            continue
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            phrases.append(phrase)
    return phrases


def expand_affiliation_basis_list(basis_list: list[Any]) -> list[str]:
    """
    Разворачивает affiliation_basis: один элемент со склеенными основаниями
    через «;» превращается в массив отдельных формулировок.
    """
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

    basis_text = "\n".join(basis_chunks)
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


def _iter_pipe_row_chunks(block: str) -> list[tuple[int, list[str]]]:
    chunks: list[tuple[int, list[str]]] = []
    current_num: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_num, current_lines
        if current_num is not None and current_lines:
            chunks.append((current_num, current_lines))
        current_num = None
        current_lines = []

    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if SECTION_HEADER_RE.match(line.strip()):
            flush()
            continue
        if line.strip().startswith("---") or "===== ТАБЛИЦЫ" in line:
            flush()
            continue
        stripped = line.strip()
        if TABLE_COLUMN_HEADER_RE.match(stripped) or is_table_column_numbering_row(
            stripped
        ):
            flush()
            continue

        match = ROW_PIPE_START_RE.match(stripped)
        if match:
            flush()
            current_num = int(match.group("num"))
            current_lines = [line]
            continue

        if current_num is not None:
            current_lines.append(line)

    flush()
    return chunks


def _row_quality(record: dict[str, Any]) -> int:
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
    return sum(_row_quality(row) for row in rows)


def _merge_record(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
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
    """
    Строки таблицы в порядке появления в PDF (с дублями номеров п/п).
    Берётся лучший блок таблицы среди всех копий на страницах.
    """
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
    """
    Число строк данных в таблице (без заголовков разделов).

    Считаются уникальные номера п/п из PDF (row_number_pdf), без дублей
    из повторных фрагментов таблицы на разных страницах text-layer.
    """
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


def _attach_orphan_basis_to_record(
    record: dict[str, Any],
    section_text: str,
    row_num: int,
    next_row: int,
) -> None:
    """Склейка хвоста основания на разрыве страницы для одной записи."""
    bases = record.get("affiliation_basis") or []
    if not bases:
        return
    last = bases[-1]
    if not _basis_needs_council_continuation(last):
        return

    window = _extract_between_rows(section_text, row_num, next_row)
    attached = False

    for match in ORPHAN_BASIS_CONT_RE.finditer(window):
        cont = _flatten(match.group("basis"))
        if not cont or BASIS_GARBAGE_RE.search(cont):
            continue
        if cont.lower().startswith("совета"):
            merged_last = _flatten(f"{last} {cont}")
            rebuilt = bases[:-1] + (split_basis_cell(merged_last) or [merged_last])
            record["affiliation_basis"] = [
                p for p in rebuilt if is_valid_basis_phrase(p)
            ]
            attached = True
            break


def _attach_orphan_basis_continuations(
    records: dict[int, dict[str, Any]], section_text: str
) -> None:
    """
  На разрыве страницы хвост основания («Лицо является членом») часто
  продолжается отдельной строкой «| | | Совета директоров Общества. | ...».
    """
    for row_num in sorted(records):
        record = records[row_num]
        bases = record.get("affiliation_basis") or []
        if not bases:
            continue
        last = bases[-1]
        if not _basis_needs_council_continuation(last):
            continue

        window = _extract_between_rows(section_text, row_num, row_num + 1)
        attached = False

        for match in ORPHAN_BASIS_CONT_RE.finditer(window):
            cont = _flatten(match.group("basis"))
            if not cont or BASIS_GARBAGE_RE.search(cont):
                continue
            if cont.lower().startswith("совета"):
                merged_last = _flatten(f"{last} {cont}")
                rebuilt = bases[:-1] + (split_basis_cell(merged_last) or [merged_last])
                record["affiliation_basis"] = [
                    p for p in rebuilt if is_valid_basis_phrase(p)
                ]
                attached = True
                break


def _row_start_re(row_num: int) -> re.Pattern[str]:
    """Паттерн начала строки таблицы с номером п/п (с точкой или без)."""
    return re.compile(
        rf"(?m)^{row_num}(?:\.\s*\||\s*\|\s*(?!\d\s*\|)|\.\s+(?=[А-ЯA-ZЁ«\"]))"
    )


def _extract_between_rows(section_text: str, row_num: int, next_row: int) -> str:
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
    """
    Все даты «Дата наступления основания» в окне одной строки pipe-таблицы.
    Сохраняет порядок и повторы (08.10.2020 может встречаться дважды).
    """
    window = _extract_between_rows(section_text, row_num, next_row or row_num + 1)
    if not window:
        return []
    return extract_basis_dates(window)


def parse_affiliate_table_records(section_text: str) -> dict[int, dict[str, Any]]:
    """
    Собирает записи таблицы аффилированных лиц из pipe-таблиц section_text.
    Возвращает словарь row_number -> поля записи.
    """
    records: dict[int, dict[str, Any]] = {}
    column3_mode = detect_column3_mode(section_text)

    blocks = _table_blocks(section_text)

    for block in blocks:
        for row_num, lines in _iter_pipe_row_chunks(block):
            parsed = _parse_row_lines(row_num, lines, column3_mode=column3_mode)
            if not parsed:
                continue
            if row_num in records:
                records[row_num] = _merge_record(records[row_num], parsed)
            else:
                parsed["row_number_pdf"] = row_num
                records[row_num] = parsed

    _attach_orphan_basis_continuations(records, section_text)

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
    """Номер п/п из JSON/PDF: int или строка «12.» → 12."""
    if value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    text = str(value).strip().rstrip(".")
    return int(text) if text.isdigit() else None


def normalize_sequential_row_numbers(affiliates: list[dict]) -> list[dict]:
    """
    Нумерация п/п — строго последовательная в порядке записей.

    Опечатки в PDF (10, 11, 12, 12, 14) приводятся к 10, 11, 12, 13, 14.
    Исходный номер из PDF сохраняется в row_number_pdf при расхождении.
    """
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
    if isinstance(row_number, int) and row_number in records:
        return records[row_number]
    return match_record_by_name(records, full_name)


def parse_pipe_table_text(table_text: str | None) -> dict[int, dict[str, Any]]:
    """
    Разбор pipe-таблицы шага 1 LLM (N | full_name | col3 | basis | dates | shares).
    Использует тот же chunk-парсер, что и section_text PDF.
    """
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
        parsed.pop("_basis_text", None)
        if row_num in records:
            records[row_num] = _merge_record(records[row_num], parsed)
        else:
            records[row_num] = parsed
    return records


def _basis_list_quality(bases: list[str]) -> int:
    if not bases:
        return 0
    score = len(bases) * 40
    for phrase in bases:
        if is_malformed_basis_phrase(phrase):
            score -= 100
        elif is_valid_basis_phrase(phrase):
            score += 25
    return score


def choose_merged_table_record(
    table_row: dict[str, Any] | None,
    step1_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Лучший источник строки по полноте полей (PDF pipe vs step1 LLM)."""
    if not table_row and not step1_row:
        return None
    if not table_row:
        return dict(step1_row)
    if not step1_row:
        return dict(table_row)

    def record_score(record: dict[str, Any]) -> int:
        score = _row_quality(record) + _basis_list_quality(
            record.get("affiliation_basis") or []
        )
        for share_key in ("share_authorized_capital_pct", "share_ordinary_stocks_pct"):
            if is_meaningful_share(record.get(share_key)):
                score += 12
        return score

    winner = (
        table_row
        if record_score(table_row) >= record_score(step1_row)
        else step1_row
    )
    merged = dict(winner)
    loser = table_row if winner is step1_row else step1_row

    if not merged.get("affiliation_basis") and loser.get("affiliation_basis"):
        merged["affiliation_basis"] = loser["affiliation_basis"]
    if not merged.get("full_name") and loser.get("full_name"):
        merged["full_name"] = loser["full_name"]
    for share_key in ("share_authorized_capital_pct", "share_ordinary_stocks_pct"):
        if is_share_placeholder(merged.get(share_key)) and is_meaningful_share(
            loser.get(share_key)
        ):
            merged[share_key] = loser[share_key]
    if merged.get("address") in (None, "", "-", "---") and loser.get("address"):
        merged["address"] = loser.get("address")

    return merged


def apply_table_records_to_affiliates(
    affiliates: list[dict],
    section_text: str,
    full_text: str | None = None,
    table_text: str | None = None,
) -> list[dict]:
    """
    Подмешивает поля таблицы в affiliates JSON.

    Источник строки: лучший из детерминированного PDF pipe и pipe-таблицы шага 1 LLM.
    При column3_mode=address адрес только из таблицы (как есть).
    """
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

        table_bases = [
            p
            for p in (table_row.get("affiliation_basis") or [])
            if is_valid_basis_phrase(p)
        ]
        if table_bases:
            row["affiliation_basis"] = table_bases
        else:
            row["affiliation_basis"] = _sanitize_basis_list(row.get("affiliation_basis") or [])

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

        final_bases = row.get("affiliation_basis") or []
        row_num = row.get("row_number_pdf") or row.get("row_number")
        section_dates: list[str] = []
        if isinstance(row_num, int):
            section_dates = extract_basis_dates_for_table_row(section_text, row_num)

        raw_dates = pick_basis_dates_for_row(
            final_bases,
            existing=row.get("basis_date") or [],
            table_dates=table_row.get("basis_date") or [],
            section_dates=section_dates,
            step1_dates=(step1_row or {}).get("basis_date") or [],
        )
        row["basis_date"] = align_basis_dates_to_bases(final_bases, raw_dates)

        for share_key in ("share_authorized_capital_pct", "share_ordinary_stocks_pct"):
            table_share = table_row.get(share_key)
            if table_share is not None and str(table_share).strip():
                if is_meaningful_share(table_share) or is_share_placeholder(
                    row.get(share_key)
                ):
                    row[share_key] = normalize_share_pct(table_share)

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
    """Оставляет одну запись на каждый row_number_pdf (лучшая по полноте полей)."""
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
    """Очистка и разбиение склеенных оснований в массиве affiliation_basis."""
    return expand_affiliation_basis_list(basis_list)
