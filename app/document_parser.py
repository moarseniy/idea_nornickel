from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx", ".xlsm", ".png", ".jpg", ".jpeg"}


@dataclass
class ParsedDocument:
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


def safe_filename(filename: str) -> str:
    stem = Path(filename).name.strip() or "document"
    return re.sub(r"[^A-Za-zА-Яа-я0-9._ -]+", "_", stem)[:180]


def parse_document(filename: str, content_type: str | None, data: bytes, max_chars: int) -> ParsedDocument:
    extension = Path(filename).suffix.lower()
    metadata: dict[str, object] = {
        "filename": filename,
        "content_type": content_type or "application/octet-stream",
        "extension": extension,
        "bytes": len(data),
    }

    if extension in {".txt", ".md"}:
        text = _decode_text(data)
    elif extension == ".csv":
        text = _parse_csv(data)
    elif extension == ".pdf":
        text, pdf_meta = _parse_pdf(data)
        metadata.update(pdf_meta)
    elif extension == ".docx":
        text, doc_meta = _parse_docx(data)
        metadata.update(doc_meta)
    elif extension in {".xlsx", ".xlsm"}:
        text, xlsx_meta = _parse_xlsx(data)
        metadata.update(xlsx_meta)
    elif extension in {".png", ".jpg", ".jpeg"}:
        text = f"Изображение или схема: {filename}. OCR в MVP не выполняется, источник сохранен как визуальное вложение."
        metadata["image_only"] = True
    else:
        text = _decode_text(data)
        metadata["warning"] = "Формат не распознан, выполнена попытка текстового декодирования."

    text = _normalize_text(text)
    if len(text) > max_chars:
        metadata["truncated"] = True
        metadata["original_chars"] = len(text)
        text = text[:max_chars]
    metadata["chars"] = len(text)
    return ParsedDocument(text=text, metadata=metadata)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _parse_csv(data: bytes) -> str:
    decoded = _decode_text(data)
    sample = decoded[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    rows = []
    for idx, row in enumerate(csv.reader(io.StringIO(decoded), dialect=dialect)):
        if idx >= 500:
            rows.append("... CSV обрезан после 500 строк ...")
            break
        rows.append(" | ".join(cell.strip() for cell in row if cell is not None))
    return "\n".join(rows)


def _parse_pdf(data: bytes) -> tuple[str, dict[str, object]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page_no, page in enumerate(reader.pages[:80], start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[стр. {page_no}]\n{page_text}")
    meta = {"pages": len(reader.pages), "parsed_pages": min(len(reader.pages), 80)}
    return "\n\n".join(pages), meta


def _parse_docx(data: bytes) -> tuple[str, dict[str, object]]:
    from docx import Document

    document = Document(io.BytesIO(data))
    chunks: list[str] = []
    for paragraph in document.paragraphs:
        value = paragraph.text.strip()
        if value:
            chunks.append(value)
    for table_no, table in enumerate(document.tables, start=1):
        chunks.append(f"[таблица {table_no}]")
        for row in table.rows[:250]:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                chunks.append(" | ".join(cells))
    return "\n".join(chunks), {"paragraphs": len(document.paragraphs), "tables": len(document.tables)}


def _parse_xlsx(data: bytes) -> tuple[str, dict[str, object]]:
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    chunks: list[str] = []
    sheet_names = workbook.sheetnames
    for sheet_name in sheet_names[:12]:
        sheet = workbook[sheet_name]
        chunks.append(f"[лист: {sheet_name}]")
        for row_no, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            if row_no > 450:
                chunks.append("... лист обрезан после 450 строк ...")
                break
            values = [_cell_to_text(value) for value in row]
            if any(values):
                chunks.append(" | ".join(values))
    workbook.close()
    return "\n".join(chunks), {"sheets": sheet_names, "parsed_sheets": min(len(sheet_names), 12)}


def _cell_to_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

