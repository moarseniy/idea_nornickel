from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _normalize_base_url(value: str | None) -> str | None:
    if not value or not value.strip():
        return None
    normalized = value.strip().rstrip("/")
    if normalized.startswith(("http://", "https://")):
        return normalized
    return f"https://{normalized}"


@dataclass(frozen=True)
class Settings:
    app_name: str
    storage_dir: Path
    db_path: Path
    uploads_dir: Path
    sample_data_dir: Path
    openai_api_key: str
    openai_model: str
    openai_base_url: str | None
    openai_graph_extraction: bool
    openai_research_model: str
    openai_research_max_sources: int
    openai_max_retries: int
    openai_timeout: float
    vision_max_workers: int
    log_level: str
    max_document_chars: int
    max_context_chars: int
    max_upload_bytes: int
    pdf_ocr_enabled: bool
    pdf_ocr_languages: list[str]
    pdf_ocr_model_dir: Path
    pdf_fast_mode: bool
    pdf_max_pages: int
    pdf_image_max_pages: int
    pdf_ocr_min_chars: int
    pdf_text_layer_min_chars: int
    pdf_render_dpi: int
    pdf_vision_max_pages: int
    openai_vision_detail: str

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)


def load_settings() -> Settings:
    storage_dir = Path(os.getenv("STORAGE_DIR", "storage")).resolve()
    db_path = Path(os.getenv("DATABASE_PATH", str(storage_dir / "hypothesis_lab.db"))).resolve()
    uploads_dir = Path(os.getenv("UPLOADS_DIR", str(storage_dir / "uploads"))).resolve()
    sample_data_dir = Path(os.getenv("SAMPLE_DATA_DIR", "data")).resolve()
    pdf_ocr_model_dir = Path(os.getenv("PDF_OCR_MODEL_DIR", str(storage_dir / "easyocr"))).resolve()
    pdf_ocr_languages = [item.strip() for item in os.getenv("PDF_OCR_LANGUAGES", "ru,en,ch_sim").split(",") if item.strip()]
    openai_vision_detail = os.getenv("OPENAI_VISION_DETAIL", "high").strip().lower() or "high"
    if openai_vision_detail not in {"low", "high", "auto"}:
        openai_vision_detail = "high"
    return Settings(
        app_name=os.getenv("APP_NAME", "Hypothesis Lab"),
        storage_dir=storage_dir,
        db_path=db_path,
        uploads_dir=uploads_dir,
        sample_data_dir=sample_data_dir,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.2").strip() or "gpt-5.2",
        openai_base_url=_normalize_base_url(os.getenv("OPENAI_BASE_URL")),
        openai_graph_extraction=_as_bool(os.getenv("OPENAI_GRAPH_EXTRACTION"), True),
        openai_research_model=(os.getenv("OPENAI_RESEARCH_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.2")).strip() or "gpt-5.2",
        openai_research_max_sources=_as_int(os.getenv("OPENAI_RESEARCH_MAX_SOURCES"), 8),
        openai_max_retries=max(0, _as_int(os.getenv("OPENAI_MAX_RETRIES"), 5)),
        openai_timeout=max(1.0, _as_float(os.getenv("OPENAI_TIMEOUT"), 300.0)),
        vision_max_workers=max(1, _as_int(os.getenv("VISION_MAX_WORKERS"), 4)),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        max_document_chars=_as_int(os.getenv("MAX_DOCUMENT_CHARS"), 120_000),
        max_context_chars=_as_int(os.getenv("MAX_CONTEXT_CHARS"), 28_000),
        max_upload_bytes=_as_int(os.getenv("MAX_UPLOAD_BYTES"), 160 * 1024 * 1024),
        pdf_ocr_enabled=_as_bool(os.getenv("PDF_OCR_ENABLED"), True),
        pdf_ocr_languages=pdf_ocr_languages or ["ru", "en", "ch_sim"],
        pdf_ocr_model_dir=pdf_ocr_model_dir,
        pdf_fast_mode=_as_bool(os.getenv("PDF_FAST_MODE"), True),
        pdf_max_pages=_as_int(os.getenv("PDF_MAX_PAGES"), 80),
        pdf_image_max_pages=_as_int(os.getenv("PDF_IMAGE_MAX_PAGES"), 8),
        pdf_ocr_min_chars=_as_int(os.getenv("PDF_OCR_MIN_CHARS"), 12),
        pdf_text_layer_min_chars=_as_int(os.getenv("PDF_TEXT_LAYER_MIN_CHARS"), 16),
        pdf_render_dpi=_as_int(os.getenv("PDF_RENDER_DPI"), 120),
        pdf_vision_max_pages=_as_int(os.getenv("PDF_VISION_MAX_PAGES"), 2),
        openai_vision_detail=openai_vision_detail,
    )


settings = load_settings()
