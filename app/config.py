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
    log_level: str
    max_document_chars: int
    max_context_chars: int
    max_upload_bytes: int

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)


def load_settings() -> Settings:
    storage_dir = Path(os.getenv("STORAGE_DIR", "storage")).resolve()
    db_path = Path(os.getenv("DATABASE_PATH", str(storage_dir / "hypothesis_lab.db"))).resolve()
    uploads_dir = Path(os.getenv("UPLOADS_DIR", str(storage_dir / "uploads"))).resolve()
    sample_data_dir = Path(os.getenv("SAMPLE_DATA_DIR", "data")).resolve()
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
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        max_document_chars=_as_int(os.getenv("MAX_DOCUMENT_CHARS"), 120_000),
        max_context_chars=_as_int(os.getenv("MAX_CONTEXT_CHARS"), 28_000),
        max_upload_bytes=_as_int(os.getenv("MAX_UPLOAD_BYTES"), 35 * 1024 * 1024),
    )


settings = load_settings()
