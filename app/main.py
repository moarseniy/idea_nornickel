from __future__ import annotations

import csv
import io
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import Database
from app.document_parser import SUPPORTED_EXTENSIONS, parse_document, safe_filename
from app.knowledge import graph_from_hypothesis, heuristic_graph_from_text
from app.openai_service import OpenAIService, OpenAIServiceError
from app.schemas import (
    ChatRequest,
    FeedbackRequest,
    GenerateRequest,
    ProjectCreate,
    ProjectUpdate,
    SampleImportRequest,
    StatusUpdate,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("hypothesis_lab")

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

db = Database(settings.db_path)
ai = OpenAIService(settings)


@app.on_event("startup")
def startup() -> None:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    db.initialize()
    logger.info(
        "Hypothesis Lab started: model=%s base_url=%s openai_key_present=%s graph_extraction=%s pdf_ocr=%s ocr_langs=%s storage=%s",
        settings.openai_model,
        settings.openai_base_url or "default",
        bool(settings.openai_api_key),
        settings.openai_graph_extraction,
        settings.pdf_ocr_enabled,
        ",".join(settings.pdf_ocr_languages),
        settings.storage_dir,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "app": settings.app_name,
        "openai_enabled": ai.enabled,
        "openai_model": settings.openai_model,
        "openai_base_url": settings.openai_base_url or "default",
        "graph_extraction": settings.openai_graph_extraction,
        "pdf_ocr_enabled": settings.pdf_ocr_enabled,
        "pdf_ocr_languages": settings.pdf_ocr_languages,
        "pdf_text_layer_min_chars": settings.pdf_text_layer_min_chars,
        "pdf_vision_max_pages": settings.pdf_vision_max_pages,
    }


@app.post("/api/openai/check")
def check_openai() -> dict[str, Any]:
    try:
        return ai.check_connection()
    except OpenAIServiceError as exc:
        logger.exception(
            "OpenAI connectivity check failed: model=%s base_url=%s error=%s",
            settings.openai_model,
            settings.openai_base_url or "default",
            exc,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/projects")
def list_projects() -> dict[str, Any]:
    return {"projects": db.list_projects()}


@app.post("/api/projects")
def create_project(payload: ProjectCreate, x_user: str | None = Header(default=None, alias="X-User")) -> dict[str, Any]:
    project = db.create_project(payload.model_dump(), _actor(x_user))
    return {"project": project}


@app.patch("/api/projects/{project_id}")
def update_project(
    project_id: str,
    payload: ProjectUpdate,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    _require_project(project_id)
    project = db.update_project(project_id, payload.model_dump(exclude_unset=True), _actor(x_user))
    return {"project": project}


@app.get("/api/projects/{project_id}/state")
def project_state(project_id: str) -> dict[str, Any]:
    project = _require_project(project_id)
    return {
        "project": project,
        "documents": db.list_documents(project_id, include_text=False),
        "hypotheses": db.list_hypotheses(project_id),
        "graph": db.get_graph(project_id),
        "feedback": db.list_feedback(project_id),
        "chat": db.list_chat(project_id),
        "events": db.list_events(project_id),
        "runtime": {
            "openai_enabled": ai.enabled,
            "openai_model": settings.openai_model,
        },
    }


@app.post("/api/projects/{project_id}/documents")
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    _require_project(project_id)
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Файл превышает MAX_UPLOAD_BYTES")
    result = _ingest_document_bytes(
        project_id=project_id,
        filename=file.filename or "document",
        content_type=file.content_type or "application/octet-stream",
        data=data,
        actor=_actor(x_user),
    )
    return result


@app.post("/api/projects/{project_id}/documents/import-samples")
def import_samples(
    project_id: str,
    payload: SampleImportRequest,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    _require_project(project_id)
    if not settings.sample_data_dir.exists():
        raise HTTPException(status_code=404, detail=f"Папка с данными не найдена: {settings.sample_data_dir}")

    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in payload.extensions}
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    actor = _actor(x_user)
    sample_paths = [
        path
        for path in settings.sample_data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and path.suffix.lower() in extensions
    ]

    for path in _ordered_sample_paths(sample_paths):
        if len(imported) >= payload.max_files:
            break
        if path.stat().st_size > settings.max_upload_bytes:
            skipped.append({"path": str(path), "reason": "too_large"})
            continue
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            result = _ingest_document_bytes(
                project_id=project_id,
                filename=str(path.relative_to(settings.sample_data_dir)),
                content_type=content_type,
                data=path.read_bytes(),
                actor=actor,
            )
            imported.append(result["document"])
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": str(path), "reason": str(exc)})

    return {"imported": imported, "skipped": skipped}


@app.post("/api/projects/{project_id}/generate")
def generate_hypotheses(
    project_id: str,
    payload: GenerateRequest,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    project = _require_project(project_id)
    documents = db.list_documents(project_id, include_text=True)
    graph = db.get_graph(project_id)
    feedback = db.list_feedback(project_id)
    weights = {**project.get("settings", {}), **payload.weights}
    actor = _actor(x_user)
    try:
        hypotheses, meta = ai.generate_hypotheses(
            project=project,
            documents=documents,
            graph=graph,
            feedback=feedback,
            count=payload.count,
            weights=weights,
            exclusions=payload.exclusions,
            include_roadmap=payload.include_roadmap,
        )
    except OpenAIServiceError as exc:
        logger.exception(
            "Hypothesis generation failed: project_id=%s model=%s actor=%s documents=%s graph_nodes=%s graph_edges=%s feedback=%s count=%s error=%s",
            project_id,
            settings.openai_model,
            actor,
            len(documents),
            len(graph.get("nodes", [])),
            len(graph.get("edges", [])),
            len(feedback),
            payload.count,
            exc,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    created = db.add_hypotheses(project_id, hypotheses, actor)
    for hypothesis in created:
        nodes, edges = graph_from_hypothesis(hypothesis)
        db.upsert_graph(project_id, nodes, edges, actor, "hypothesis-generated", source_id=hypothesis["id"])
    return {"hypotheses": created, "meta": meta, "state": project_state(project_id)}


@app.post("/api/projects/{project_id}/chat")
def chat(
    project_id: str,
    payload: ChatRequest,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    project = _require_project(project_id)
    actor = _actor(x_user)
    try:
        answer, meta = ai.chat(
            project=project,
            hypotheses=db.list_hypotheses(project_id),
            graph=db.get_graph(project_id),
            feedback=db.list_feedback(project_id),
            chat_history=db.list_chat(project_id),
            message=payload.message,
        )
    except OpenAIServiceError as exc:
        logger.exception(
            "Chat failed: project_id=%s model=%s actor=%s message_chars=%s error=%s",
            project_id,
            settings.openai_model,
            actor,
            len(payload.message),
            exc,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    user_message = db.add_chat_message(project_id, "user", actor, payload.message, event=True)
    assistant_message = db.add_chat_message(project_id, "assistant", "AI", answer)
    return {"messages": [user_message, assistant_message], "meta": meta, "state": project_state(project_id)}


@app.patch("/api/projects/{project_id}/hypotheses/{hypothesis_id}/status")
def update_hypothesis_status(
    project_id: str,
    hypothesis_id: str,
    payload: StatusUpdate,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    _require_project(project_id)
    hypothesis = db.update_hypothesis_status(project_id, hypothesis_id, payload.status, _actor(x_user))
    if not hypothesis:
        raise HTTPException(status_code=404, detail="Гипотеза не найдена")
    return {"hypothesis": hypothesis, "state": project_state(project_id)}


@app.post("/api/projects/{project_id}/hypotheses/{hypothesis_id}/feedback")
def add_hypothesis_feedback(
    project_id: str,
    hypothesis_id: str,
    payload: FeedbackRequest,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    _require_project(project_id)
    if not db.get_hypothesis(hypothesis_id):
        raise HTTPException(status_code=404, detail="Гипотеза не найдена")
    feedback = db.add_feedback(
        project_id=project_id,
        hypothesis_id=hypothesis_id,
        actor=_actor(x_user),
        rating=payload.rating,
        outcome=payload.outcome,
        comment=payload.comment,
    )
    return {"feedback": feedback, "state": project_state(project_id)}


@app.get("/api/projects/{project_id}/export.json")
def export_json(project_id: str) -> JSONResponse:
    _require_project(project_id)
    return JSONResponse(db.export_project(project_id))


@app.get("/api/projects/{project_id}/export.csv")
def export_csv(project_id: str) -> Response:
    _require_project(project_id)
    hypotheses = db.list_hypotheses(project_id)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "id",
            "title",
            "statement",
            "score",
            "novelty",
            "feasibility",
            "impact",
            "risk",
            "status",
            "created_at",
        ],
    )
    writer.writeheader()
    for item in hypotheses:
        writer.writerow({key: item.get(key) for key in writer.fieldnames})
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="hypotheses.csv"'},
    )


def _ingest_document_bytes(
    project_id: str,
    filename: str,
    content_type: str,
    data: bytes,
    actor: str,
) -> dict[str, Any]:
    display_name = safe_filename(filename)
    parsed = parse_document(
        display_name,
        content_type,
        data,
        settings.max_document_chars,
        pdf_ocr_enabled=settings.pdf_ocr_enabled,
        pdf_ocr_languages=settings.pdf_ocr_languages,
        pdf_ocr_model_dir=settings.pdf_ocr_model_dir,
        pdf_ocr_min_chars=settings.pdf_ocr_min_chars,
        pdf_text_layer_min_chars=settings.pdf_text_layer_min_chars,
        pdf_render_dpi=settings.pdf_render_dpi,
        pdf_vision_max_pages=settings.pdf_vision_max_pages,
    )
    vision_meta = _enrich_vision_images(parsed)
    upload_dir = settings.uploads_dir / project_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}_{display_name}"
    stored_path = upload_dir / stored_name
    stored_path.write_bytes(data)

    document = db.add_document(
        project_id=project_id,
        filename=display_name,
        content_type=content_type,
        text=parsed.text,
        metadata=parsed.metadata,
        path=str(stored_path),
        actor=actor,
    )
    nodes, edges = heuristic_graph_from_text(parsed.text, document["id"], document["filename"])
    llm_nodes, llm_edges, extractor_meta = ai.extract_graph(parsed.text, document["id"], document["filename"])
    graph_update = db.upsert_graph(
        project_id,
        [*nodes, *llm_nodes],
        [*edges, *llm_edges],
        actor,
        "document-ingested",
        source_id=document["id"],
    )
    return {"document": document, "graph_update": graph_update, "extractor": extractor_meta, "vision": vision_meta}


def _enrich_vision_images(parsed: Any) -> dict[str, Any]:
    if not parsed.vision_images:
        return {"mode": "not_needed", "count": 0}

    items: list[dict[str, Any]] = []
    vision_chunks: list[str] = []
    for image in parsed.vision_images:
        vision_text, vision_meta = ai.describe_image(image.data, image.label, image.content_type)
        item_meta = {
            "label": image.label,
            "reason": image.reason,
            "page_no": image.page_no,
            **vision_meta,
        }
        items.append(item_meta)
        if vision_text:
            vision_chunks.append(f"[{image.label}]\n{vision_text}")
            logger.info("Vision extracted text: label=%s chars=%s", image.label, len(vision_text))
        elif vision_meta.get("mode") == "failed":
            logger.warning("Vision failed: label=%s reason=%s", image.label, vision_meta.get("reason"))

    vision_summary = {
        "mode": "processed",
        "count": len(parsed.vision_images),
        "text_items": len(vision_chunks),
        "items": items,
    }
    parsed.metadata["vision"] = vision_summary
    if vision_chunks:
        separator = "\n\n" if parsed.text else ""
        combined = f"{parsed.text}{separator}Vision-анализ OpenAI:\n" + "\n\n".join(vision_chunks)
        if len(combined) > settings.max_document_chars:
            parsed.metadata["truncated"] = True
            parsed.metadata["original_chars"] = len(combined)
            combined = combined[: settings.max_document_chars]
        parsed.text = combined
        parsed.metadata["chars"] = len(parsed.text)
    return vision_summary


def _ordered_sample_paths(paths: list[Path]) -> list[Path]:
    buckets: dict[str, list[Path]] = {
        "regulation_images": [],
        "scheme_images": [],
        "core_documents": [],
        "technical_pdfs": [],
        "other": [],
    }
    for path in paths:
        buckets[_sample_bucket(path)].append(path)

    for bucket_paths in buckets.values():
        bucket_paths.sort(key=_sample_path_key)

    ordered: list[Path] = []
    cycle = ("regulation_images", "scheme_images", "core_documents", "regulation_images", "scheme_images", "core_documents", "technical_pdfs")
    while any(buckets.values()):
        before = len(ordered)
        for bucket in cycle:
            if buckets[bucket]:
                ordered.append(buckets[bucket].pop(0))
        if len(ordered) == before:
            ordered.extend(buckets["other"])
            buckets["other"] = []
    return ordered


def _sample_bucket(path: Path) -> str:
    path_key = _sample_path_key(path)
    extension = path.suffix.lower()
    if extension in IMAGE_EXTENSIONS and "регламенты/" in path_key:
        return "regulation_images"
    if extension in IMAGE_EXTENSIONS and "схемы флотации/" in path_key:
        return "scheme_images"
    if extension in {".docx", ".xlsx", ".xlsm", ".txt", ".md", ".csv"}:
        return "core_documents"
    if extension == ".pdf" and "дополнительные материалы/" not in path_key:
        return "technical_pdfs"
    return "other"


def _sample_path_key(path: Path) -> str:
    try:
        relative = path.relative_to(settings.sample_data_dir)
    except ValueError:
        relative = path
    return "/".join(part.casefold() for part in relative.parts)


def _require_project(project_id: str) -> dict[str, Any]:
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден")
    return project


def _actor(value: str | None) -> str:
    actor = (value or "researcher").strip()
    return actor[:80] or "researcher"
