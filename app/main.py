from __future__ import annotations

import csv
import io
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
        "graph_extraction": settings.openai_graph_extraction,
    }


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

    for path in sorted(settings.sample_data_dir.rglob("*")):
        if len(imported) >= payload.max_files:
            break
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS or path.suffix.lower() not in extensions:
            continue
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
    parsed = parse_document(display_name, content_type, data, settings.max_document_chars)
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
    return {"document": document, "graph_update": graph_update, "extractor": extractor_meta}


def _require_project(project_id: str) -> dict[str, Any]:
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден")
    return project


def _actor(value: str | None) -> str:
    actor = (value or "researcher").strip()
    return actor[:80] or "researcher"
