from __future__ import annotations

import csv
import io
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any

import fitz
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import Database
from app.document_parser import SUPPORTED_EXTENSIONS, normalize_ocr_languages, parse_document, safe_filename
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
    ocr_languages: str | None = Form(default=None),
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
        ocr_languages=_request_ocr_languages(ocr_languages),
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
                ocr_languages=_request_ocr_languages(payload.ocr_languages),
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
    return _generate_hypotheses_core(project_id, payload, _actor(x_user))


@app.post("/api/projects/{project_id}/generate-with-files")
async def generate_hypotheses_with_files(
    project_id: str,
    payload_json: str = Form(...),
    files: list[UploadFile] | None = File(default=None),
    ocr_languages: str | None = Form(default=None),
    x_user: str | None = Header(default=None, alias="X-User"),
) -> dict[str, Any]:
    try:
        payload = GenerateRequest.model_validate_json(payload_json)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Некорректные параметры генерации: {exc}") from exc

    prompt_documents: list[dict[str, Any]] = []
    selected_ocr_languages = _request_ocr_languages(ocr_languages)
    for file in files or []:
        data = await file.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"Файл {file.filename} превышает MAX_UPLOAD_BYTES")
        prompt_documents.append(
            _parse_prompt_attachment(
                filename=file.filename or "prompt-attachment",
                content_type=file.content_type or "application/octet-stream",
                data=data,
                ocr_languages=selected_ocr_languages,
            )
        )
    return _generate_hypotheses_core(project_id, payload, _actor(x_user), prompt_documents=prompt_documents)


def _generate_hypotheses_core(
    project_id: str,
    payload: GenerateRequest,
    actor: str,
    prompt_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    project = _require_project(project_id)
    prompt_documents = prompt_documents or []
    documents = [*prompt_documents, *db.list_documents(project_id, include_text=True)]
    graph = db.get_graph(project_id)
    feedback = db.list_feedback(project_id)
    weights = {**project.get("settings", {}), **payload.weights}
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
    meta = {**meta, "prompt_attachments": len(prompt_documents)}
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
    project = _require_project(project_id)
    hypotheses = db.list_hypotheses(project_id)
    weights = _export_weights(project)
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
        values = _export_hypothesis_values(item, weights)
        writer.writerow(
            {
                **{key: values.get(key, item.get(key)) for key in writer.fieldnames},
                "score": f"{values['score']:.1f}",
                "novelty": f"{values['novelty']:.0f}",
                "feasibility": f"{values['feasibility']:.0f}",
                "impact": f"{values['impact']:.0f}",
                "risk": f"{values['risk']:.0f}",
            }
        )
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="hypotheses.csv"'},
    )


@app.get("/api/projects/{project_id}/export.md")
def export_markdown(project_id: str) -> Response:
    _require_project(project_id)
    data = db.export_project(project_id)
    content = _render_markdown_export(data)
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="hypotheses.md"'},
    )


@app.get("/api/projects/{project_id}/export.pdf")
def export_pdf(project_id: str) -> Response:
    _require_project(project_id)
    data = db.export_project(project_id)
    content = _render_pdf_export(data)
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="hypotheses.pdf"'},
    )


def _render_markdown_export(data: dict[str, Any]) -> str:
    project = data.get("project") or {}
    hypotheses = data.get("hypotheses") or []
    documents = data.get("documents") or []
    weights = _export_weights(project)
    lines = [
        f"# {project.get('name') or 'Hypothesis Lab'}",
        "",
        f"**Домен:** {project.get('domain') or '-'}",
        f"**Цель / KPI:** {project.get('goal') or '-'}",
        f"**Ограничения:** {project.get('constraints') or '-'}",
        "",
        f"_Источников: {len(documents)} · Гипотез: {len(hypotheses)}_",
        "",
        "## Ранжирование",
        "",
        "| # | Score | Статус | Гипотеза | Новизна | Реализ. | Эффект | Риск |",
        "|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for index, item in enumerate(hypotheses, start=1):
        values = _export_hypothesis_values(item, weights)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"{values['score']:.1f}",
                    _md_cell(values.get("status")),
                    _md_cell(values.get("title")),
                    f"{values['novelty']:.0f}",
                    f"{values['feasibility']:.0f}",
                    f"{values['impact']:.0f}",
                    f"{values['risk']:.0f}",
                ]
            )
            + " |"
        )
    for index, item in enumerate(hypotheses, start=1):
        values = _export_hypothesis_values(item, weights)
        lines.extend(
            [
                "",
                f"## {index}. {item.get('title') or 'Гипотеза'}",
                "",
                f"**Score:** {values['score']:.1f} · **Статус:** {item.get('status') or '-'}",
                "",
                item.get("statement") or "",
                "",
            ]
        )
        mechanism = item.get("mechanism") or item.get("rationale")
        if mechanism:
            lines.extend(["**Механизм / обоснование:**", "", str(mechanism), ""])
        if item.get("evidence"):
            lines.extend(["**Источники:**", ""])
            for evidence in item.get("evidence") or []:
                source = evidence.get("source") or "source"
                quote = evidence.get("quote") or evidence.get("why") or ""
                lines.append(f"- **{source}:** {quote}")
            lines.append("")
        if item.get("roadmap"):
            lines.extend(["**Дорожная карта:**", ""])
            for step in item.get("roadmap") or []:
                lines.append(f"- {step.get('step')}. {step.get('title')} -> {step.get('output')}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_pdf_export(data: dict[str, Any]) -> bytes:
    project = data.get("project") or {}
    hypotheses = data.get("hypotheses") or []
    documents = data.get("documents") or []
    weights = _export_weights(project)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 42

    def add_page() -> None:
        nonlocal page, y
        page = doc.new_page(width=595, height=842)
        y = 42

    def text_block(text: str, rect_height: float, *, size: float = 10.5, color: tuple[float, float, float] = (0.12, 0.12, 0.12)) -> None:
        nonlocal y
        if y + rect_height > 800:
            add_page()
        page.insert_textbox(
            fitz.Rect(42, y, 553, y + rect_height),
            str(text or ""),
            fontsize=size,
            fontname="helv",
            color=color,
            align=fitz.TEXT_ALIGN_LEFT,
        )
        y += rect_height + 6

    page.draw_rect(fitz.Rect(0, 0, 595, 842), color=(0.98, 0.96, 0.92), fill=(0.98, 0.96, 0.92))
    page.draw_rect(fitz.Rect(32, 32, 563, 116), color=(0.88, 0.58, 0.18), fill=(0.88, 0.58, 0.18))
    page.insert_textbox(
        fitz.Rect(48, 48, 547, 76),
        project.get("name") or "Hypothesis Lab",
        fontsize=18,
        fontname="helv",
        color=(0.05, 0.05, 0.04),
    )
    page.insert_textbox(
        fitz.Rect(48, 78, 547, 104),
        f"{len(documents)} источников · {len(hypotheses)} гипотез",
        fontsize=10,
        fontname="helv",
        color=(0.14, 0.12, 0.09),
    )
    y = 134
    text_block(f"Домен: {project.get('domain') or '-'}", 22, size=10.5)
    text_block(f"Цель / KPI: {project.get('goal') or '-'}", 44, size=10.5)

    for index, item in enumerate(hypotheses, start=1):
        values = _export_hypothesis_values(item, weights)
        if y + 134 > 800:
            add_page()
        page.draw_rect(fitz.Rect(36, y, 559, y + 124), color=(0.86, 0.82, 0.73), fill=(1, 0.99, 0.96), width=0.6)
        page.draw_rect(fitz.Rect(36, y, 41, y + 124), color=(0.88, 0.58, 0.18), fill=(0.88, 0.58, 0.18))
        page.insert_textbox(
            fitz.Rect(52, y + 12, 455, y + 42),
            f"{index}. {item.get('title') or 'Гипотеза'}",
            fontsize=12.5,
            fontname="helv",
            color=(0.08, 0.08, 0.07),
        )
        page.insert_textbox(
            fitz.Rect(470, y + 12, 545, y + 42),
            f"{values['score']:.1f}",
            fontsize=18,
            fontname="helv",
            color=(0.88, 0.58, 0.18),
            align=fitz.TEXT_ALIGN_RIGHT,
        )
        page.insert_textbox(
            fitz.Rect(52, y + 45, 545, y + 86),
            item.get("statement") or "",
            fontsize=9.5,
            fontname="helv",
            color=(0.18, 0.17, 0.15),
        )
        metrics = (
            f"status: {item.get('status') or '-'}   "
            f"novelty {values['novelty']:.0f} · "
            f"feasibility {values['feasibility']:.0f} · "
            f"impact {values['impact']:.0f} · "
            f"risk {values['risk']:.0f}"
        )
        page.insert_textbox(
            fitz.Rect(52, y + 92, 545, y + 114),
            metrics,
            fontsize=8.5,
            fontname="helv",
            color=(0.38, 0.35, 0.29),
        )
        y += 138

    payload = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return payload


def _export_weights(project: dict[str, Any]) -> dict[str, float]:
    settings_weights = project.get("settings") if isinstance(project, dict) else {}
    return {
        "novelty": float((settings_weights or {}).get("novelty", 0.25)),
        "feasibility": float((settings_weights or {}).get("feasibility", 0.25)),
        "impact": float((settings_weights or {}).get("impact", 0.35)),
        "risk": float((settings_weights or {}).get("risk", 0.15)),
    }


def _export_hypothesis_values(item: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    novelty = _metric_for_export(item.get("novelty"))
    feasibility = _metric_for_export(item.get("feasibility"))
    impact = _metric_for_export(item.get("impact"))
    risk = _metric_for_export(item.get("risk"))
    raw_values = [float(item.get(key) or 0) for key in ("novelty", "feasibility", "impact", "risk")]
    looks_unit_scaled = any(0 < value <= 1 for value in raw_values) and all(value <= 1 for value in raw_values)
    score = (
        novelty * weights["novelty"]
        + feasibility * weights["feasibility"]
        + impact * weights["impact"]
        + (100 - risk) * weights["risk"]
        if looks_unit_scaled
        else float(item.get("score") or 0)
    )
    return {
        **item,
        "score": score,
        "novelty": novelty,
        "feasibility": feasibility,
        "impact": impact,
        "risk": risk,
    }


def _metric_for_export(value: Any) -> float:
    number = float(value or 0)
    return number * 100 if 0 < number <= 1 else number


def _md_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _ingest_document_bytes(
    project_id: str,
    filename: str,
    content_type: str,
    data: bytes,
    actor: str,
    ocr_languages: list[str] | None = None,
) -> dict[str, Any]:
    display_name = safe_filename(filename)
    selected_ocr_languages = ocr_languages or settings.pdf_ocr_languages
    parsed = parse_document(
        display_name,
        content_type,
        data,
        settings.max_document_chars,
        pdf_ocr_enabled=settings.pdf_ocr_enabled,
        pdf_ocr_languages=selected_ocr_languages,
        pdf_ocr_model_dir=settings.pdf_ocr_model_dir,
        pdf_ocr_min_chars=settings.pdf_ocr_min_chars,
        pdf_text_layer_min_chars=settings.pdf_text_layer_min_chars,
        pdf_render_dpi=settings.pdf_render_dpi,
        pdf_vision_max_pages=settings.pdf_vision_max_pages,
    )
    vision_meta = _enrich_vision_images(parsed, selected_ocr_languages)
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


def _parse_prompt_attachment(
    filename: str,
    content_type: str,
    data: bytes,
    ocr_languages: list[str] | None = None,
) -> dict[str, Any]:
    display_name = safe_filename(filename)
    selected_ocr_languages = ocr_languages or settings.pdf_ocr_languages
    parsed = parse_document(
        display_name,
        content_type,
        data,
        settings.max_document_chars,
        pdf_ocr_enabled=settings.pdf_ocr_enabled,
        pdf_ocr_languages=selected_ocr_languages,
        pdf_ocr_model_dir=settings.pdf_ocr_model_dir,
        pdf_ocr_min_chars=settings.pdf_ocr_min_chars,
        pdf_text_layer_min_chars=settings.pdf_text_layer_min_chars,
        pdf_render_dpi=settings.pdf_render_dpi,
        pdf_vision_max_pages=settings.pdf_vision_max_pages,
    )
    vision_meta = _enrich_vision_images(parsed, selected_ocr_languages)
    return {
        "id": f"prompt:{uuid.uuid4()}",
        "filename": f"[тестовый промпт] {display_name}",
        "content_type": content_type,
        "text": parsed.text,
        "metadata": {**parsed.metadata, "prompt_attachment": True, "vision": vision_meta},
        "path": None,
    }


def _enrich_vision_images(parsed: Any, ocr_languages: list[str]) -> dict[str, Any]:
    if not parsed.vision_images:
        return {"mode": "not_needed", "count": 0}

    items: list[dict[str, Any]] = []
    vision_chunks: list[str] = []
    for image in parsed.vision_images:
        vision_text, vision_meta = ai.describe_image(image.data, image.label, image.content_type, ocr_languages=ocr_languages)
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


def _request_ocr_languages(value: str | list[str] | None) -> list[str] | None:
    languages = normalize_ocr_languages(value, default=[])
    return languages or None


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
