from __future__ import annotations

import json
import re
from typing import Any

from app.config import Settings
from app.knowledge import edge_id, score_hypothesis, stable_id


class OpenAIServiceError(RuntimeError):
    """Raised when an OpenAI-backed action cannot be completed."""


class OpenAIService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.openai_enabled

    def generate_hypotheses(
        self,
        project: dict[str, Any],
        documents: list[dict[str, Any]],
        graph: dict[str, list[dict[str, Any]]],
        feedback: list[dict[str, Any]],
        count: int,
        weights: dict[str, float],
        exclusions: list[str],
        include_roadmap: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.enabled:
            raise OpenAIServiceError("OPENAI_API_KEY не задан. Генерация гипотез требует OpenAI API.")

        prompt = _build_hypothesis_prompt(project, documents, graph, feedback, count, weights, exclusions, include_roadmap, self.settings.max_context_chars)
        instructions = (
            "Ты исследовательский AI-agent для генерации научно-исследовательских гипотез в материаловедении, "
            "обогащении и металлургии. Формируй конкретные проверяемые гипотезы, не выдумывай источники, "
            "явно отделяй факты из контекста от допущений. Ответ верни строго валидным JSON без markdown."
        )
        try:
            text = self._call_text(instructions=instructions, prompt=prompt, max_output_tokens=6000)
            payload = _extract_json(text)
            hypotheses = payload.get("hypotheses", []) if isinstance(payload, dict) else []
            normalized = [_normalize_hypothesis(item, weights) for item in hypotheses[:count] if isinstance(item, dict)]
            if not normalized:
                raise ValueError("OpenAI response did not contain hypotheses")
            return normalized, {"mode": "openai", "model": self.settings.openai_model}
        except Exception as exc:  # noqa: BLE001
            raise OpenAIServiceError(f"OpenAI generation failed: {exc}") from exc

    def extract_graph(self, document_text: str, source_id: str, source_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        if not self.enabled or not self.settings.openai_graph_extraction or len(document_text.strip()) < 80:
            return [], [], {"mode": "skipped"}

        prompt = f"""
Источник: {source_name}

Извлеки из текста компактный граф знаний для исследовательской системы. Нужны только предметные сущности:
материалы, процессы, реагенты, свойства, метрики, оборудование, риски, наблюдения.

Верни JSON:
{{
  "nodes": [{{"label": "...", "type": "material|process|reagent|property|metric|equipment|risk|observation", "summary": "..."}}],
  "edges": [{{"source": "label узла", "target": "label узла", "relation": "influences|used_in|processed_by|measured_by|constrained_by|associated_with", "evidence": "короткий фрагмент", "confidence": 0.0}}]
}}

Текст:
{document_text[: self.settings.max_context_chars]}
""".strip()
        instructions = "Извлекай только явно поддержанные текстом связи. Ответ строго JSON без markdown."
        try:
            text = self._call_text(instructions=instructions, prompt=prompt, max_output_tokens=3000)
            payload = _extract_json(text)
            raw_nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
            raw_edges = payload.get("edges", []) if isinstance(payload, dict) else []
            label_to_id: dict[str, str] = {}
            nodes: list[dict[str, Any]] = []
            for raw in raw_nodes[:48]:
                if not isinstance(raw, dict):
                    continue
                label = str(raw.get("label") or "").strip()
                kind = str(raw.get("type") or "concept").strip().lower()
                if not label:
                    continue
                nid = stable_id(kind, label)
                label_to_id[label.lower()] = nid
                nodes.append(
                    {
                        "id": nid,
                        "label": label[:100],
                        "type": kind,
                        "summary": str(raw.get("summary") or "")[:400],
                        "weight": 2.8,
                        "source_ids": [source_id],
                    }
                )
            edges: list[dict[str, Any]] = []
            for raw in raw_edges[:80]:
                if not isinstance(raw, dict):
                    continue
                source_label = str(raw.get("source") or "").strip().lower()
                target_label = str(raw.get("target") or "").strip().lower()
                source = label_to_id.get(source_label)
                target = label_to_id.get(target_label)
                if not source or not target or source == target:
                    continue
                relation = str(raw.get("relation") or "associated_with").strip()
                confidence = _as_float(raw.get("confidence"), 0.65)
                edges.append(
                    {
                        "id": edge_id(source, target, relation),
                        "source": source,
                        "target": target,
                        "relation": relation,
                        "evidence": str(raw.get("evidence") or "")[:240],
                        "weight": 1.5 + confidence,
                        "source_ids": [source_id],
                    }
                )
            return nodes, edges, {"mode": "openai", "model": self.settings.openai_model}
        except Exception as exc:  # noqa: BLE001
            return [], [], {"mode": "failed", "reason": str(exc)}

    def chat(
        self,
        project: dict[str, Any],
        hypotheses: list[dict[str, Any]],
        graph: dict[str, list[dict[str, Any]]],
        feedback: list[dict[str, Any]],
        chat_history: list[dict[str, Any]],
        message: str,
    ) -> tuple[str, dict[str, Any]]:
        if not self.enabled:
            raise OpenAIServiceError("OPENAI_API_KEY не задан. Чат требует OpenAI API.")

        prompt = _build_chat_prompt(project, hypotheses, graph, feedback, chat_history, message, self.settings.max_context_chars)
        instructions = (
            "Ты интерактивный научный ассистент проекта. Отвечай по-русски, кратко, предметно. "
            "Если эксперт корректирует систему, явно сформулируй, как это должно повлиять на следующие гипотезы, критерии или граф знаний. "
            "Не притворяйся, что провел лабораторные опыты."
        )
        try:
            text = self._call_text(instructions=instructions, prompt=prompt, max_output_tokens=2200)
            return text.strip(), {"mode": "openai", "model": self.settings.openai_model}
        except Exception as exc:  # noqa: BLE001
            raise OpenAIServiceError(f"OpenAI chat failed: {exc}") from exc

    def _call_text(self, instructions: str, prompt: str, max_output_tokens: int) -> str:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": self.settings.openai_api_key}
        if self.settings.openai_base_url:
            kwargs["base_url"] = self.settings.openai_base_url
        client = OpenAI(**kwargs)
        response = client.responses.create(
            model=self.settings.openai_model,
            instructions=instructions,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        return _response_to_text(response)


def _build_hypothesis_prompt(
    project: dict[str, Any],
    documents: list[dict[str, Any]],
    graph: dict[str, list[dict[str, Any]]],
    feedback: list[dict[str, Any]],
    count: int,
    weights: dict[str, float],
    exclusions: list[str],
    include_roadmap: bool,
    max_chars: int,
) -> str:
    docs_context = _documents_context(documents, max_chars)
    graph_context = _graph_context(graph)
    feedback_context = _feedback_context(feedback)
    return f"""
Проект: {project.get("name")}
Домен: {project.get("domain")}
Цель/KPI: {project.get("goal") or "не задано"}
Ограничения: {project.get("constraints") or "не заданы"}
Веса ранжирования: {weights}
Исключить направления: {", ".join(exclusions) if exclusions else "нет"}
Нужна дорожная карта: {include_roadmap}

Контекст источников:
{docs_context}

Сводка графа знаний:
{graph_context}

Экспертный фидбэк:
{feedback_context}

Сгенерируй {count} гипотез. Каждая гипотеза должна быть проверяемой лабораторно, с механизмом, рисками,
ожидаемой ценностью и источниками из контекста.

JSON-схема ответа:
{{
  "hypotheses": [
    {{
      "title": "короткое название",
      "statement": "проверяемое утверждение в формате если/то",
      "rationale": "обоснование",
      "mechanism": "ожидаемый механизм влияния",
      "novelty": 0,
      "feasibility": 0,
      "impact": 0,
      "risk": 0,
      "uncertainty": "что неизвестно и как это влияет на доверие",
      "evidence": [{{"source": "имя файла", "quote": "короткая цитата или пересказ", "why": "зачем источник релевантен"}}],
      "roadmap": [{{"step": 1, "title": "шаг проверки", "output": "артефакт или критерий"}}]
    }}
  ]
}}
""".strip()


def _build_chat_prompt(
    project: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    graph: dict[str, list[dict[str, Any]]],
    feedback: list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
    message: str,
    max_chars: int,
) -> str:
    top_hypotheses = "\n".join(
        f"- #{idx}: {item.get('title')} | score={item.get('score')} | status={item.get('status')}\n  {item.get('statement')}"
        for idx, item in enumerate(hypotheses[:8], start=1)
    )
    history = "\n".join(f"{item.get('role')}({item.get('actor')}): {item.get('content')}" for item in chat_history[-10:])
    return f"""
Проект: {project.get("name")}
Цель: {project.get("goal") or "не задано"}
Ограничения: {project.get("constraints") or "не заданы"}

Топ гипотез:
{top_hypotheses or "пока нет гипотез"}

Граф:
{_graph_context(graph)}

Фидбэк:
{_feedback_context(feedback)}

История чата:
{history[-max_chars // 3:]}

Сообщение пользователя:
{message}
""".strip()


def _documents_context(documents: list[dict[str, Any]], max_chars: int) -> str:
    chunks: list[str] = []
    budget = max_chars
    for doc in documents[:10]:
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        excerpt = text[: min(3500, budget)]
        chunks.append(f"### {doc.get('filename')}\n{excerpt}")
        budget -= len(excerpt)
        if budget <= 1000:
            break
    return "\n\n".join(chunks) or "Источники пока не загружены."


def _graph_context(graph: dict[str, list[dict[str, Any]]]) -> str:
    nodes = graph.get("nodes", [])[:28]
    edges = graph.get("edges", [])[:36]
    node_lines = [f"{node.get('label')} ({node.get('type')}, w={node.get('weight')})" for node in nodes]
    id_to_label = {node.get("id"): node.get("label") for node in graph.get("nodes", [])}
    edge_lines = [
        f"{id_to_label.get(edge.get('source'), edge.get('source'))} -[{edge.get('relation')}]-> {id_to_label.get(edge.get('target'), edge.get('target'))}"
        for edge in edges
    ]
    return "Узлы: " + "; ".join(node_lines) + "\nСвязи: " + "; ".join(edge_lines)


def _feedback_context(feedback: list[dict[str, Any]]) -> str:
    if not feedback:
        return "Нет."
    return "\n".join(
        f"- {item.get('actor')}: rating={item.get('rating')}, outcome={item.get('outcome')}, {item.get('comment')}"
        for item in feedback[:12]
    )


def _normalize_hypothesis(item: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    novelty = _as_float(item.get("novelty"), 50)
    feasibility = _as_float(item.get("feasibility"), 50)
    impact = _as_float(item.get("impact"), 50)
    risk = _as_float(item.get("risk"), 50)
    return {
        "title": str(item.get("title") or "Гипотеза")[:180],
        "statement": str(item.get("statement") or "")[:1600],
        "rationale": str(item.get("rationale") or "")[:2200],
        "mechanism": str(item.get("mechanism") or "")[:1600],
        "novelty": novelty,
        "feasibility": feasibility,
        "impact": impact,
        "risk": risk,
        "score": _as_float(item.get("score"), score_hypothesis(novelty, feasibility, impact, risk, weights)),
        "uncertainty": str(item.get("uncertainty") or "")[:1200],
        "evidence": item.get("evidence") if isinstance(item.get("evidence"), list) else [],
        "roadmap": item.get("roadmap") if isinstance(item.get("roadmap"), list) else [],
    }


def _extract_json(text: str) -> dict[str, Any]:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?", "", clean).strip()
    clean = re.sub(r"```$", "", clean).strip()
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(clean[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    return payload


def _response_to_text(response: Any) -> str:
    try:
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        if chunks:
            return "\n".join(chunks)
    except Exception:  # noqa: BLE001
        pass
    return str(response)


def _as_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return default


