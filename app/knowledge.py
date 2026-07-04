from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter
from typing import Any


DOMAIN_TERMS: list[tuple[str, str, tuple[str, ...]]] = [
    ("material", "хвосты", ("tailings", "хвост")),
    ("material", "руда", ("ore", "сырье")),
    ("material", "концентрат", ("concentrate",)),
    ("material", "медь", ("copper", "cu")),
    ("material", "никель", ("nickel", "ni")),
    ("material", "золото", ("gold", "au")),
    ("material", "серебро", ("silver", "ag")),
    ("material", "платина", ("platinum", "pt", "платиноид")),
    ("process", "флотация", ("flotation", "флот")),
    ("process", "измельчение", ("grinding", "доизмельчение", "помол")),
    ("process", "классификация", ("гидроциклон", "classification")),
    ("process", "выщелачивание", ("leaching", "цианирование")),
    ("process", "окислительный обжиг", ("roasting", "обжиг")),
    ("process", "магнитная сепарация", ("magnetic separation",)),
    ("reagent", "собиратель", ("collector", "ксантогенат", "xanthate")),
    ("reagent", "пенообразователь", ("frother", "пена")),
    ("reagent", "депрессор", ("depressant", "подавитель")),
    ("reagent", "активатор", ("activator", "медный купорос", "cuso4")),
    ("reagent", "известь", ("lime", "cao")),
    ("property", "извлечение", ("recovery", "извлеч")),
    ("property", "содержание", ("grade", "массовая доля")),
    ("property", "крупность", ("particle size", "мкм", "класс крупности")),
    ("property", "pH", ("кислотность",)),
    ("property", "расход реагента", ("дозировка", "кг/т", "g/t")),
    ("metric", "экономический эффект", ("себестоимость", "opex", "capex", "стоимость")),
    ("risk", "шламование", ("slime", "тонкие шламы")),
    ("risk", "переизмельчение", ("overgrinding",)),
    ("equipment", "флотомашина", ("машина флотации", "камера флотации")),
    ("equipment", "мельница", ("mill", "мельницы")),
]

NUMERIC_RE = re.compile(r"\b\d+(?:[,.]\d+)?\s?(?:%|г/т|кг/т|мкм|мм|т/ч|ppm|pH)\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?。;])\s+|\n+")


def stable_id(kind: str, label: str) -> str:
    digest = hashlib.sha1(f"{kind}:{label.lower()}".encode("utf-8")).hexdigest()[:12]
    return f"{kind}:{digest}"


def edge_id(source: str, target: str, relation: str) -> str:
    digest = hashlib.sha1(f"{source}|{target}|{relation}".encode("utf-8")).hexdigest()[:16]
    return f"edge:{digest}"


def score_hypothesis(novelty: float, feasibility: float, impact: float, risk: float, weights: dict[str, float] | None = None) -> float:
    weights = weights or {}
    novelty_w = float(weights.get("novelty", 0.25))
    feasibility_w = float(weights.get("feasibility", 0.25))
    impact_w = float(weights.get("impact", 0.35))
    risk_w = float(weights.get("risk", 0.15))
    total = novelty_w + feasibility_w + impact_w + risk_w or 1
    raw = novelty * novelty_w + feasibility * feasibility_w + impact * impact_w + (100 - risk) * risk_w
    return round(raw / total, 1)


def heuristic_graph_from_text(text: str, source_id: str, source_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    source_node_id = f"source:{source_id}"
    nodes[source_node_id] = {
        "id": source_node_id,
        "label": source_name[:80],
        "type": "source",
        "summary": "Загруженный источник",
        "weight": 1.0,
        "source_ids": [source_id],
    }

    lower_text = text.lower()
    for kind, label, aliases in DOMAIN_TERMS:
        aliases_full = (label, *aliases)
        count = sum(lower_text.count(alias.lower()) for alias in aliases_full)
        if count:
            nid = stable_id(kind, label)
            nodes[nid] = {
                "id": nid,
                "label": label,
                "type": kind,
                "summary": f"Упомянуто в источнике {source_name}",
                "weight": min(8.0, 1.0 + math.log(count + 1)),
                "source_ids": [source_id],
            }
            eid = edge_id(source_node_id, nid, "contains")
            edges[eid] = {
                "id": eid,
                "source": source_node_id,
                "target": nid,
                "relation": "contains",
                "evidence": f"{source_name}: {count} упомин.",
                "weight": min(6.0, 1.0 + math.log(count + 1)),
                "source_ids": [source_id],
            }

    numeric_values = NUMERIC_RE.findall(text)
    for value, count in Counter(numeric_values[:80]).most_common(24):
        nid = stable_id("metric", value)
        nodes[nid] = {
            "id": nid,
            "label": value,
            "type": "metric",
            "summary": "Числовой параметр или результат эксперимента",
            "weight": min(5.0, 1.0 + count),
            "source_ids": [source_id],
        }
        eid = edge_id(source_node_id, nid, "reports")
        edges[eid] = {
            "id": eid,
            "source": source_node_id,
            "target": nid,
            "relation": "reports",
            "evidence": value,
            "weight": 1.0 + count,
            "source_ids": [source_id],
        }

    for sentence in SENTENCE_RE.split(text[:60_000]):
        sentence_l = sentence.lower()
        present = []
        for kind, label, aliases in DOMAIN_TERMS:
            if any(alias.lower() in sentence_l for alias in (label, *aliases)):
                present.append((stable_id(kind, label), kind, label))
        for (left_id, left_kind, left_label), (right_id, right_kind, right_label) in itertools.combinations(present[:6], 2):
            relation = _relation_for(left_kind, right_kind)
            eid = edge_id(left_id, right_id, relation)
            if eid not in edges:
                evidence = sentence.strip()
                edges[eid] = {
                    "id": eid,
                    "source": left_id,
                    "target": right_id,
                    "relation": relation,
                    "evidence": evidence[:220],
                    "weight": 1.0,
                    "source_ids": [source_id],
                }
            else:
                edges[eid]["weight"] = float(edges[eid].get("weight", 1)) + 0.3

    return list(nodes.values()), list(edges.values())


def graph_from_hypothesis(hypothesis: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hid = f"hypothesis:{hypothesis['id']}"
    title = hypothesis.get("title") or hypothesis.get("statement", "Гипотеза")
    text = " ".join(
        str(hypothesis.get(key, ""))
        for key in ("title", "statement", "rationale", "mechanism", "uncertainty")
    )
    nodes = [
        {
            "id": hid,
            "label": title[:90],
            "type": "hypothesis",
            "summary": hypothesis.get("statement", ""),
            "weight": 3.0,
            "source_ids": [hypothesis["id"]],
        }
    ]
    edges = []
    lower = text.lower()
    for kind, label, aliases in DOMAIN_TERMS:
        if any(alias.lower() in lower for alias in (label, *aliases)):
            nid = stable_id(kind, label)
            nodes.append(
                {
                    "id": nid,
                    "label": label,
                    "type": kind,
                    "summary": "Связано с гипотезой",
                    "weight": 2.0,
                    "source_ids": [hypothesis["id"]],
                }
            )
            edges.append(
                {
                    "id": edge_id(nid, hid, "supports"),
                    "source": nid,
                    "target": hid,
                    "relation": "supports",
                    "evidence": title[:160],
                    "weight": 2.0,
                    "source_ids": [hypothesis["id"]],
                }
            )
    return nodes, edges


def _relation_for(left_kind: str, right_kind: str) -> str:
    pair = {left_kind, right_kind}
    if "process" in pair and "property" in pair:
        return "influences"
    if "reagent" in pair and "process" in pair:
        return "used_in"
    if "material" in pair and "process" in pair:
        return "processed_by"
    if "risk" in pair:
        return "constrained_by"
    if "equipment" in pair and "process" in pair:
        return "implemented_with"
    return "associated_with"
