const state = {
  projects: [],
  projectId: null,
  state: null,
  weights: { novelty: 0.25, feasibility: 0.25, impact: 0.35, risk: 0.15 },
};

const graphView = {
  key: "",
  nodes: [],
  edges: [],
  nodeById: new Map(),
  edgeById: new Map(),
  transform: { x: 0, y: 0, k: 1 },
  selected: null,
  width: 0,
  height: 0,
  dragging: null,
  panning: null,
  moved: false,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

document.addEventListener("DOMContentLoaded", init);

async function init() {
  $("#actorInput").value = localStorage.getItem("hl.actor") || "researcher";
  bindEvents();
  await refreshProjects();
}

function bindEvents() {
  $("#actorInput").addEventListener("input", (event) => {
    localStorage.setItem("hl.actor", event.target.value.trim() || "researcher");
  });

  $("#projectSelect").addEventListener("change", async (event) => {
    state.projectId = event.target.value;
    localStorage.setItem("hl.projectId", state.projectId);
    await loadState();
  });

  $("#newProjectBtn").addEventListener("click", createProject);
  $("#projectForm").addEventListener("submit", saveProject);
  $("#uploadBtn").addEventListener("click", () => $("#fileInput").click());
  $("#fileInput").addEventListener("change", uploadFiles);
  $("#importBtn").addEventListener("click", importSamples);
  $("#generateBtn").addEventListener("click", generateHypotheses);
  $("#exportJsonBtn").addEventListener("click", () => openExport("json"));
  $("#exportCsvBtn").addEventListener("click", () => openExport("csv"));
  $("#chatForm").addEventListener("submit", sendChat);
  $("#graphZoomOut").addEventListener("click", () => zoomGraph(0.82));
  $("#graphZoomIn").addEventListener("click", () => zoomGraph(1.22));
  $("#graphFit").addEventListener("click", () => fitGraphToViewport(true));

  const graphSvg = $("#graphSvg");
  graphSvg.addEventListener("wheel", onGraphWheel, { passive: false });
  graphSvg.addEventListener("pointerdown", onGraphPointerDown);
  graphSvg.addEventListener("pointermove", onGraphPointerMove);
  graphSvg.addEventListener("pointerup", onGraphPointerUp);
  graphSvg.addEventListener("pointercancel", onGraphPointerUp);

  window.addEventListener("resize", () => {
    clearTimeout(window.__graphResizeTimer);
    window.__graphResizeTimer = setTimeout(drawGraph, 120);
  });

  $$(".range-row input").forEach((input) => {
    input.addEventListener("input", () => {
      state.weights[input.dataset.weight] = Number(input.value);
      renderWeights();
    });
  });

  $("#hypothesesList").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-status]");
    if (!button) return;
    await updateStatus(button.dataset.id, button.dataset.status);
  });

  $("#hypothesesList").addEventListener("submit", async (event) => {
    if (!event.target.matches(".feedback-form")) return;
    event.preventDefault();
    await sendFeedback(event.target);
  });
}

async function refreshProjects() {
  const payload = await api("/api/projects");
  state.projects = payload.projects || [];
  if (!state.projects.length) {
    await createProject({
      preventDefault() {},
      silent: true,
    });
    return;
  }
  const saved = localStorage.getItem("hl.projectId");
  state.projectId = state.projects.some((item) => item.id === saved) ? saved : state.projects[0].id;
  renderProjectSelect();
  await loadState();
}

async function createProject(event) {
  event?.preventDefault?.();
  setBusy(true);
  try {
    const payload = await api("/api/projects", {
      method: "POST",
      json: {
        name: "Хвосты и флотация",
        domain: "Обогащение цветных и благородных металлов",
        goal: "Повысить извлечение ценных компонентов из хвостов при сохранении качества концентрата.",
        constraints: "Использовать доступное лабораторное флотационное оборудование, минимизировать рост расхода реагентов и CAPEX.",
        team: [actor()],
      },
    });
    state.projectId = payload.project.id;
    localStorage.setItem("hl.projectId", state.projectId);
    await refreshProjects();
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function saveProject(event) {
  event.preventDefault();
  if (!state.projectId) return;
  setBusy(true);
  try {
    await api(`/api/projects/${state.projectId}`, {
      method: "PATCH",
      json: {
        name: $("#projectName").value,
        domain: $("#projectDomain").value,
        goal: $("#projectGoal").value,
        constraints: $("#projectConstraints").value,
        settings: state.weights,
      },
    });
    await refreshProjects();
    toast("Проект сохранен");
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function loadState() {
  if (!state.projectId) return;
  setBusy(true);
  try {
    state.state = await api(`/api/projects/${state.projectId}/state`);
    renderAll();
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function uploadFiles(event) {
  const files = Array.from(event.target.files || []);
  if (!files.length || !state.projectId) return;
  setBusy(true);
  try {
    for (const file of files) {
      const form = new FormData();
      form.append("file", file);
      await api(`/api/projects/${state.projectId}/documents`, { method: "POST", body: form });
    }
    await loadState();
    toast(`Загружено файлов: ${files.length}`);
  } catch (error) {
    toast(error.message);
  } finally {
    event.target.value = "";
    setBusy(false);
  }
}

async function importSamples() {
  if (!state.projectId) return;
  setBusy(true);
  try {
    const payload = await api(`/api/projects/${state.projectId}/documents/import-samples`, {
      method: "POST",
      json: { max_files: 12, extensions: [".png", ".jpg", ".jpeg", ".docx", ".xlsx", ".pdf"] },
    });
    await loadState();
    toast(`Импортировано: ${payload.imported.length}`);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function generateHypotheses() {
  if (!state.projectId) return;
  setBusy(true);
  try {
    const exclusions = $("#exclusionsInput").value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const payload = await api(`/api/projects/${state.projectId}/generate`, {
      method: "POST",
      json: {
        count: Number($("#countInput").value || 5),
        weights: state.weights,
        exclusions,
        include_roadmap: true,
      },
    });
    state.state = payload.state;
    renderAll();
    toast("Гипотезы сгенерированы");
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function updateStatus(id, status) {
  setBusy(true);
  try {
    const payload = await api(`/api/projects/${state.projectId}/hypotheses/${id}/status`, {
      method: "PATCH",
      json: { status },
    });
    state.state = payload.state;
    renderAll();
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function sendFeedback(form) {
  const id = form.dataset.id;
  setBusy(true);
  try {
    const payload = await api(`/api/projects/${state.projectId}/hypotheses/${id}/feedback`, {
      method: "POST",
      json: {
        rating: Number($("[name='rating']", form).value || 0) || null,
        outcome: $("[name='outcome']", form).value || null,
        comment: $("[name='comment']", form).value,
      },
    });
    state.state = payload.state;
    renderAll();
    toast("Фидбэк сохранен");
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function sendChat(event) {
  event.preventDefault();
  const input = $("#chatInput");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  setBusy(true);
  try {
    const payload = await api(`/api/projects/${state.projectId}/chat`, {
      method: "POST",
      json: { message },
    });
    state.state = payload.state;
    renderAll();
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

function renderAll() {
  renderProjectSelect();
  renderProjectForm();
  renderWeights();
  renderRuntime();
  renderDocuments();
  renderHypotheses();
  renderEvents();
  renderChat();
  drawGraph();
}

function renderProjectSelect() {
  $("#projectSelect").innerHTML = state.projects
    .map((project) => `<option value="${project.id}" ${project.id === state.projectId ? "selected" : ""}>${escapeHtml(project.name)}</option>`)
    .join("");
}

function renderProjectForm() {
  const project = state.state?.project;
  if (!project) return;
  $("#projectName").value = project.name || "";
  $("#projectDomain").value = project.domain || "";
  $("#projectGoal").value = project.goal || "";
  $("#projectConstraints").value = project.constraints || "";
  state.weights = { ...state.weights, ...(project.settings || {}) };
  $("#projectMeta").textContent = `${state.state.documents.length} источников · ${state.state.hypotheses.length} гипотез`;
}

function renderWeights() {
  const map = {
    novelty: "#weightNovelty",
    feasibility: "#weightFeasibility",
    impact: "#weightImpact",
    risk: "#weightRisk",
  };
  Object.entries(map).forEach(([key, selector]) => {
    const value = Number(state.weights[key] ?? 0).toFixed(2);
    const input = $(`input[data-weight="${key}"]`);
    if (input) input.value = value;
    $(selector).textContent = value;
  });
}

function renderRuntime() {
  const runtime = state.state?.runtime;
  const badge = $("#runtimeBadge");
  if (!runtime) return;
  badge.classList.toggle("ok", Boolean(runtime.openai_enabled));
  badge.textContent = runtime.openai_enabled ? `OpenAI · ${runtime.openai_model}` : "No API key";
}

function renderDocuments() {
  const docs = state.state?.documents || [];
  $("#documentCount").textContent = String(docs.length);
  $("#documentsList").innerHTML = docs.length
    ? docs
        .map(
          (doc) => `<div class="doc-row">
            <b>${escapeHtml(doc.filename)}</b>
            <span>${escapeHtml(doc.content_type)} · ${doc.metadata?.chars || 0} знаков</span>
          </div>`,
        )
        .join("")
    : `<div class="empty">Источников нет</div>`;
}

function renderHypotheses() {
  const hypotheses = state.state?.hypotheses || [];
  const list = $("#hypothesesList");
  if (!hypotheses.length) {
    list.innerHTML = `<div class="empty">Гипотез пока нет</div>`;
    return;
  }
  list.innerHTML = hypotheses.map(renderHypothesis).join("");
}

function renderHypothesis(item) {
  const metrics = [
    ["Новизна", item.novelty],
    ["Реализ.", item.feasibility],
    ["Эффект", item.impact],
    ["Риск", item.risk],
  ];
  const evidence = (item.evidence || [])
    .slice(0, 2)
    .map((ev) => `<span><b>${escapeHtml(ev.source || "source")}</b>: ${escapeHtml(ev.quote || ev.why || "")}</span>`)
    .join("");
  const roadmap = (item.roadmap || [])
    .slice(0, 3)
    .map((step) => `<span>${escapeHtml(step.step || "")}. ${escapeHtml(step.title || "")} · ${escapeHtml(step.output || "")}</span>`)
    .join("");
  const statuses = [
    ["draft", "Черновик"],
    ["review", "Проверка"],
    ["experiment", "Опыт"],
    ["confirmed", "Да"],
    ["rejected", "Нет"],
  ];
  return `<article class="hypothesis-card ${escapeHtml(item.status)}">
    <div class="hypothesis-title">
      <h3>${escapeHtml(item.title)}</h3>
      <div class="score">${Number(item.score || 0).toFixed(1)}</div>
    </div>
    <div class="statement">${escapeHtml(item.statement)}</div>
    <div class="muted">${escapeHtml(item.mechanism || item.rationale || "")}</div>
    <div class="metrics">${metrics
      .map(([label, value]) => `<div class="metric"><span>${label}</span><b>${Number(value || 0).toFixed(0)}</b></div>`)
      .join("")}</div>
    ${evidence ? `<div class="evidence">${evidence}</div>` : ""}
    ${roadmap ? `<div class="roadmap">${roadmap}</div>` : ""}
    <div class="status-row">${statuses
      .map(
        ([status, label]) =>
          `<button type="button" class="${item.status === status ? "active" : ""}" data-id="${item.id}" data-status="${status}">${label}</button>`,
      )
      .join("")}</div>
    <form class="feedback-form" data-id="${item.id}">
      <select name="rating" aria-label="Оценка">
        <option value="">Оценка</option>
        <option value="5">5</option><option value="4">4</option><option value="3">3</option><option value="2">2</option><option value="1">1</option>
      </select>
      <select name="outcome" aria-label="Исход">
        <option value="">Исход</option>
        <option value="confirmed">Подтв.</option>
        <option value="rejected">Опр.</option>
        <option value="experiment">В опыт</option>
      </select>
      <textarea name="comment" rows="1" placeholder="Комментарий эксперта"></textarea>
      <button type="submit">✓</button>
    </form>
  </article>`;
}

function renderEvents() {
  const events = state.state?.events || [];
  $("#eventsList").innerHTML = events.length
    ? events
        .map(
          (event) => `<div class="event-row">
            <b>v${event.version_no} · ${escapeHtml(event.action)}</b>
            <span>${escapeHtml(event.actor)} · ${formatDate(event.created_at)}</span>
          </div>`,
        )
        .join("")
    : `<div class="empty">Событий нет</div>`;
}

function renderChat() {
  const chat = state.state?.chat || [];
  const log = $("#chatLog");
  log.innerHTML = chat.length
    ? chat
        .map(
          (message) => `<div class="message ${escapeHtml(message.role)}">
            <strong>${escapeHtml(message.actor)} · ${formatDate(message.created_at)}</strong>
            ${escapeHtml(message.content)}
          </div>`,
        )
        .join("")
    : `<div class="empty">Чат пуст</div>`;
  log.scrollTop = log.scrollHeight;
}

function drawGraph() {
  const graph = state.state?.graph || { nodes: [], edges: [] };
  const svg = $("#graphSvg");
  svg.innerHTML = "";
  const width = Math.max(360, svg.clientWidth || 720);
  const height = Math.max(260, svg.clientHeight || 360);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  graphView.width = width;
  graphView.height = height;

  const previousNodes = graphView.nodeById;
  const nodes = graph.nodes.slice(0, 90).map((node, index) => {
    const previous = previousNodes.get(node.id);
    return {
      ...node,
      index,
      x: previous?.x,
      y: previous?.y,
      pinned: previous?.pinned || false,
      degree: 0,
    };
  });
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = graph.edges
    .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
    .slice(0, 180)
    .map((edge, index) => ({ ...edge, index }));
  $("#graphStats").textContent = `${graph.nodes.length} узлов · ${graph.edges.length} связей`;

  if (!nodes.length) {
    graphView.key = "";
    graphView.nodes = [];
    graphView.edges = [];
    graphView.nodeById = new Map();
    graphView.edgeById = new Map();
    graphView.selected = null;
    $("#graphDetails").hidden = true;
    svg.appendChild(svgText(width / 2, height / 2, "Граф пуст", "middle", "empty-text"));
    return;
  }

  edges.forEach((edge) => {
    const source = nodes.find((node) => node.id === edge.source);
    const target = nodes.find((node) => node.id === edge.target);
    if (source) source.degree += 1;
    if (target) target.degree += 1;
  });

  const key = graphKey(nodes, edges);
  const needsLayout = key !== graphView.key || nodes.some((node) => !Number.isFinite(node.x) || !Number.isFinite(node.y));
  if (needsLayout) {
    layoutGraph(nodes, edges, width, height);
    graphView.transform = { x: 0, y: 0, k: 1 };
  }

  graphView.key = key;
  graphView.nodes = nodes;
  graphView.edges = edges;
  graphView.nodeById = new Map(nodes.map((node) => [node.id, node]));
  graphView.edgeById = new Map(edges.map((edge) => [edge.id, edge]));
  if (graphView.selected?.type === "node" && !graphView.nodeById.has(graphView.selected.id)) graphView.selected = null;
  if (graphView.selected?.type === "edge" && !graphView.edgeById.has(graphView.selected.id)) graphView.selected = null;

  if (needsLayout) fitGraphToViewport(false);
  renderGraphScene();
  renderGraphDetails();
}

function renderGraphScene() {
  const svg = $("#graphSvg");
  svg.innerHTML = "";
  svg.appendChild(graphDefs());

  const viewport = svgEl("g", {
    transform: `translate(${graphView.transform.x},${graphView.transform.y}) scale(${graphView.transform.k})`,
  });
  const edgeLayer = svgEl("g", { class: "edge-layer" });
  const hitLayer = svgEl("g", { class: "edge-hit-layer" });
  const nodeLayer = svgEl("g", { class: "node-layer" });
  const labelLayer = svgEl("g", { class: "label-layer" });
  viewport.append(edgeLayer, hitLayer, nodeLayer, labelLayer);
  svg.appendChild(viewport);

  graphView.edges.forEach((edge) => {
    const source = graphView.nodeById.get(edge.source);
    const target = graphView.nodeById.get(edge.target);
    if (!source || !target) return;
    const pathValue = edgePath(source, target, edge.index);
    const selected = graphView.selected?.type === "edge" && graphView.selected.id === edge.id;
    const path = svgEl("path", {
      d: pathValue,
      class: `graph-edge ${selected ? "selected" : ""}`,
      fill: "none",
      stroke: selected ? "#08786f" : "#9db0a8",
      "stroke-width": Math.max(1.1, Math.min(4.2, Number(edge.weight || 1))),
      opacity: selected ? 0.95 : 0.52,
      "marker-end": "url(#arrow)",
      "data-edge-id": edge.id,
    });
    const title = svgEl("title");
    title.textContent = `${edge.relation}: ${edge.evidence || ""}`;
    path.appendChild(title);
    edgeLayer.appendChild(path);
    hitLayer.appendChild(svgEl("path", { d: pathValue, class: "edge-hit", "data-edge-id": edge.id }));
  });

  graphView.nodes.forEach((node) => {
    const radius = nodeRadius(node);
    node.r = radius;
    const selected = graphView.selected?.type === "node" && graphView.selected.id === node.id;
    const group = svgEl("g", {
      class: `graph-node ${selected ? "selected" : ""}`,
      transform: `translate(${node.x},${node.y})`,
      "data-node-id": node.id,
    });
    const halo = svgEl("circle", {
      r: radius + 8,
      fill: colorForType(node.type),
      opacity: selected ? 0.18 : 0.08,
    });
    const circle = svgEl("circle", {
      r: radius,
      fill: colorForType(node.type),
      stroke: "#fff",
      "stroke-width": 2,
    });
    const title = svgEl("title");
    title.textContent = `${node.label} · ${node.type}\n${node.summary || ""}`;
    circle.appendChild(title);
    group.append(halo, circle);
    nodeLayer.appendChild(group);
  });

  drawGraphLabels(labelLayer, graphView.nodes, graphView.width);
}

function drawGraphLabels(layer, nodes, width) {
  const placed = [];
  const sorted = [...nodes].sort((a, b) => {
    const priority = (node) => (node.type === "hypothesis" ? 3 : node.type === "source" ? 2 : 1);
    return priority(b) - priority(a) || Number(b.degree || 0) - Number(a.degree || 0) || Number(b.weight || 0) - Number(a.weight || 0);
  });

  sorted.forEach((node) => {
    const text = trim(node.label, node.type === "hypothesis" || node.type === "source" ? 30 : 22);
    const textWidth = Math.min(186, Math.max(42, text.length * 7.1));
    const textHeight = 18;
    let x = node.x + (node.r || 8) + 7;
    if (x + textWidth + 10 > width) {
      x = node.x - (node.r || 8) - textWidth - 12;
    }
    const y = node.y - textHeight / 2;
    const box = { x, y, w: textWidth + 8, h: textHeight };
    if (placed.some((other) => intersects(box, other))) return;
    placed.push(box);

    const selected = graphView.selected?.type === "node" && graphView.selected.id === node.id;
    const group = svgEl("g", {
      class: `graph-label ${selected ? "selected" : ""}`,
      "data-node-id": node.id,
    });
    group.appendChild(
      svgEl("rect", {
        x: box.x,
        y: box.y,
        width: box.w,
        height: box.h,
        rx: 4,
        class: "label-bg",
        "data-node-id": node.id,
      }),
    );
    group.appendChild(svgText(box.x + 4, box.y + 13, text, "start", "node-label"));
    layer.appendChild(group);
  });
}

function layoutGraph(nodes, edges, width, height) {
  const centerX = width / 2;
  const centerY = height / 2;
  const byType = groupBy(nodes, (node) => node.type || "concept");
  Object.values(byType).forEach((items) => {
    items.sort((a, b) => Number(b.degree || 0) - Number(a.degree || 0) || String(a.label).localeCompare(String(b.label)));
  });
  const orderedTypes = ["hypothesis", "source", "process", "material", "reagent", "property", "metric", "equipment", "risk", "observation", "concept"];
  orderedTypes.forEach((type, typeIndex) => {
    const items = byType[type] || [];
    const ring = typeRing(type, width, height);
    const phase = stableNumber(type) * Math.PI * 2 + typeIndex * 0.23;
    items.forEach((node, index) => {
      const angle = phase + (Math.PI * 2 * index) / Math.max(1, items.length);
      const jitter = 1 + (stableNumber(node.id) - 0.5) * 0.12;
      node.x = centerX + Math.cos(angle) * ring * jitter;
      node.y = centerY + Math.sin(angle) * ring * jitter;
      node.vx = 0;
      node.vy = 0;
    });
  });

  const byId = new Map(nodes.map((node) => [node.id, node]));
  const minDistance = 48;
  for (let step = 0; step < 210; step += 1) {
    const cooling = 1 - step / 210;
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const a = nodes[i];
        const b = nodes[j];
        const dx = a.x - b.x || 0.1;
        const dy = a.y - b.y || 0.1;
        const distance = Math.max(1, Math.hypot(dx, dy));
        const desired = minDistance + nodeRadius(a) + nodeRadius(b);
        const repulsion = (desired * desired * 0.018) / distance;
        a.vx += (dx / distance) * repulsion;
        a.vy += (dy / distance) * repulsion;
        b.vx -= (dx / distance) * repulsion;
        b.vy -= (dy / distance) * repulsion;
        if (distance < desired) {
          const push = (desired - distance) * 0.018;
          a.vx += (dx / distance) * push;
          a.vy += (dy / distance) * push;
          b.vx -= (dx / distance) * push;
          b.vy -= (dy / distance) * push;
        }
      }
    }
    edges.forEach((edge) => {
      const a = byId.get(edge.source);
      const b = byId.get(edge.target);
      if (!a || !b) return;
      const dx = b.x - a.x || 0.1;
      const dy = b.y - a.y || 0.1;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const target = edgeTargetDistance(a, b);
      const force = (distance - target) * 0.0055 * cooling;
      a.vx += (dx / distance) * force;
      a.vy += (dy / distance) * force;
      b.vx -= (dx / distance) * force;
      b.vy -= (dy / distance) * force;
    });
    nodes.forEach((node) => {
      const ring = typeRing(node.type, width, height);
      const dx = node.x - centerX || 0.1;
      const dy = node.y - centerY || 0.1;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const radial = (ring - distance) * 0.003;
      node.vx += (dx / distance) * radial;
      node.vy += (dy / distance) * radial;
      node.vx += (centerX - node.x) * 0.0009;
      node.vy += (centerY - node.y) * 0.0009;
      node.x = clamp(node.x + node.vx, 36, width - 36);
      node.y = clamp(node.y + node.vy, 34, height - 34);
      node.vx *= 0.72;
      node.vy *= 0.72;
    });
  }
}

function onGraphWheel(event) {
  event.preventDefault();
  zoomGraph(event.deltaY > 0 ? 0.88 : 1.14, graphPointFromEvent(event));
}

function onGraphPointerDown(event) {
  const nodeElement = event.target.closest?.("[data-node-id]");
  const edgeElement = event.target.closest?.("[data-edge-id]");
  const point = graphPointFromEvent(event);
  graphView.moved = false;
  $("#graphSvg").setPointerCapture?.(event.pointerId);

  if (nodeElement) {
    const node = graphView.nodeById.get(nodeElement.dataset.nodeId);
    if (!node) return;
    const world = screenToWorld(point);
    graphView.dragging = {
      pointerId: event.pointerId,
      nodeId: node.id,
      offsetX: node.x - world.x,
      offsetY: node.y - world.y,
    };
    return;
  }

  if (edgeElement) {
    graphView.pendingSelection = { type: "edge", id: edgeElement.dataset.edgeId };
    return;
  }

  graphView.panning = {
    pointerId: event.pointerId,
    startX: point.x,
    startY: point.y,
    originX: graphView.transform.x,
    originY: graphView.transform.y,
  };
}

function onGraphPointerMove(event) {
  const point = graphPointFromEvent(event);
  if (graphView.dragging?.pointerId === event.pointerId) {
    const node = graphView.nodeById.get(graphView.dragging.nodeId);
    if (!node) return;
    const world = screenToWorld(point);
    node.x = clamp(world.x + graphView.dragging.offsetX, -2000, 2000);
    node.y = clamp(world.y + graphView.dragging.offsetY, -2000, 2000);
    node.pinned = true;
    graphView.moved = true;
    renderGraphScene();
    renderGraphDetails();
    return;
  }

  if (graphView.panning?.pointerId === event.pointerId) {
    graphView.transform.x = graphView.panning.originX + point.x - graphView.panning.startX;
    graphView.transform.y = graphView.panning.originY + point.y - graphView.panning.startY;
    graphView.moved = true;
    renderGraphScene();
  }
}

function onGraphPointerUp(event) {
  $("#graphSvg").releasePointerCapture?.(event.pointerId);
  if (graphView.dragging?.pointerId === event.pointerId) {
    if (!graphView.moved) selectGraphItem("node", graphView.dragging.nodeId);
    graphView.dragging = null;
    return;
  }
  if (graphView.pendingSelection && !graphView.moved) {
    selectGraphItem(graphView.pendingSelection.type, graphView.pendingSelection.id);
  } else if (graphView.panning && !graphView.moved) {
    graphView.selected = null;
    renderGraphScene();
    renderGraphDetails();
  }
  graphView.pendingSelection = null;
  graphView.panning = null;
}

function selectGraphItem(type, id) {
  graphView.selected = { type, id };
  renderGraphScene();
  renderGraphDetails();
}

function renderGraphDetails() {
  const details = $("#graphDetails");
  const selected = graphView.selected;
  if (!selected) {
    details.hidden = true;
    details.innerHTML = "";
    return;
  }

  if (selected.type === "node") {
    const node = graphView.nodeById.get(selected.id);
    if (!node) {
      details.hidden = true;
      return;
    }
    const related = graphView.edges.filter((edge) => edge.source === node.id || edge.target === node.id);
    details.innerHTML = `<button class="tool-button graph-close" type="button" title="Закрыть">×</button>
      <h3>${escapeHtml(node.label)}</h3>
      <div class="graph-detail-meta">
        <span class="graph-chip">${escapeHtml(typeLabel(node.type))}</span>
        <span class="graph-chip">${related.length} связей</span>
        <span class="graph-chip">w ${Number(node.weight || 1).toFixed(1)}</span>
      </div>
      <p>${escapeHtml(node.summary || "Нет краткого описания.")}</p>
      ${related.length ? `<p>${escapeHtml(related.slice(0, 5).map((edge) => relationLine(edge, node.id)).join("; "))}</p>` : ""}`;
  } else {
    const edge = graphView.edgeById.get(selected.id);
    if (!edge) {
      details.hidden = true;
      return;
    }
    const source = graphView.nodeById.get(edge.source);
    const target = graphView.nodeById.get(edge.target);
    details.innerHTML = `<button class="tool-button graph-close" type="button" title="Закрыть">×</button>
      <h3>${escapeHtml(source?.label || edge.source)} → ${escapeHtml(target?.label || edge.target)}</h3>
      <div class="graph-detail-meta">
        <span class="graph-chip">${escapeHtml(edge.relation || "relation")}</span>
        <span class="graph-chip">w ${Number(edge.weight || 1).toFixed(1)}</span>
      </div>
      <p>${escapeHtml(edge.evidence || "Нет фрагмента-доказательства.")}</p>`;
  }
  details.hidden = false;
  $(".graph-close", details)?.addEventListener("click", () => {
    graphView.selected = null;
    renderGraphScene();
    renderGraphDetails();
  });
}

function fitGraphToViewport(animate) {
  if (!graphView.nodes.length) return;
  const padding = 34;
  const bounds = graphBounds(graphView.nodes);
  const width = Math.max(1, bounds.maxX - bounds.minX);
  const height = Math.max(1, bounds.maxY - bounds.minY);
  const k = clamp(Math.min((graphView.width - padding * 2) / width, (graphView.height - padding * 2) / height), 0.36, 1.7);
  graphView.transform = {
    k,
    x: graphView.width / 2 - ((bounds.minX + bounds.maxX) / 2) * k,
    y: graphView.height / 2 - ((bounds.minY + bounds.maxY) / 2) * k,
  };
  if (animate) toast("Граф выровнен");
  renderGraphScene();
}

function zoomGraph(factor, center = { x: graphView.width / 2, y: graphView.height / 2 }) {
  const previous = graphView.transform;
  const nextK = clamp(previous.k * factor, 0.25, 3.2);
  const world = screenToWorld(center);
  graphView.transform = {
    k: nextK,
    x: center.x - world.x * nextK,
    y: center.y - world.y * nextK,
  };
  renderGraphScene();
}

function screenToWorld(point) {
  return {
    x: (point.x - graphView.transform.x) / graphView.transform.k,
    y: (point.y - graphView.transform.y) / graphView.transform.k,
  };
}

function graphPointFromEvent(event) {
  const rect = $("#graphSvg").getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function graphDefs() {
  const defs = svgEl("defs");
  const marker = svgEl("marker", {
    id: "arrow",
    viewBox: "0 0 10 10",
    refX: 9,
    refY: 5,
    markerWidth: 5,
    markerHeight: 5,
    orient: "auto-start-reverse",
  });
  marker.appendChild(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#9db0a8", opacity: 0.85 }));
  defs.appendChild(marker);
  return defs;
}

function edgePath(source, target, index) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.max(1, Math.hypot(dx, dy));
  const offset = ((index % 5) - 2) * Math.min(18, distance * 0.04);
  const nx = -dy / distance;
  const ny = dx / distance;
  const sourceR = source.r || nodeRadius(source);
  const targetR = target.r || nodeRadius(target);
  const x1 = source.x + (dx / distance) * sourceR;
  const y1 = source.y + (dy / distance) * sourceR;
  const x2 = target.x - (dx / distance) * (targetR + 5);
  const y2 = target.y - (dy / distance) * (targetR + 5);
  const cx = (x1 + x2) / 2 + nx * offset;
  const cy = (y1 + y2) / 2 + ny * offset;
  return `M ${x1.toFixed(1)} ${y1.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
}

function graphBounds(nodes) {
  return nodes.reduce(
    (bounds, node) => ({
      minX: Math.min(bounds.minX, node.x - nodeRadius(node) - 44),
      maxX: Math.max(bounds.maxX, node.x + nodeRadius(node) + 80),
      minY: Math.min(bounds.minY, node.y - nodeRadius(node) - 28),
      maxY: Math.max(bounds.maxY, node.y + nodeRadius(node) + 28),
    }),
    { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity },
  );
}

function graphKey(nodes, edges) {
  return `${nodes.map((node) => node.id).sort().join("|")}::${edges.map((edge) => edge.id).sort().join("|")}`;
}

function nodeRadius(node) {
  const base = node.type === "hypothesis" ? 12 : node.type === "source" ? 10 : 8;
  return Math.max(base, Math.min(22, base + Number(node.weight || 1) * 1.25 + Math.sqrt(Number(node.degree || 0)) * 1.5));
}

function edgeTargetDistance(source, target) {
  if (source.type === "hypothesis" || target.type === "hypothesis") return 118;
  if (source.type === "source" || target.type === "source") return 138;
  return 104;
}

function typeRing(type, width, height) {
  const radius = Math.min(width, height) * 0.42;
  const factors = {
    hypothesis: 0.28,
    source: 0.34,
    process: 0.56,
    material: 0.66,
    reagent: 0.76,
    property: 0.86,
    metric: 0.94,
    equipment: 0.72,
    risk: 0.9,
    observation: 0.82,
  };
  return radius * (factors[type] || 0.84);
}

function relationLine(edge, nodeId) {
  const otherId = edge.source === nodeId ? edge.target : edge.source;
  const other = graphView.nodeById.get(otherId);
  return `${edge.relation || "related"}: ${other?.label || otherId}`;
}

function typeLabel(type) {
  return (
    {
      source: "источник",
      material: "материал",
      process: "процесс",
      reagent: "реагент",
      property: "свойство",
      metric: "метрика",
      equipment: "оборудование",
      risk: "риск",
      hypothesis: "гипотеза",
      observation: "наблюдение",
    }[type] || "концепт"
  );
}

function groupBy(items, getter) {
  return items.reduce((groups, item) => {
    const key = getter(item);
    groups[key] = groups[key] || [];
    groups[key].push(item);
    return groups;
  }, {});
}

function stableNumber(value) {
  let hash = 0;
  const text = String(value || "");
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  }
  return (hash % 10000) / 10000;
}

async function api(path, options = {}) {
  const headers = { "X-User": actor(), ...(options.headers || {}) };
  const fetchOptions = { ...options, headers };
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    fetchOptions.body = JSON.stringify(options.json);
    delete fetchOptions.json;
  }
  const response = await fetch(path, fetchOptions);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

function openExport(type) {
  if (!state.projectId) return;
  window.open(`/api/projects/${state.projectId}/export.${type}`, "_blank", "noopener");
}

function setBusy(value) {
  $("#busyIndicator").hidden = !value;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => {
    node.hidden = true;
  }, 3600);
}

function actor() {
  return $("#actorInput").value.trim() || "researcher";
}

function svgEl(name, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function svgText(x, y, text, anchor, className) {
  const element = svgEl("text", { x, y, "text-anchor": anchor, class: className });
  element.textContent = text;
  return element;
}

function colorForType(type) {
  return {
    source: "#1f2523",
    material: "#08786f",
    process: "#6653a8",
    reagent: "#c95243",
    property: "#2f8d58",
    metric: "#a77a11",
    equipment: "#4c6f88",
    risk: "#9d3c35",
    hypothesis: "#0b9a8c",
    observation: "#607064",
  }[type] || "#607064";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDate(value) {
  if (!value) return "";
  return new Date(value).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function trim(value, length) {
  const text = String(value || "");
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function intersects(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}
