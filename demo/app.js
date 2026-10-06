const state = { config: null, file: null, sample: null, result: null, previewUrl: null };
const $ = (id) => document.getElementById(id);

async function loadConfig() {
  const response = await fetch("/api/config");
  state.config = await response.json();
  const sampleSelect = $("sample-select");
  state.config.samples.forEach((sample) => {
    const option = document.createElement("option");
    option.value = sample.id;
    option.textContent = sample.label;
    sampleSelect.append(option);
  });
  const engineList = $("engine-list");
  Object.entries(state.config.engines).forEach(([id, engine], index) => {
    const label = document.createElement("label");
    label.className = "engine-option";
    label.innerHTML = `<input type="radio" name="engine" value="${id}" ${index === 0 ? "checked" : ""}><strong>${engine.label}</strong><span>${engine.description}</span>`;
    engineList.append(label);
  });
  engineList.addEventListener("change", updateEngineNote);
  updateEngineNote();
}

function updateEngineNote() {
  const selected = document.querySelector('input[name="engine"]:checked');
  if (!selected || !state.config) return;
  $("engine-note").textContent = selected.value === "paddleocr-vl"
    ? "PaddleOCR-VL produced perfect text on the clean Bangla sample, but took about 6 minutes on this CPU."
    : "Tesseract typically finishes a page in under one second on this test machine.";
}

function selectFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return showToast("Please select an image file.");
  if (file.size > 15 * 1024 * 1024) return showToast("The image must be 15 MB or smaller.");
  state.file = file;
  state.sample = null;
  $("sample-select").value = "";
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  showPreview(state.previewUrl, file.name);
}

function selectSample(id) {
  if (!id) return;
  const sample = state.config.samples.find((item) => item.id === id);
  if (!sample) return;
  state.file = null;
  state.sample = sample;
  $("file-input").value = "";
  showPreview(sample.url, sample.label);
}

function showPreview(url, name) {
  clearResults();
  $("preview-image").src = url;
  $("document-name").textContent = name;
  $("preview-empty").hidden = true;
  $("preview-stage").hidden = false;
  $("clear-button").hidden = false;
  $("run-button").disabled = false;
  $("overlay").innerHTML = "";
}

function clearDocument() {
  state.file = null;
  state.sample = null;
  state.result = null;
  $("file-input").value = "";
  $("sample-select").value = "";
  $("document-name").textContent = "No document selected";
  $("preview-empty").hidden = false;
  $("preview-stage").hidden = true;
  $("clear-button").hidden = true;
  $("run-button").disabled = true;
  clearResults();
}

function clearResults() {
  $("results").hidden = true;
  $("overlay").innerHTML = "";
}

function fileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function runOcr() {
  const engine = document.querySelector('input[name="engine"]:checked')?.value;
  if (!engine || (!state.file && !state.sample)) return;
  showStatus(engine === "paddleocr-vl");
  try {
    const payload = { engine };
    if (state.sample) payload.sample_id = state.sample.id;
    else {
      payload.filename = state.file.name;
      payload.content_base64 = await fileAsDataUrl(state.file);
    }
    const response = await fetch("/api/ocr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "OCR failed.");
    state.result = body.result;
    renderResult(body.result);
  } catch (error) {
    showToast(error.message, 7000);
  } finally {
    hideStatus();
  }
}

let timerId = null;
function showStatus(slow) {
  $("status-title").textContent = slow ? "Parsing document structure" : "Reading your document";
  $("status-detail").textContent = slow ? "PaddleOCR-VL is recognizing layout and generating text locally." : "Tesseract is detecting Bangla and English text regions.";
  $("status-overlay").hidden = false;
  const started = Date.now();
  const tick = () => {
    const total = Math.floor((Date.now() - started) / 1000);
    $("elapsed-time").textContent = `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
  };
  tick();
  timerId = setInterval(tick, 1000);
}

function hideStatus() {
  clearInterval(timerId);
  $("status-overlay").hidden = true;
}

function renderResult(result) {
  $("metric-engine").textContent = result.engine;
  $("metric-time").textContent = formatDuration(result.elapsed_seconds);
  $("metric-blocks").textContent = result.blocks.length;
  $("text-output").textContent = result.text || "No text detected.";
  $("character-count").textContent = `${result.text.length.toLocaleString()} characters`;
  const list = $("blocks-list");
  list.innerHTML = "";
  result.blocks.forEach((block, index) => {
    const item = document.createElement("div");
    item.className = "block-item";
    const confidence = block.confidence == null ? "" : ` · ${(block.confidence * 100).toFixed(1)}% confidence`;
    item.innerHTML = `<span class="block-index">${String(index + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(block.block_type)}${confidence}</strong><span>${escapeHtml(shorten(block.text, 150))}</span></div>`;
    list.append(item);
  });
  if (!result.blocks.length) list.innerHTML = '<div class="block-item"><span>—</span><div><span>No regions detected.</span></div></div>';
  drawOverlay(result.blocks);
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawOverlay(blocks) {
  const image = $("preview-image");
  const overlay = $("overlay");
  const stage = $("preview-stage");
  const position = () => {
    const imageRect = image.getBoundingClientRect();
    const stageRect = stage.getBoundingClientRect();
    overlay.style.left = `${imageRect.left - stageRect.left + stage.scrollLeft}px`;
    overlay.style.top = `${imageRect.top - stageRect.top + stage.scrollTop}px`;
    overlay.style.width = `${imageRect.width}px`;
    overlay.style.height = `${imageRect.height}px`;
    overlay.setAttribute("viewBox", `0 0 ${image.naturalWidth} ${image.naturalHeight}`);
  };
  overlay.innerHTML = "";
  blocks.filter((block) => block.bbox).forEach((block, index) => {
    const [x1, y1, x2, y2] = block.bbox;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x1);
    rect.setAttribute("y", y1);
    rect.setAttribute("width", Math.max(0, x2 - x1));
    rect.setAttribute("height", Math.max(0, y2 - y1));
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${index + 1}. ${block.text}`;
    rect.append(title);
    overlay.append(rect);
  });
  if (image.complete) position(); else image.addEventListener("load", position, { once: true });
}

function formatDuration(seconds) {
  if (seconds < 60) return `${seconds.toFixed(2)} sec`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}
function shorten(text, length) { return text.length > length ? `${text.slice(0, length)}…` : text; }
function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = value ?? "";
  return node.innerHTML;
}
function showToast(message, duration = 3500) {
  const toast = $("toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast.timeout);
  showToast.timeout = setTimeout(() => { toast.hidden = true; }, duration);
}

$("file-input").addEventListener("change", (event) => selectFile(event.target.files[0]));
$("sample-select").addEventListener("change", (event) => selectSample(event.target.value));
$("clear-button").addEventListener("click", clearDocument);
$("run-button").addEventListener("click", runOcr);
$("copy-button").addEventListener("click", async () => {
  await navigator.clipboard.writeText(state.result?.text || "");
  showToast("Extracted text copied.");
});
$("download-button").addEventListener("click", () => {
  if (!state.result) return;
  const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `ocr-result-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

const dropzone = $("dropzone");
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
window.addEventListener("resize", () => state.result && drawOverlay(state.result.blocks));
loadConfig().catch(() => showToast("Could not load the local demo configuration."));
