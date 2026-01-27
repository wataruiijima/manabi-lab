const canvas = document.getElementById("pad");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("roundStatus");
const modelStatusEl = document.getElementById("modelStatus");
const timerEl = document.getElementById("timer");
const meterFill = document.getElementById("meterFill");
const equationEl = document.getElementById("equation");
const equationShell = document.querySelector(".equation-shell");
const equationHint = document.getElementById("equationHint");
const candidatesEl = document.getElementById("candidates");
const clearBtn = document.getElementById("clearBtn");
const startBtn = document.getElementById("startBtn");
const loadingOverlay = document.getElementById("loadingOverlay");

const inkCanvas = document.createElement("canvas");
const inkCtx = inkCanvas.getContext("2d");
const processingCanvas = document.createElement("canvas");
const processingCtx = processingCanvas.getContext("2d");
const digitCanvas = document.createElement("canvas");
const digitCtx = digitCanvas.getContext("2d");

let isDrawing = false;
let lastPoint = null;
let activePointerId = null;
let session = null;
let modelReady = false;
let recognizeTimer = null;
let recognizeInFlight = false;
let recognizePending = false;

let roundActive = false;
let roundStart = 0;
let roundRaf = null;
let currentAnswer = null;

const baseLineWidth = 4;
const maxLineWidth = 12;
const modelUrl = "../model/mnist-onnx/mnist-12.onnx";
const digitSize = 28;
const roundDurationMs = 10000;

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  inkCanvas.width = canvas.width;
  inkCanvas.height = canvas.height;
  processingCanvas.width = Math.floor(rect.width);
  processingCanvas.height = Math.floor(rect.height);
  digitCanvas.width = digitSize;
  digitCanvas.height = digitSize;

  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  inkCtx.setTransform(ratio, 0, 0, ratio, 0, 0);

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#1c1b19";

  inkCtx.lineCap = "round";
  inkCtx.lineJoin = "round";
  inkCtx.strokeStyle = "#1c1b19";
}

function getPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
    pressure: event.pressure || 0.5,
    pointerType: event.pointerType || "mouse",
  };
}

function getLineWidth(point) {
  const pressure = point.pointerType === "pen" ? point.pressure : 0.5;
  return baseLineWidth + pressure * (maxLineWidth - baseLineWidth);
}

function setCanvasEnabled(enabled) {
  document.body.classList.toggle("canvas-disabled", !enabled);
  if (!enabled) {
    loadingOverlay.setAttribute("aria-hidden", "false");
  } else {
    loadingOverlay.setAttribute("aria-hidden", "true");
  }
}

function setStatus(text) {
  statusEl.textContent = text;
}

function setModelStatus(text) {
  modelStatusEl.textContent = `Model: ${text}`;
}

function formatModelError(error) {
  if (!error) return "unknown error";
  if (typeof error === "string") return error;
  if (error.message) return error.message;
  return "model load failed";
}

function startDrawing(event) {
  event.preventDefault();
  const point = getPoint(event);
  if (point.pointerType === "touch") return;
  isDrawing = true;
  lastPoint = { x: point.x, y: point.y, width: getLineWidth(point) };
  activePointerId = event.pointerId;
  canvas.setPointerCapture(event.pointerId);
}

function stopDrawing(event) {
  event.preventDefault();
  if (!isDrawing) return;
  if (activePointerId !== event.pointerId) return;
  isDrawing = false;
  lastPoint = null;
  activePointerId = null;
  if (event.pointerId != null) {
    canvas.releasePointerCapture(event.pointerId);
  }
  scheduleRecognize();
}

function draw(event) {
  event.preventDefault();
  if (!isDrawing) return;
  if (activePointerId !== event.pointerId) return;
  const point = getPoint(event);
  const nextPoint = { x: point.x, y: point.y, width: getLineWidth(point) };

  ctx.lineWidth = nextPoint.width;
  ctx.beginPath();
  ctx.moveTo(lastPoint.x, lastPoint.y);
  ctx.lineTo(nextPoint.x, nextPoint.y);
  ctx.stroke();

  inkCtx.lineWidth = nextPoint.width;
  inkCtx.beginPath();
  inkCtx.moveTo(lastPoint.x, lastPoint.y);
  inkCtx.lineTo(nextPoint.x, nextPoint.y);
  inkCtx.stroke();

  lastPoint = nextPoint;
  scheduleRecognize();
}

function clearPad() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  inkCtx.clearRect(0, 0, rect.width, rect.height);
  updateCandidates([]);
}

function getInkImageData() {
  const rect = canvas.getBoundingClientRect();
  processingCtx.clearRect(0, 0, rect.width, rect.height);
  processingCtx.drawImage(inkCanvas, 0, 0, rect.width, rect.height);
  return processingCtx.getImageData(0, 0, rect.width, rect.height);
}

function findDigitSegments(imageData) {
  const { data, width, height } = imageData;
  const visited = new Uint8Array(width * height);
  const boxes = [];
  const minPixels = 40;

  const index = (x, y) => y * width + x;
  // alpha がしきい値を超えるピクセルをインクとして扱う。
  const isInk = (x, y) => data[index(x, y) * 4 + 3] > 20;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = index(x, y);
      if (visited[idx] || !isInk(x, y)) continue;

      // 連結成分のフラッドフィルでインク領域と境界を収集する。
      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      let count = 0;
      const stack = [{ x, y }];
      visited[idx] = 1;

      while (stack.length) {
        const { x: cx, y: cy } = stack.pop();
        count += 1;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dy === 0) continue;
            const nx = cx + dx;
            const ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            const nidx = index(nx, ny);
            if (visited[nidx] || !isInk(nx, ny)) continue;
            visited[nidx] = 1;
            stack.push({ x: nx, y: ny });
          }
        }
      }

      // 小さすぎる塊はノイズとして無視する。
      if (count >= minPixels) {
        boxes.push({
          x: minX,
          y: minY,
          w: maxX - minX + 1,
          h: maxY - minY + 1,
          count,
        });
      }
    }
  }

  // 左から右の順に並べて桁認識に使う。
  return boxes.sort((a, b) => a.x - b.x);
}

function makeDigitInput(segment) {
  if (!segment) return null;
  const padding = Math.round(Math.max(segment.w, segment.h) * 0.2);
  const sx = Math.max(segment.x - padding, 0);
  const sy = Math.max(segment.y - padding, 0);
  const sw = Math.min(segment.w + padding * 2, processingCanvas.width - sx);
  const sh = Math.min(segment.h + padding * 2, processingCanvas.height - sy);

  digitCtx.clearRect(0, 0, digitSize, digitSize);
  digitCtx.fillStyle = "white";
  digitCtx.fillRect(0, 0, digitSize, digitSize);
  digitCtx.imageSmoothingEnabled = true;

  const targetSize = 26;
  const scale = targetSize / Math.max(sw, sh);
  const dw = sw * scale;
  const dh = sh * scale;
  const dx = (digitSize - dw) / 2;
  const dy = (digitSize - dh) / 2;

  digitCtx.drawImage(processingCanvas, sx, sy, sw, sh, dx, dy, dw, dh);

  const imageData = digitCtx.getImageData(0, 0, digitSize, digitSize).data;
  const data = new Float32Array(digitSize * digitSize);
  for (let i = 0; i < data.length; i += 1) {
    const r = imageData[i * 4];
    const normalized = 1 - r / 255;
    data[i] = Math.min(1, Math.max(0, (normalized - 0.1) / 0.9));
  }

  return { tensor: new ort.Tensor("float32", data, [1, 1, digitSize, digitSize]) };
}

function getTopDigits(scores, count) {
  const entries = scores.map((score, digit) => ({ digit, score }));
  entries.sort((a, b) => b.score - a.score);
  return entries.slice(0, count);
}

function buildCandidates(rankLists, limit) {
  let beams = [{ text: "", score: 0 }];
  for (const ranks of rankLists) {
    const next = [];
    for (const beam of beams) {
      for (const rank of ranks) {
        next.push({
          text: beam.text + rank.digit,
          score: beam.score + rank.score,
        });
      }
    }
    next.sort((a, b) => b.score - a.score);
    beams = next.slice(0, limit);
  }
  const seen = new Set();
  return beams.filter((beam) => {
    if (seen.has(beam.text)) return false;
    seen.add(beam.text);
    return true;
  });
}

function updateCandidates(list) {
  candidatesEl.textContent = "";
  if (list.length === 0) {
    const empty = document.createElement("div");
    empty.textContent = "--";
    empty.className = "hint";
    candidatesEl.appendChild(empty);
    return;
  }
  for (const item of list) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "candidate";
    btn.textContent = item.text;
    btn.dataset.value = item.text;
    candidatesEl.appendChild(btn);
  }
}

async function ensureModel() {
  if (modelReady) return true;
  if (!window.ort) {
    setModelStatus("ONNX Runtime not found");
    return false;
  }
  setModelStatus("loading...");
  try {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = false;
    ort.env.wasm.wasmPaths = {
      "ort-wasm.wasm": "../vendor/ort-wasm.wasm",
      "ort-wasm-simd.wasm": "../vendor/ort-wasm-simd.wasm",
      "ort-wasm-threaded.wasm": "../vendor/ort-wasm-threaded.wasm",
      "ort-wasm-simd-threaded.wasm": "../vendor/ort-wasm-simd-threaded.wasm",
    };
    session = await ort.InferenceSession.create(modelUrl);
    modelReady = true;
    setModelStatus("loaded");
    setCanvasEnabled(true);
    return true;
  } catch (error) {
    console.error(error);
    setModelStatus(`load failed: ${formatModelError(error)}`);
    setCanvasEnabled(false);
    return false;
  }
}

function scheduleRecognize() {
  if (!modelReady) return;
  if (recognizeTimer) clearTimeout(recognizeTimer);
  recognizeTimer = setTimeout(() => {
    void runRecognize();
  }, 200);
}

async function runRecognize() {
  if (recognizeInFlight) {
    recognizePending = true;
    return;
  }
  recognizeInFlight = true;
  await recognizeDigits();
  recognizeInFlight = false;
  if (recognizePending) {
    recognizePending = false;
    scheduleRecognize();
  }
}

async function recognizeDigits() {
  const ready = await ensureModel();
  if (!ready) return;

  const imageData = getInkImageData();
  const segments = findDigitSegments(imageData);

  if (segments.length === 0) {
    updateCandidates([]);
    return;
  }

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const rankLists = [];

  for (const segment of segments) {
    const result = makeDigitInput(segment);
    if (!result) continue;
    const output = await session.run({ [inputName]: result.tensor });
    const scores = Array.from(output[outputName].data);
    rankLists.push(getTopDigits(scores, 3));
  }

  const candidates = buildCandidates(rankLists, 5);
  updateCandidates(candidates);
}

function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function pickQuestion() {
  const useMultiply = Math.random() < 0.5;
  if (useMultiply) {
    const a = randInt(10, 99);
    const b = randInt(10, 99);
    return { text: `${a} × ${b}`, answer: a * b };
  }
  const divisor = randInt(10, 99);
  const quotient = randInt(10, 99);
  const dividend = divisor * quotient;
  return { text: `${dividend} ÷ ${divisor}`, answer: quotient };
}

function setEquation(text) {
  equationEl.textContent = text;
  equationEl.classList.remove("success", "fail", "growing");
  equationShell.classList.remove("hit");
  void equationEl.offsetWidth;
  equationEl.classList.add("growing");
}

function updateTimerDisplay(remainingMs) {
  const seconds = Math.max(0, remainingMs / 1000);
  timerEl.textContent = `${seconds.toFixed(1)}s`;
  const ratio = Math.max(0, Math.min(1, remainingMs / roundDurationMs));
  meterFill.style.width = `${ratio * 100}%`;
}

function tickRound() {
  const now = performance.now();
  const elapsed = now - roundStart;
  const remaining = roundDurationMs - elapsed;
  updateTimerDisplay(remaining);
  if (remaining <= 0) {
    failRound();
    return;
  }
  roundRaf = requestAnimationFrame(tickRound);
}

function startRound() {
  const question = pickQuestion();
  currentAnswer = question.answer;
  setEquation(question.text);
  equationHint.textContent = "10秒以内に答えを撃て！";
  setStatus("GO");
  roundActive = true;
  roundStart = performance.now();
  if (roundRaf) cancelAnimationFrame(roundRaf);
  updateTimerDisplay(roundDurationMs);
  tickRound();
  clearPad();
}

function endRound() {
  roundActive = false;
  if (roundRaf) cancelAnimationFrame(roundRaf);
  roundRaf = null;
  equationEl.classList.remove("growing");
}

function winRound() {
  endRound();
  setStatus("CLEAR!");
  equationShell.classList.add("hit");
  equationEl.classList.add("success");
  equationHint.textContent = "次の問題へ…";
  setTimeout(() => {
    startRound();
  }, 800);
}

function failRound() {
  endRound();
  setStatus("TIME UP");
  equationEl.classList.add("fail");
  equationHint.textContent = "もう一度スタート";
}

function launchProjectile(button, value, isCorrect) {
  const startRect = button.getBoundingClientRect();
  const endRect = equationEl.getBoundingClientRect();
  const startX = startRect.left + startRect.width / 2;
  const startY = startRect.top + startRect.height / 2;
  const endX = endRect.left + endRect.width / 2;
  const endY = endRect.top + endRect.height / 2;
  const dx = endX - startX;
  const dy = endY - startY;

  const projectile = document.createElement("div");
  projectile.className = `projectile${isCorrect ? "" : " miss"}`;
  projectile.textContent = value;
  projectile.style.left = `${startX}px`;
  projectile.style.top = `${startY}px`;
  document.body.appendChild(projectile);

  const fly = projectile.animate(
    [
      { transform: "translate(-50%, -50%) scale(1)" },
      { transform: `translate(${dx}px, ${dy}px) scale(1)` },
    ],
    { duration: 380, easing: "cubic-bezier(0.3, 0.7, 0.2, 1)" }
  );

  fly.onfinish = () => {
    if (isCorrect) {
      projectile.remove();
      winRound();
      return;
    }
    const fall = projectile.animate(
      [
        { transform: `translate(${dx}px, ${dy}px) scale(1)`, opacity: 1 },
        { transform: `translate(${dx}px, ${dy + 120}px) scale(0.6)`, opacity: 0 },
      ],
      { duration: 260, easing: "ease-in" }
    );
    fall.onfinish = () => projectile.remove();
    equationEl.classList.add("fail");
    setTimeout(() => equationEl.classList.remove("fail"), 500);
  };
}

function handleCandidateClick(event) {
  const btn = event.target.closest("button");
  if (!btn) return;
  if (!roundActive) return;
  const value = btn.dataset.value;
  if (!value) return;
  const numeric = Number.parseInt(value, 10);
  if (Number.isNaN(numeric)) return;
  const isCorrect = numeric === currentAnswer;
  launchProjectile(btn, value, isCorrect);
}

resizeCanvas();
setCanvasEnabled(false);
setStatus("READY");
updateTimerDisplay(roundDurationMs);

if (window.ort) {
  setModelStatus("loading...");
  void ensureModel();
} else {
  setModelStatus("ONNX Runtime not found");
}

window.addEventListener("resize", resizeCanvas);

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);
canvas.addEventListener("contextmenu", (event) => event.preventDefault());

clearBtn.addEventListener("click", clearPad);
startBtn.addEventListener("click", startRound);

candidatesEl.addEventListener("click", handleCandidateClick);
