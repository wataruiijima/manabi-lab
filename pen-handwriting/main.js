const canvas = document.getElementById("pad");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const modelStatusEl = document.getElementById("modelStatus");
const resultEl = document.getElementById("result");
const clearBtn = document.getElementById("clearBtn");
const recognizeBtn = document.getElementById("recognizeBtn");
const pressureToggle = document.getElementById("pressureToggle");

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

const baseLineWidth = 3;
const maxLineWidth = 12;
const modelUrl = "model/mnist-onnx/mnist-8.onnx";
const digitSize = 28;

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
  ctx.strokeStyle = "#1b1b1b";

  inkCtx.lineCap = "round";
  inkCtx.lineJoin = "round";
  inkCtx.strokeStyle = "#1b1b1b";

  redrawGuideLines();
}

function redrawGuideLines() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const lineGap = 48;
  ctx.save();
  ctx.strokeStyle = "#efe5d5";
  ctx.lineWidth = 1;
  for (let y = lineGap; y < rect.height; y += lineGap) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(rect.width, y);
    ctx.stroke();
  }
  ctx.restore();
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

function setStatus(text) {
  statusEl.textContent = text;
}

function setModelStatus(text) {
  modelStatusEl.textContent = `Model: ${text}`;
}

function setResult(text) {
  resultEl.textContent = `Result: ${text}`;
}

function startDrawing(event) {
  event.preventDefault();
  const point = getPoint(event);
  if (point.pointerType !== "pen") {
    setStatus(`Ignored (${point.pointerType})`);
    return;
  }
  isDrawing = true;
  lastPoint = point;
  activePointerId = event.pointerId;
  setStatus(`Drawing (${point.pointerType})`);
  canvas.setPointerCapture(event.pointerId);
}

function stopDrawing(event) {
  if (!isDrawing) return;
  if (activePointerId !== event.pointerId) return;
  isDrawing = false;
  lastPoint = null;
  activePointerId = null;
  setStatus("Ready");
  if (event.pointerId != null) {
    canvas.releasePointerCapture(event.pointerId);
  }
}

function draw(event) {
  if (!isDrawing) return;
  if (activePointerId !== event.pointerId) return;
  const point = getPoint(event);
  const pressure = pressureToggle.checked ? point.pressure : 0.5;
  const width = baseLineWidth + pressure * (maxLineWidth - baseLineWidth);

  ctx.lineWidth = width;
  inkCtx.lineWidth = width;

  ctx.beginPath();
  ctx.moveTo(lastPoint.x, lastPoint.y);
  ctx.lineTo(point.x, point.y);
  ctx.stroke();

  inkCtx.beginPath();
  inkCtx.moveTo(lastPoint.x, lastPoint.y);
  inkCtx.lineTo(point.x, point.y);
  inkCtx.stroke();

  lastPoint = point;
}

function clearCanvas() {
  const rect = canvas.getBoundingClientRect();
  inkCtx.clearRect(0, 0, rect.width, rect.height);
  redrawGuideLines();
  setResult("-");
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
  const isInk = (x, y) => data[index(x, y) * 4 + 3] > 20;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = index(x, y);
      if (visited[idx] || !isInk(x, y)) continue;

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

  return boxes.sort((a, b) => a.x - b.x);
}

function makeDigitInput(segment) {
  const padding = Math.round(Math.max(segment.w, segment.h) * 0.2);
  const sx = Math.max(segment.x - padding, 0);
  const sy = Math.max(segment.y - padding, 0);
  const sw = Math.min(segment.w + padding * 2, processingCanvas.width - sx);
  const sh = Math.min(segment.h + padding * 2, processingCanvas.height - sy);

  digitCtx.clearRect(0, 0, digitSize, digitSize);
  digitCtx.fillStyle = "white";
  digitCtx.fillRect(0, 0, digitSize, digitSize);
  digitCtx.drawImage(processingCanvas, sx, sy, sw, sh, 0, 0, digitSize, digitSize);

  const imageData = digitCtx.getImageData(0, 0, digitSize, digitSize).data;
  const data = new Float32Array(digitSize * digitSize);
  for (let i = 0; i < data.length; i += 1) {
    const r = imageData[i * 4];
    const value = 1 - r / 255;
    data[i] = value;
  }

  return new ort.Tensor("float32", data, [1, 1, digitSize, digitSize]);
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
      "ort-wasm.wasm": "vendor/ort-wasm.wasm",
      "ort-wasm-simd.wasm": "vendor/ort-wasm-simd.wasm",
      "ort-wasm-threaded.wasm": "vendor/ort-wasm-threaded.wasm",
      "ort-wasm-simd-threaded.wasm": "vendor/ort-wasm-simd-threaded.wasm",
    };
    session = await ort.InferenceSession.create(modelUrl);
    modelReady = true;
    setModelStatus("loaded");
    return true;
  } catch (error) {
    console.error(error);
    setModelStatus("missing model files");
    return false;
  }
}

async function recognizeDigits() {
  const ready = await ensureModel();
  if (!ready) return;

  const imageData = getInkImageData();
  const segments = findDigitSegments(imageData);

  if (segments.length === 0) {
    setResult("No digits");
    return;
  }

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const predictions = [];

  for (const segment of segments) {
    const input = makeDigitInput(segment);
    const output = await session.run({ [inputName]: input });
    const scores = output[outputName].data;

    let bestIndex = 0;
    let bestScore = scores[0];
    for (let i = 1; i < scores.length; i += 1) {
      if (scores[i] > bestScore) {
        bestScore = scores[i];
        bestIndex = i;
      }
    }
    predictions.push(bestIndex);
  }

  setResult(predictions.join(""));
}

resizeCanvas();
setResult("-");

if (window.ort) {
  setModelStatus("ready to load");
} else {
  setModelStatus("ONNX Runtime not found");
}

window.addEventListener("resize", () => {
  resizeCanvas();
});

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);

clearBtn.addEventListener("click", clearCanvas);
recognizeBtn.addEventListener("click", recognizeDigits);
