const canvas = document.getElementById("pad");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const clearBtn = document.getElementById("clearBtn");
const pressureToggle = document.getElementById("pressureToggle");

let isDrawing = false;
let lastPoint = null;
let activePointerId = null;

const baseLineWidth = 3;
const maxLineWidth = 12;

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  ctx.scale(ratio, ratio);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#1b1b1b";
  redrawGuideLines();
}

function redrawGuideLines() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = canvas.getBoundingClientRect();
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
  ctx.beginPath();
  ctx.moveTo(lastPoint.x, lastPoint.y);
  ctx.lineTo(point.x, point.y);
  ctx.stroke();

  lastPoint = point;
}

function clearCanvas() {
  redrawGuideLines();
}

resizeCanvas();
window.addEventListener("resize", () => {
  resizeCanvas();
});

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);

clearBtn.addEventListener("click", clearCanvas);
