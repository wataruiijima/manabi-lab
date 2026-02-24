const rows = [
  ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
  ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
  ["Z", "X", "C", "V", "B", "N", "M"],
];

const targetKeyEl = document.getElementById("targetKey");
const feedbackEl = document.getElementById("feedback");
const typedLogEl = document.getElementById("typedLog");
const scoreEl = document.getElementById("score");
const comboEl = document.getElementById("combo");
const keyboardEl = document.getElementById("keyboard");
const projectilesEl = document.getElementById("projectiles");
const enemyEl = document.getElementById("enemy");

let target = "A";
let score = 0;
let combo = 0;
let typedLog = [];

const keyButtons = new Map();
const positionMap = new Map();
rows.forEach((row, r) => {
  row.forEach((key, c) => {
    positionMap.set(key, { r, c });
  });
});

function buildKeyboard() {
  rows.forEach((row) => {
    const rowEl = document.createElement("div");
    rowEl.className = "keyboard-row";
    row.forEach((key) => {
      const keyEl = document.createElement("div");
      keyEl.className = "key";
      keyEl.textContent = key;
      keyEl.dataset.key = key;
      rowEl.appendChild(keyEl);
      keyButtons.set(key, keyEl);
    });
    keyboardEl.appendChild(rowEl);
  });
}

function neighborsOf(key) {
  const pos = positionMap.get(key);
  if (!pos) return [];
  const near = [];
  for (let dr = -1; dr <= 1; dr += 1) {
    for (let dc = -1; dc <= 1; dc += 1) {
      if (dr === 0 && dc === 0) continue;
      const rk = rows[pos.r + dr];
      if (!rk) continue;
      const neighbor = rk[pos.c + dc];
      if (neighbor) near.push(neighbor);
    }
  }
  return near;
}

function setNextTarget() {
  const flat = rows.flat();
  target = flat[Math.floor(Math.random() * flat.length)];
  targetKeyEl.textContent = target;
  renderKeyboard();
}

function renderKeyboard(pressed = "") {
  const near = neighborsOf(target);
  keyButtons.forEach((el, key) => {
    el.classList.toggle("target", key === target);
    el.classList.toggle("near", near.includes(key));
    el.classList.toggle("pressed", key === pressed);
  });
}

function spawnProjectile(letter, rating) {
  const proj = document.createElement("div");
  proj.className = "projectile";
  proj.textContent = letter;
  if (rating === "perfect") proj.style.color = "#ff4d6d";
  if (rating === "good") proj.style.color = "#4d96ff";
  if (rating === "miss") proj.style.color = "#808080";

  projectilesEl.appendChild(proj);
  setTimeout(() => proj.remove(), 700);

  enemyEl.classList.add("hit");
  setTimeout(() => enemyEl.classList.remove("hit"), 180);
}

function updateLog(letter) {
  typedLog.unshift(letter);
  typedLog = typedLog.slice(0, 14);
  typedLogEl.textContent = typedLog.join(" ");
}

function judge(input) {
  const near = neighborsOf(target);
  if (input === target) return "perfect";
  if (near.includes(input)) return "good";
  return "miss";
}

function applyScore(rating) {
  if (rating === "perfect") {
    score += 120;
    combo += 1;
    feedbackEl.textContent = "Perfect! すごい！";
    feedbackEl.style.color = "#ff4d6d";
  } else if (rating === "good") {
    score += 60;
    combo += 1;
    feedbackEl.textContent = "Good! おしい！";
    feedbackEl.style.color = "#2f74d0";
  } else {
    score += 10;
    combo = 0;
    feedbackEl.textContent = "Miss! つぎいこう！";
    feedbackEl.style.color = "#666";
  }

  scoreEl.textContent = String(score);
  comboEl.textContent = String(combo);
}

function normalizeInputKey(event) {
  if (event.isComposing) return "";

  if (/^Key[A-Z]$/.test(event.code)) {
    return event.code.replace("Key", "");
  }

  const key = (event.key || "").toUpperCase();
  if (/^[A-Z]$/.test(key)) return key;
  return "";
}

document.addEventListener("keydown", (event) => {
  const input = normalizeInputKey(event);
  if (!input) return;

  const rating = judge(input);
  updateLog(input);
  applyScore(rating);
  spawnProjectile(input, rating);
  renderKeyboard(input);

  if (rating !== "miss") {
    setTimeout(setNextTarget, 240);
  }

  setTimeout(() => renderKeyboard(), 120);
});

buildKeyboard();
setNextTarget();
