const letterRows = [
  ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
  ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
  ["Z", "X", "C", "V", "B", "N", "M"],
];

const keyboardLayouts = {
  JIS: [
    [
      { id: "ESC", label: "Esc", size: "wide", group: "function" },
      ...Array.from({ length: 12 }, (_, i) => ({ id: `F${i + 1}`, label: `F${i + 1}`, group: "function" })),
    ],
    [
      { id: "HANKAKU", label: "半/全", size: "wide" },
      ..."1234567890".split("").map((v) => ({ id: v, label: v, group: "number" })),
      { id: "-", label: "-", group: "number" },
      { id: "^", label: "^", group: "number" },
      { id: "BACKSPACE", label: "Back", size: "xwide" },
    ],
    [{ id: "TAB", label: "Tab", size: "wide" }, ...letterRows[0].map((k) => ({ id: k, label: k })), { id: "@", label: "@" }, { id: "[", label: "[" }],
    [{ id: "CAPS", label: "Caps", size: "wide" }, ...letterRows[1].map((k) => ({ id: k, label: k })), { id: ";", label: ";" }, { id: ":", label: ":" }, { id: "]", label: "]" }, { id: "ENTER", label: "Enter", size: "xwide" }],
    [{ id: "SHIFT", label: "Shift", size: "xwide" }, ...letterRows[2].map((k) => ({ id: k, label: k })), { id: ",", label: "," }, { id: ".", label: "." }, { id: "/", label: "/" }],
  ],
  US: [
    [
      { id: "ESC", label: "Esc", size: "wide", group: "function" },
      ...Array.from({ length: 12 }, (_, i) => ({ id: `F${i + 1}`, label: `F${i + 1}`, group: "function" })),
    ],
    [
      { id: "`", label: "`", group: "number" },
      ..."1234567890".split("").map((v) => ({ id: v, label: v, group: "number" })),
      { id: "-", label: "-", group: "number" },
      { id: "=", label: "=", group: "number" },
      { id: "BACKSPACE", label: "Back", size: "xwide" },
    ],
    [{ id: "TAB", label: "Tab", size: "wide" }, ...letterRows[0].map((k) => ({ id: k, label: k })), { id: "[", label: "[" }, { id: "]", label: "]" }, { id: "\\", label: "\\" }],
    [{ id: "CAPS", label: "Caps", size: "wide" }, ...letterRows[1].map((k) => ({ id: k, label: k })), { id: ";", label: ";" }, { id: "'", label: "'" }, { id: "ENTER", label: "Enter", size: "xwide" }],
    [{ id: "SHIFT", label: "Shift", size: "xwide" }, ...letterRows[2].map((k) => ({ id: k, label: k })), { id: ",", label: "," }, { id: ".", label: "." }, { id: "/", label: "/" }],
  ],
};

const settings = { layout: "JIS", showFunctionKeys: true, showNumberRow: true };

const similarKeyGroups = [["O", "D", "Q"], ["P", "R", "B"], ["I", "L", "J"], ["U", "V", "Y"], ["C", "G"], ["M", "N"], ["S", "Z"], ["K", "X"]];
const similarKeyMap = new Map();
similarKeyGroups.forEach((group) => group.forEach((key) => similarKeyMap.set(key, group.filter((candidate) => candidate !== key))));

const getById = (id) => document.getElementById(id);
const targetKeyEl = getById("targetKey");
const targetModeEl = getById("targetMode");
const targetGuideEl = getById("targetGuide");
const feedbackEl = getById("feedback");
const typedLogEl = getById("typedLog");
const scoreEl = getById("score");
const comboEl = getById("combo");
const waveEl = getById("wave");
const perfectStreakEl = getById("perfectStreak");
const keyboardEl = getById("keyboard");
const keyboardGuideOverlayEl = getById("keyboardGuideOverlay");
const inputGaugeFillEl = getById("inputGaugeFill");
const playerHpFillEl = getById("playerHpFill");
const retryOverlayEl = getById("retryOverlay");
const retryButtonEl = getById("retryButton");
const retryScoreTextEl = getById("retryScoreText");
const clearOverlayEl = getById("clearOverlay");
const clearButtonEl = getById("clearButton");
const clearScoreTextEl = getById("clearScoreText");
const targetPanelEl = document.querySelector(".target-panel");
const startOverlayEl = getById("startOverlay");
const startButtonEl = getById("startButton");
const openSettingsButtonEl = getById("openSettingsButton");
const settingsOverlayEl = getById("settingsOverlay");
const layoutSelectEl = getById("layoutSelect");
const toggleFunctionRowEl = getById("toggleFunctionRow");
const toggleNumberRowEl = getById("toggleNumberRow");
const applySettingsButtonEl = getById("applySettingsButton");
const closeSettingsButtonEl = getById("closeSettingsButton");
const gateOverlayEl = getById("gateOverlay");
const gateQuestionEl = getById("gateQuestion");
const gateAnswerInputEl = getById("gateAnswerInput");
const gateSubmitButtonEl = getById("gateSubmitButton");
const gateFeedbackEl = getById("gateFeedback");
const gateCancelButtonEl = getById("gateCancelButton");

const wordMissions = [{ word: "GO", emoji: "🚀", label: "ごー" }, { word: "CAT", emoji: "🐱", label: "ねこ" }, { word: "DOG", emoji: "🐶", label: "いぬ" }, { word: "STAR", emoji: "⭐", label: "ほし" }, { word: "MOON", emoji: "🌙", label: "つき" }, { word: "APPLE", emoji: "🍎", label: "りんご" }];

let targetSequence = ["A"]; let targetIndex = 0; let score = 0; let combo = 0; let defeatedCount = 0; let typedLog = [];
let playerHp = 100; let inputDanger = 0; let isGameOver = false; let perfectStreak = 0; let currentMission = null;
let lastInputAt = Date.now(); const idleHintMs = 1700; const clearScoreThreshold = 3000; let isCleared = false;
let isGameStarted = false; let gateAction = null; let gateAnswer = null;

const keyButtons = new Map();
const positionMap = new Map();
letterRows.forEach((row, r) => row.forEach((key, c) => positionMap.set(key, { r, c })));

function currentTarget() { return targetSequence[targetIndex]; }
function neighborsOf(key) { const pos = positionMap.get(key); if (!pos) return []; const list = []; for (let dr = -1; dr <= 1; dr += 1) { for (let dc = -1; dc <= 1; dc += 1) { if (dr === 0 && dc === 0) continue; const row = letterRows[pos.r + dr]; if (!row) continue; const near = row[pos.c + dc]; if (near) list.push(near); } } return list; }
function helperKeysOf(key) { return [...neighborsOf(key), ...(similarKeyMap.get(key) || [])]; }
function keyDistance(fromKey, toKey) {
  const from = positionMap.get(fromKey); const to = positionMap.get(toKey);
  if (!from || !to) return Infinity;
  const dr = Math.abs(from.r - to.r); const dc = Math.abs(from.c - to.c);
  return Math.max(dr, dc);
}
function targetDistanceByCombo() {
  if (combo >= 20 || perfectStreak >= 10) return 3;
  if (combo >= 8 || perfectStreak >= 4) return 2;
  return 1;
}
function randomFrom(list, fallback) { return list.length > 0 ? list[Math.floor(Math.random() * list.length)] : fallback; }
function pickTargetNear(previous) {
  const flat = letterRows.flat();
  if (!previous || !positionMap.has(previous)) return randomFrom(flat, "A");
  const maxDistance = targetDistanceByCombo();
  const candidates = flat.filter((k) => keyDistance(previous, k) <= maxDistance);
  return randomFrom(candidates, previous);
}
function directionArrow(fromKey, toKey) {
  const from = positionMap.get(fromKey); const to = positionMap.get(toKey);
  if (!from || !to) return "";
  const dr = to.r - from.r; const dc = to.c - from.c;
  if (dr === 0 && dc === 0) return "";
  if (dr < 0 && dc === 0) return "↑";
  if (dr > 0 && dc === 0) return "↓";
  if (dr === 0 && dc < 0) return "←";
  if (dr === 0 && dc > 0) return "→";
  if (dr < 0 && dc < 0) return "↖";
  if (dr < 0 && dc > 0) return "↗";
  if (dr > 0 && dc < 0) return "↙";
  return "↘";
}

function buildKeyboard() {
  keyboardEl.innerHTML = "";
  if (keyboardGuideOverlayEl) keyboardGuideOverlayEl.innerHTML = "";
  keyButtons.clear();
  keyboardEl.classList.toggle("layout-jis", settings.layout === "JIS");
  keyboardEl.classList.toggle("layout-us", settings.layout === "US");
  const rows = keyboardLayouts[settings.layout];
  rows.forEach((row) => {
    const visible = row.filter((key) => {
      if (key.group === "function" && !settings.showFunctionKeys) return false;
      if (key.group === "number" && !settings.showNumberRow) return false;
      return true;
    });
    if (visible.length === 0) return;
    const rowEl = document.createElement("div");
    rowEl.className = "keyboard-row";
    const rowAnchor = visible[0]?.id;
    if (rowAnchor === "CAPS") rowEl.classList.add("row-caps");
    visible.forEach((key) => {
      const keyEl = document.createElement("div");
      keyEl.className = `key ${key.size || ""}`.trim();
      if (!/^[A-Z]$/.test(key.id)) keyEl.classList.add("non-input");
      if (settings.layout === "JIS" && key.id === "ENTER") keyEl.classList.add("jis-enter");
      if (settings.layout === "JIS" && (key.id === "[" || key.id === "]")) keyEl.classList.add("jis-bracket");
      if (settings.layout === "JIS" && key.id === "TAB") keyEl.classList.add("jis-tab");
      if (settings.layout === "JIS" && key.id === "CAPS") keyEl.classList.add("jis-caps");
      if (settings.layout === "JIS" && key.id === "SHIFT") keyEl.classList.add("jis-shift");
      keyEl.textContent = key.label;
      rowEl.appendChild(keyEl);
      if (/^[A-Z]$/.test(key.id)) keyButtons.set(key.id, keyEl);
    });
    keyboardEl.appendChild(rowEl);
  });
}

function missionLengthByCombo() { if (combo >= 34 || perfectStreak >= 14) return 5; if (combo >= 24 || perfectStreak >= 10) return 4; if (combo >= 14 || perfectStreak >= 6) return 3; if (combo >= 7 || perfectStreak >= 3) return 2; return 1; }
function updateTargetDisplay() {
  targetKeyEl.innerHTML = targetSequence.map((char, index) => `<span class="letter ${index < targetIndex ? "done" : index === targetIndex ? "current" : "upcoming"}">${char}</span>`).join("");
  if (targetSequence.length >= 2) {
    const missionName = currentMission ? `${currentMission.label} ${currentMission.emoji}` : "ことば";
    targetModeEl.textContent = `${missionName} ${targetIndex + 1}/${targetSequence.length}`;
  } else { targetModeEl.textContent = "ひとつ うつ"; }

  targetGuideEl.textContent = "";
}
function setNextTarget(forceSingle = false) {
  if (forceSingle) {
    currentMission = null;
    targetSequence = ["A"];
  } else {
    currentMission = null;
    const previousKey = typedLog[0] || currentTarget() || "A";
    targetSequence = [pickTargetNear(previousKey)];
  }
  targetIndex = 0; updateTargetDisplay(); renderKeyboard();
}


function keyCenterOf(key) {
  const el = keyButtons.get(key);
  if (!el || !keyboardGuideOverlayEl) return null;
  const keyRect = el.getBoundingClientRect();
  const overlayRect = keyboardGuideOverlayEl.getBoundingClientRect();
  return { x: keyRect.left + keyRect.width / 2 - overlayRect.left, y: keyRect.top + keyRect.height / 2 - overlayRect.top };
}

function renderKeyboardGuide(fromKey, toKey) {
  if (!keyboardGuideOverlayEl) return;
  keyboardGuideOverlayEl.innerHTML = "";
  const from = keyCenterOf(fromKey);
  const to = keyCenterOf(toKey);
  if (!from || !to) return;
  const w = keyboardGuideOverlayEl.clientWidth || 1;
  const h = keyboardGuideOverlayEl.clientHeight || 1;
  keyboardGuideOverlayEl.setAttribute("viewBox", `0 0 ${w} ${h}`);

  const ns = "http://www.w3.org/2000/svg";
  const path = document.createElementNS(ns, "path");
  const cpX = (from.x + to.x) / 2;
  const cpY = Math.min(from.y, to.y) - Math.max(18, Math.abs(to.x - from.x) * 0.08);
  path.setAttribute("d", `M ${from.x} ${from.y} Q ${cpX} ${cpY} ${to.x} ${to.y}`);
  path.setAttribute("class", "guide-path");

  const ring = document.createElementNS(ns, "circle");
  ring.setAttribute("cx", String(to.x));
  ring.setAttribute("cy", String(to.y));
  ring.setAttribute("r", "13");
  ring.setAttribute("class", "guide-ring");

  const angle = Math.atan2(to.y - cpY, to.x - cpX);
  const arrowLen = 14;
  const arrowWidth = 8;
  const bx = to.x - Math.cos(angle) * arrowLen;
  const by = to.y - Math.sin(angle) * arrowLen;
  const lx = bx + Math.cos(angle + Math.PI / 2) * arrowWidth;
  const ly = by + Math.sin(angle + Math.PI / 2) * arrowWidth;
  const rx = bx + Math.cos(angle - Math.PI / 2) * arrowWidth;
  const ry = by + Math.sin(angle - Math.PI / 2) * arrowWidth;

  const arrow = document.createElementNS(ns, "polygon");
  arrow.setAttribute("points", `${to.x},${to.y} ${lx},${ly} ${rx},${ry}`);
  arrow.setAttribute("class", "guide-arrow");

  keyboardGuideOverlayEl.append(path, ring, arrow);
}



function renderKeyboard(pressed = "") {
  const target = currentTarget(); const helper = helperKeysOf(target); const idleElapsed = Date.now() - lastInputAt;
  const isIdle = !pressed && idleElapsed >= idleHintMs; const hintStage = !pressed ? Phaser.Math.Clamp((idleElapsed - idleHintMs) / 2100, 0, 1) : 0;
  const previousKey = typedLog[0] || "";
  const dist = keyDistance(previousKey, target);
  const showGuide = previousKey && Number.isFinite(dist) && dist > 0;
  keyButtons.forEach((el, key) => {
    el.classList.toggle("target", key === target);
    el.classList.toggle("near", helper.includes(key));
    el.classList.toggle("pressed", key === pressed);
    el.classList.toggle("hint", key === target && isIdle);
    if (key === target && isIdle) el.style.setProperty("--hint-stage", hintStage.toFixed(2)); else el.style.removeProperty("--hint-stage");
  });
  if (showGuide) renderKeyboardGuide(previousKey, target);
  else if (keyboardGuideOverlayEl) keyboardGuideOverlayEl.innerHTML = "";
}

function normalizeInputKey(event) { if (event.isComposing) return ""; if (/^Key[A-Z]$/.test(event.code)) return event.code.replace("Key", ""); const key = (event.key || "").toUpperCase(); return /^[A-Z]$/.test(key) ? key : ""; }
function judge(input, target) { const helper = helperKeysOf(target); if (input === target) return "perfect"; if (helper.includes(input)) return "good"; return "miss"; }
function updateLog(input) { typedLog.unshift(input); typedLog = typedLog.slice(0, 24); typedLogEl.textContent = typedLog.join(" "); }
function setFeedback(rating, text) { feedbackEl.className = rating || ""; feedbackEl.textContent = text; targetPanelEl.classList.remove("flash", "flash-perfect", "flash-good", "flash-miss"); }
function setInputDanger(next) { inputDanger = Phaser.Math.Clamp(next, 0, 100); inputGaugeFillEl.style.width = `${inputDanger}%`; }
function setPlayerHp(next) { playerHp = Phaser.Math.Clamp(next, 0, 100); playerHpFillEl.style.width = `${playerHp}%`; }
function consumeDangerByRating(rating) { const deltaMap = { perfect: -34, good: -22, miss: 13 }; setInputDanger(inputDanger + (deltaMap[rating] || 0)); }
function showRetryOverlay() { retryScoreTextEl.textContent = `すこあ ${score} / たおしたかず ${defeatedCount}`; retryOverlayEl.classList.remove("hidden"); }
function hideRetryOverlay() { retryOverlayEl.classList.add("hidden"); }
function showClearOverlay() { clearScoreTextEl.textContent = `すこあ ${score} / たおしたかず ${defeatedCount}`; clearOverlayEl.classList.remove("hidden"); }
function hideClearOverlay() { clearOverlayEl.classList.add("hidden"); }

function resetGameState() {
  targetSequence = ["A"]; targetIndex = 0; score = 0; combo = 0; defeatedCount = 0; typedLog = []; isGameOver = false; perfectStreak = 0; currentMission = null;
  lastInputAt = Date.now(); isCleared = false; scoreEl.textContent = "0"; comboEl.textContent = "0"; waveEl.textContent = "0"; perfectStreakEl.textContent = "0";
  typedLogEl.textContent = "-"; targetGuideEl.textContent = ""; setInputDanger(20); setPlayerHp(100); setFeedback("", "キーを押して攻撃！"); setNextTarget(true); hideClearOverlay(); hideRetryOverlay();
  if (sceneRef && sceneRef.resetBattle) sceneRef.resetBattle();
}

function triggerGameOver() { if (isGameOver) return; isGameOver = true; setFeedback("miss", "げきつい…！もういちど ちょうせん"); if (sceneRef?.playGameOverEffect) sceneRef.playGameOverEffect(); showRetryOverlay(); }
class Sfx { constructor() { this.ctx = null; } unlock() { if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)(); if (this.ctx.state === "suspended") this.ctx.resume(); }
  tone(freq, duration = 0.09, type = "square", gain = 0.06) { if (!this.ctx) return; const now = this.ctx.currentTime; const osc = this.ctx.createOscillator(); const amp = this.ctx.createGain(); osc.type = type; osc.frequency.setValueAtTime(freq, now); amp.gain.setValueAtTime(gain, now); amp.gain.exponentialRampToValueAtTime(0.001, now + duration); osc.connect(amp).connect(this.ctx.destination); osc.start(now); osc.stop(now + duration); }
  perfect() { this.unlock(); this.tone(760, 0.08, "triangle", 0.07); this.tone(980, 0.1, "triangle", 0.06); }
  good() { this.unlock(); this.tone(520, 0.09, "sine", 0.06); }
  miss() { this.unlock(); this.tone(190, 0.11, "sawtooth", 0.05); }
  playerHit() { this.unlock(); this.tone(140, 0.12, "square", 0.07); this.tone(110, 0.14, "square", 0.06); }
}
const sfx = new Sfx(); let sceneRef = null;

function viewportHeight() { return window.visualViewport ? window.visualViewport.height : window.innerHeight; }
function viewportWidth() { return window.visualViewport ? window.visualViewport.width : window.innerWidth; }
function syncAppHeightVar() { document.documentElement.style.setProperty("--app-height", `${Math.round(viewportHeight())}px`); }
function syncGameSize() { syncAppHeightVar(); if (!sceneRef?.scale) return; sceneRef.scale.resize(Math.round(viewportWidth()), Math.round(viewportHeight())); }

class InvaderScene extends Phaser.Scene {
  constructor() { super("invader"); }
  create() {
    const { width, height } = this.scale;
    this.cameras.main.setBackgroundColor("#060c1a");
    this.enemyPoints = [{ x: 0.18, y: 0.3 }, { x: 0.84, y: 0.24 }, { x: 0.16, y: 0.42 }, { x: 0.86, y: 0.38 }, { x: 0.2, y: 0.2 }, { x: 0.82, y: 0.46 }]; this.enemyPointIndex = 0;
    this.starA = this.add.tileSprite(0, 0, width, height, this.makeStarTexture(0x4d6aa8, 2)).setOrigin(0);
    this.starB = this.add.tileSprite(0, 0, width, height, this.makeStarTexture(0x84a1dd, 2)).setOrigin(0);
    this.grid = this.add.tileSprite(0, 0, width, height, this.makeGridTexture()).setOrigin(0).setAlpha(0.26);
    this.enemyMonsters = ["👾", "👹", "🤖", "🦖", "🐙", "🦇", "👻"]; this.enemyLabels = ["いんべーだー", "おに", "ろぼ", "きょうりゅう", "たこ", "こうもり", "おばけ"]; this.enemyMonsterIndex = 0;
    this.enemy = this.add.text(width * 0.78, height * 0.28, this.enemyMonsters[0], { fontSize: `${Math.max(84, Math.floor(width * 0.08))}px` }).setOrigin(0.5);
    this.enemySwayTween = this.tweens.add({
      targets: this.enemy,
      x: this.enemy.x + 12,
      duration: 760,
      yoyo: true,
      repeat: -1,
      ease: "Sine.InOut",
      onYoyo: () => { this.enemy.setScale(-1, 1); },
      onRepeat: () => { this.enemy.setScale(1, 1); },
    });
    this.player = this.add.text(width * 0.14, height * 0.88, "🛸", { fontSize: `${Math.max(56, Math.floor(width * 0.05))}px` }).setOrigin(0.5);
    this.enemyName = this.add.text(width * 0.78, height * 0.18, "てき: いんべーだー", { fontFamily: "monospace", fontSize: "20px", color: "#a9c1ff" }).setOrigin(0.5);
    this.enemyHpDots = this.add.text(width * 0.78, height * 0.36, "● ● ●", { fontFamily: "monospace", fontSize: "22px", color: "#b8ebff", stroke: "#000000", strokeThickness: 4 }).setOrigin(0.5).setAlpha(0.88);
    this.impactText = this.add.text(width * 0.78, height * 0.11, "", { fontFamily: "monospace", fontSize: "52px", color: "#ffffff", stroke: "#000000", strokeThickness: 8 }).setOrigin(0.5).setAlpha(0);
    this.attackWarning = this.add.text(width * 0.5, height * 0.48, "", { fontFamily: "monospace", fontSize: "56px", color: "#ff6585", stroke: "#0c0310", strokeThickness: 8 }).setOrigin(0.5).setAlpha(0);
    this.scale.on("resize", () => this.handleResize()); this.enemyHp = 100; this.updateEnemyHpDots(); sceneRef = this; this.moveEnemyToNewSpot(false);
  }
  makeStarTexture(color, size) { const key = `star-${color}-${size}`; if (this.textures.exists(key)) return key; const g = this.make.graphics({ x: 0, y: 0, add: false }); g.fillStyle(0x000000, 0); g.fillRect(0, 0, 256, 256); g.fillStyle(color, 1); for (let i = 0; i < 120; i += 1) g.fillCircle(Phaser.Math.Between(0, 255), Phaser.Math.Between(0, 255), size); g.generateTexture(key, 256, 256); g.destroy(); return key; }
  makeGridTexture() { const key = "grid-tex"; if (this.textures.exists(key)) return key; const g = this.make.graphics({ x: 0, y: 0, add: false }); g.lineStyle(1, 0xffffff, 0.5); for (let x = 0; x <= 128; x += 16) g.lineBetween(x, 0, x, 128); for (let y = 0; y <= 128; y += 16) g.lineBetween(0, y, 128, y); g.generateTexture(key, 128, 128); g.destroy(); return key; }
  moveEnemyToNewSpot(animate = true) { let nextIndex = Phaser.Math.Between(0, this.enemyPoints.length - 1); if (nextIndex === this.enemyPointIndex) nextIndex = (nextIndex + 1) % this.enemyPoints.length; this.enemyPointIndex = nextIndex; const { width, height } = this.scale; const p = this.enemyPoints[nextIndex]; const nx = width * p.x; const ny = height * p.y; if (!animate) { this.enemy.setPosition(nx, ny); this.syncEnemyHud(); return; } this.tweens.add({ targets: this.enemy, x: nx, y: ny, duration: 320, ease: "Cubic.Out", onUpdate: () => this.syncEnemyHud() }); }
  syncEnemyHud() { this.enemyName.setPosition(this.enemy.x, this.enemy.y - 100); this.enemyHpDots.setPosition(this.enemy.x, this.enemy.y + 78); this.impactText.setPosition(this.enemy.x, this.enemy.y - 126); }
  enemyLifePips() { return Phaser.Math.Clamp(Math.ceil(this.enemyHp / 34), 0, 3); }
  updateEnemyHpDots() { const lives = this.enemyLifePips(); this.enemyHpDots.setText(Array.from({ length: 3 }, (_, i) => (i < lives ? "◆" : "◇")).join(" ")); }
  hitStop(target, mode = "enemy") { this.cameras.main.shake(130, mode === "enemy" ? 0.005 : 0.012); this.tweens.add({ targets: target, x: target.x + Phaser.Math.Between(-8, 8), y: target.y + Phaser.Math.Between(-7, 7), yoyo: true, repeat: 1, duration: 28 }); }
  playGameOverEffect() { this.cameras.main.flash(100, 255, 70, 95, true); }
  playAttackFlash(rating) { const tint = rating === "perfect" ? [70, 255, 180] : [90, 220, 255]; this.cameras.main.flash(80, tint[0], tint[1], tint[2], true); }
  playPlayerHitFlash() { this.cameras.main.flash(120, 255, 72, 92, true); }
  spawnImpactBurst(x, y, color = 0xff77aa) { for (let i = 0; i < 8; i += 1) { const dot = this.add.circle(x, y, Phaser.Math.Between(3, 6), color, 0.95); const a = Phaser.Math.FloatBetween(0, Math.PI * 2); const dist = Phaser.Math.Between(40, 95); this.tweens.add({ targets: dot, x: x + Math.cos(a) * dist, y: y + Math.sin(a) * dist, alpha: 0, scale: 0.2, duration: 380, ease: "Cubic.Out", onComplete: () => dot.destroy() }); } }
  fireLetter(letter, rating) { const { width, height } = this.scale; const colors = { perfect: "#61ffb6", good: "#5fe5ff", miss: "#a6b2cb" }; const damageMap = { perfect: 28, good: 16, miss: 0 }; const bullet = this.add.text(width * 0.18, height * 0.88, letter, { fontFamily: "monospace", fontSize: rating === "perfect" ? "72px" : "56px", color: colors[rating], stroke: "#000", strokeThickness: 8 }).setOrigin(0.5).setScale(0.5); this.tweens.add({ targets: bullet, x: this.enemy.x - 28, y: this.enemy.y + 8, scale: 1.18, alpha: { from: 0.45, to: 1 }, duration: rating === "perfect" ? 280 : 360, onComplete: () => bullet.destroy() }); const damage = damageMap[rating]; if (damage > 0) { this.hitStop(this.enemy); this.playAttackFlash(rating); this.applyDamage(damage); } }
  fireWordBonus(emoji) { const badge = this.add.text(this.player.x + 24, this.player.y - 36, emoji, { fontSize: "56px", stroke: "#000", strokeThickness: 6 }).setOrigin(0.5); this.tweens.add({ targets: badge, x: this.enemy.x, y: this.enemy.y - 10, alpha: 0, duration: 420, onComplete: () => { this.applyDamage(22); badge.destroy(); } }); }
  applyDamage(damage) { this.enemyHp = Math.max(0, this.enemyHp - damage); this.updateEnemyHpDots(); if (this.enemyHp === 0) { defeatedCount += 1; waveEl.textContent = String(defeatedCount); this.enemyMonsterIndex = (this.enemyMonsterIndex + 1) % this.enemyMonsters.length; this.enemy.setText(this.enemyMonsters[this.enemyMonsterIndex]); this.enemyName.setText(`てき: ${this.enemyLabels[this.enemyMonsterIndex]}`); this.enemyHp = 100; this.updateEnemyHpDots(); this.moveEnemyToNewSpot(); } }
  enemyAttack() { if (isGameOver || !isGameStarted) return; setPlayerHp(playerHp - 18); sfx.playerHit(); if (playerHp <= 0) triggerGameOver(); }
  resetBattle() { this.enemyHp = 100; this.updateEnemyHpDots(); this.enemyPointIndex = 0; this.moveEnemyToNewSpot(false); }
  update(_, delta) { this.starA.tilePositionY -= 0.05 * delta; this.starB.tilePositionY -= 0.1 * delta; this.grid.tilePositionY -= 0.04 * delta; this.syncEnemyHud(); if (isGameOver || !isGameStarted) return; setInputDanger(inputDanger + delta * 0.008); if (inputDanger >= 100) { setInputDanger(42); this.enemyAttack(); } }
  handleResize() { const { width, height } = this.scale; this.starA.setSize(width, height); this.starB.setSize(width, height); this.grid.setSize(width, height); this.player.setPosition(width * 0.14, height * 0.88); const p = this.enemyPoints[this.enemyPointIndex]; this.enemy.setPosition(width * p.x, height * p.y); this.syncEnemyHud(); this.updateEnemyHpDots(); this.attackWarning.setPosition(width * 0.5, height * 0.48); }
}

function handleScore(rating, missionComplete = false) {
  if (rating === "perfect") { perfectStreak += 1; score += 120 + combo * 4; combo += 1; setFeedback("perfect", missionComplete ? "かんぺき！ことば せいこう！" : "かんぺき！"); sfx.perfect(); }
  else if (rating === "good") { perfectStreak = 0; score += 70 + combo * 2; combo += 1; setFeedback("good", missionComplete ? "いいね！ことば せいこう！" : "いいね！"); sfx.good(); }
  else { perfectStreak = 0; score += 10; combo = 0; setFeedback("miss", "ざんねん！"); sfx.miss(); }
  scoreEl.textContent = String(score); comboEl.textContent = String(combo); perfectStreakEl.textContent = String(perfectStreak);
  if (!isCleared && score >= clearScoreThreshold) { isCleared = true; isGameOver = true; showClearOverlay(); }
}

function showGate(action) {
  gateAction = action;
  const a = Phaser.Math.Between(2, 9); const b = Phaser.Math.Between(1, 9);
  gateAnswer = a * b;
  gateQuestionEl.textContent = `${a} × ${b} = ?`;
  gateAnswerInputEl.value = "";
  gateFeedbackEl.textContent = "かけざんの こたえを いれてください";
  gateOverlayEl.classList.remove("hidden");
  setTimeout(() => gateAnswerInputEl.focus(), 40);
}

function submitGateAnswer() {
  const entered = Number(gateAnswerInputEl.value);
  if (!Number.isFinite(entered)) {
    gateFeedbackEl.textContent = "すうじを いれてください";
    return;
  }
  if (entered !== gateAnswer) {
    gateFeedbackEl.textContent = "ちがいます。もういちど";
    gateAnswerInputEl.select();
    return;
  }
  gateOverlayEl.classList.add("hidden");
  if (gateAction === "settings") settingsOverlayEl.classList.remove("hidden");
}

document.addEventListener("keydown", (event) => {
  if (isGameOver || !isGameStarted) return;
  const input = normalizeInputKey(event); if (!input) return;
  sfx.unlock(); lastInputAt = Date.now();
  const target = currentTarget(); const rating = judge(input, target); updateLog(input); renderKeyboard(input); consumeDangerByRating(rating);
  if (rating === "miss") { handleScore(rating, false); sceneRef?.fireLetter(input, rating); targetIndex = 0; updateTargetDisplay(); setTimeout(() => renderKeyboard(), 120); return; }
  const wasLastStep = targetIndex === targetSequence.length - 1; sceneRef?.fireLetter(input, rating);
  if (wasLastStep) { handleScore(rating, targetSequence.length >= 2); if (currentMission) sceneRef?.fireWordBonus(currentMission.emoji); setTimeout(() => setNextTarget(), 240); }
  else { handleScore(rating, false); targetIndex += 1; updateTargetDisplay(); }
  setTimeout(() => renderKeyboard(), 120);
});

startButtonEl.addEventListener("click", () => {
  startOverlayEl.classList.add("hidden");
  isGameStarted = true;
  resetGameState();
});
openSettingsButtonEl.addEventListener("click", () => showGate("settings"));
closeSettingsButtonEl.addEventListener("click", () => settingsOverlayEl.classList.add("hidden"));
gateCancelButtonEl.addEventListener("click", () => gateOverlayEl.classList.add("hidden"));
gateSubmitButtonEl.addEventListener("click", submitGateAnswer);
gateAnswerInputEl.addEventListener("keydown", (event) => { if (event.key === "Enter") submitGateAnswer(); });
applySettingsButtonEl.addEventListener("click", () => {
  settings.layout = layoutSelectEl.value;
  settings.showFunctionKeys = toggleFunctionRowEl.checked;
  settings.showNumberRow = toggleNumberRowEl.checked;
  buildKeyboard();
  renderKeyboard();
  settingsOverlayEl.classList.add("hidden");
});

retryButtonEl.addEventListener("click", () => { hideRetryOverlay(); resetGameState(); });
clearButtonEl.addEventListener("click", () => { hideClearOverlay(); resetGameState(); });

syncAppHeightVar();
window.addEventListener("resize", syncGameSize);
if (window.visualViewport) { window.visualViewport.addEventListener("resize", syncGameSize); window.visualViewport.addEventListener("scroll", syncGameSize); }
new Phaser.Game({ type: Phaser.AUTO, parent: "gameRoot", width: Math.round(viewportWidth()), height: Math.round(viewportHeight()), backgroundColor: "#060c1a", scene: InvaderScene, scale: { mode: Phaser.Scale.RESIZE, autoCenter: Phaser.Scale.CENTER_BOTH } });

setInputDanger(20); setPlayerHp(100); buildKeyboard(); setNextTarget(true); hideRetryOverlay(); hideClearOverlay();
setInterval(() => { if (!isGameOver && isGameStarted) renderKeyboard(); }, 180);
