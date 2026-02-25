const rows = [
  ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
  ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
  ["Z", "X", "C", "V", "B", "N", "M"],
];

const similarKeyGroups = [
  ["O", "D", "Q"],
  ["P", "R", "B"],
  ["I", "L", "J"],
  ["U", "V", "Y"],
  ["C", "G"],
  ["M", "N"],
  ["S", "Z"],
  ["K", "X"],
];

const similarKeyMap = new Map();
similarKeyGroups.forEach((group) => {
  group.forEach((key) => {
    const siblings = group.filter((candidate) => candidate !== key);
    similarKeyMap.set(key, siblings);
  });
});

const targetKeyEl = document.getElementById("targetKey");
const targetModeEl = document.getElementById("targetMode");
const feedbackEl = document.getElementById("feedback");
const typedLogEl = document.getElementById("typedLog");
const scoreEl = document.getElementById("score");
const comboEl = document.getElementById("combo");
const waveEl = document.getElementById("wave");
const perfectStreakEl = document.getElementById("perfectStreak");
const keyboardEl = document.getElementById("keyboard");
const inputGaugeFillEl = document.getElementById("inputGaugeFill");
const playerHpFillEl = document.getElementById("playerHpFill");
const retryOverlayEl = document.getElementById("retryOverlay");
const retryButtonEl = document.getElementById("retryButton");
const retryScoreTextEl = document.getElementById("retryScoreText");
const clearOverlayEl = document.getElementById("clearOverlay");
const clearButtonEl = document.getElementById("clearButton");
const clearScoreTextEl = document.getElementById("clearScoreText");
const targetPanelEl = document.querySelector(".target-panel");

const wordMissions = [
  { word: "GO", emoji: "🚀", label: "ごー" },
  { word: "CAT", emoji: "🐱", label: "ねこ" },
  { word: "DOG", emoji: "🐶", label: "いぬ" },
  { word: "STAR", emoji: "⭐", label: "ほし" },
  { word: "MOON", emoji: "🌙", label: "つき" },
  { word: "APPLE", emoji: "🍎", label: "りんご" },
];

let targetSequence = ["A"];
let targetIndex = 0;
let score = 0;
let combo = 0;
let defeatedCount = 0;
let typedLog = [];
let playerHp = 100;
let inputDanger = 0;
let isGameOver = false;
let perfectStreak = 0;
let currentMission = null;
let lastInputAt = Date.now();
const idleHintMs = 1700;
const clearScoreThreshold = 3000;
let isCleared = false;

const keyButtons = new Map();
const positionMap = new Map();
rows.forEach((row, r) => row.forEach((key, c) => positionMap.set(key, { r, c })));

function currentTarget() {
  return targetSequence[targetIndex];
}

function buildKeyboard() {
  rows.forEach((row) => {
    const rowEl = document.createElement("div");
    rowEl.className = "keyboard-row";
    row.forEach((key) => {
      const keyEl = document.createElement("div");
      keyEl.className = "key";
      keyEl.textContent = key;
      rowEl.appendChild(keyEl);
      keyButtons.set(key, keyEl);
    });
    keyboardEl.appendChild(rowEl);
  });
}

function neighborsOf(key) {
  const pos = positionMap.get(key);
  if (!pos) return [];

  const list = [];
  for (let dr = -1; dr <= 1; dr += 1) {
    for (let dc = -1; dc <= 1; dc += 1) {
      if (dr === 0 && dc === 0) continue;
      const row = rows[pos.r + dr];
      if (!row) continue;
      const near = row[pos.c + dc];
      if (near) list.push(near);
    }
  }
  return list;
}

function helperKeysOf(key) {
  return [...neighborsOf(key), ...(similarKeyMap.get(key) || [])];
}

function missionLengthByCombo() {
  if (combo >= 24 || perfectStreak >= 10) return 4;
  if (combo >= 14 || perfectStreak >= 6) return 3;
  if (combo >= 7 || perfectStreak >= 3) return 2;
  return 1;
}

function updateTargetDisplay() {
  const chars = targetSequence
    .map((char, index) => {
      if (index < targetIndex) return `<span class="letter done">${char}</span>`;
      if (index === targetIndex) return `<span class="letter current">${char}</span>`;
      return `<span class="letter upcoming">${char}</span>`;
    })
    .join("");

  targetKeyEl.innerHTML = chars;
  targetKeyEl.dataset.mode = targetSequence.length >= 2 ? "word" : "single";

  if (targetSequence.length >= 2) {
    const missionName = currentMission ? `${currentMission.label} ${currentMission.emoji}` : "ことば";
    targetModeEl.textContent = `${missionName} ${targetIndex + 1}/${targetSequence.length}`;
  } else {
    targetModeEl.textContent = "ひとつ うつ";
  }
}

function setNextTarget(forceSingle = false) {
  const flat = rows.flat();
  if (forceSingle) {
    currentMission = null;
    targetSequence = [flat[Math.floor(Math.random() * flat.length)]];
  } else {
    const unlockedLength = missionLengthByCombo();
    const pool = wordMissions.filter((mission) => mission.word.length <= unlockedLength);
    const shouldUseMission = unlockedLength >= 2 && Math.random() < 0.62;

    if (shouldUseMission && pool.length > 0) {
      currentMission = pool[Math.floor(Math.random() * pool.length)];
      targetSequence = currentMission.word.split("");
    } else {
      currentMission = null;
      targetSequence = [flat[Math.floor(Math.random() * flat.length)]];
    }
  }

  targetIndex = 0;
  updateTargetDisplay();
  renderKeyboard();
}

function renderKeyboard(pressed = "") {
  const target = currentTarget();
  const helper = helperKeysOf(target);
  const idleElapsed = Date.now() - lastInputAt;
  const isIdle = !pressed && idleElapsed >= idleHintMs;
  const hintStage = !pressed ? Phaser.Math.Clamp((idleElapsed - idleHintMs) / 2100, 0, 1) : 0;
  keyButtons.forEach((el, key) => {
    el.classList.toggle("target", key === target);
    el.classList.toggle("near", helper.includes(key));
    el.classList.toggle("pressed", key === pressed);
    el.classList.toggle("hint", key === target && isIdle);
    if (key === target && isIdle) {
      el.style.setProperty("--hint-stage", hintStage.toFixed(2));
    } else {
      el.style.removeProperty("--hint-stage");
    }
  });
}

function normalizeInputKey(event) {
  if (event.isComposing) return "";
  if (/^Key[A-Z]$/.test(event.code)) return event.code.replace("Key", "");
  const key = (event.key || "").toUpperCase();
  return /^[A-Z]$/.test(key) ? key : "";
}

function judge(input, target) {
  const helper = helperKeysOf(target);
  if (input === target) return "perfect";
  if (helper.includes(input)) return "good";
  return "miss";
}

function updateLog(input) {
  typedLog.unshift(input);
  typedLog = typedLog.slice(0, 24);
  typedLogEl.textContent = typedLog.join(" ");
}

function setFeedback(rating, text) {
  feedbackEl.className = rating ? rating : "";
  feedbackEl.textContent = text;
  feedbackEl.classList.remove("pulse");
  targetPanelEl.classList.remove("flash", "flash-perfect", "flash-good", "flash-miss");
  void feedbackEl.offsetWidth;
  feedbackEl.classList.add("pulse");
  targetPanelEl.classList.add("flash");
  if (rating) targetPanelEl.classList.add(`flash-${rating}`);
}

function setInputDanger(next) {
  inputDanger = Phaser.Math.Clamp(next, 0, 100);
  inputGaugeFillEl.style.width = `${inputDanger}%`;
}

function setPlayerHp(next) {
  playerHp = Phaser.Math.Clamp(next, 0, 100);
  playerHpFillEl.style.width = `${playerHp}%`;
}

function consumeDangerByRating(rating) {
  const deltaMap = { perfect: -34, good: -22, miss: 13 };
  setInputDanger(inputDanger + (deltaMap[rating] || 0));
}

function showRetryOverlay() {
  retryScoreTextEl.textContent = `すこあ ${score} / たおしたかず ${defeatedCount}`;
  retryOverlayEl.classList.remove("hidden");
  retryOverlayEl.setAttribute("aria-hidden", "false");
}

function hideRetryOverlay() {
  retryOverlayEl.classList.add("hidden");
  retryOverlayEl.setAttribute("aria-hidden", "true");
}

function showClearOverlay() {
  clearScoreTextEl.textContent = `すこあ ${score} / たおしたかず ${defeatedCount}`;
  clearOverlayEl.classList.remove("hidden");
  clearOverlayEl.setAttribute("aria-hidden", "false");
}

function hideClearOverlay() {
  clearOverlayEl.classList.add("hidden");
  clearOverlayEl.setAttribute("aria-hidden", "true");
}

function resetGameState() {
  targetSequence = ["A"];
  targetIndex = 0;
  score = 0;
  combo = 0;
  defeatedCount = 0;
  typedLog = [];
  isGameOver = false;
  perfectStreak = 0;
  currentMission = null;
  lastInputAt = Date.now();
  isCleared = false;
  scoreEl.textContent = "0";
  comboEl.textContent = "0";
  waveEl.textContent = "0";
  perfectStreakEl.textContent = "0";
  typedLogEl.textContent = "-";
  setInputDanger(20);
  setPlayerHp(100);
  setFeedback("", "キーを押して攻撃！");
  setNextTarget(true);
  hideClearOverlay();
  if (sceneRef && sceneRef.resetBattle) sceneRef.resetBattle();
}

function triggerGameOver() {
  if (isGameOver) return;
  isGameOver = true;
  setFeedback("miss", "げきつい…！もういちど ちょうせん");
  if (sceneRef && sceneRef.playGameOverEffect) sceneRef.playGameOverEffect();
  showRetryOverlay();
}

class Sfx {
  constructor() {
    this.ctx = null;
  }

  unlock() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === "suspended") this.ctx.resume();
  }

  tone(freq, duration = 0.09, type = "square", gain = 0.06) {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const amp = this.ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, now);
    amp.gain.setValueAtTime(gain, now);
    amp.gain.exponentialRampToValueAtTime(0.001, now + duration);
    osc.connect(amp).connect(this.ctx.destination);
    osc.start(now);
    osc.stop(now + duration);
  }

  perfect() {
    this.unlock();
    this.tone(760, 0.08, "triangle", 0.07);
    this.tone(980, 0.1, "triangle", 0.06);
  }

  good() {
    this.unlock();
    this.tone(520, 0.09, "sine", 0.06);
  }

  miss() {
    this.unlock();
    this.tone(190, 0.11, "sawtooth", 0.05);
  }

  playerHit() {
    this.unlock();
    this.tone(140, 0.12, "square", 0.07);
    this.tone(110, 0.14, "square", 0.06);
  }
}

const sfx = new Sfx();

let sceneRef = null;

function viewportHeight() {
  return window.visualViewport ? window.visualViewport.height : window.innerHeight;
}

function viewportWidth() {
  return window.visualViewport ? window.visualViewport.width : window.innerWidth;
}

function syncAppHeightVar() {
  const vh = Math.round(viewportHeight());
  document.documentElement.style.setProperty("--app-height", `${vh}px`);
}

function syncGameSize() {
  syncAppHeightVar();
  if (!sceneRef || !sceneRef.scale) return;
  sceneRef.scale.resize(Math.round(viewportWidth()), Math.round(viewportHeight()));
}


class InvaderScene extends Phaser.Scene {
  constructor() {
    super("invader");
  }

  create() {
    const { width, height } = this.scale;

    this.cameras.main.setBackgroundColor("#060c1a");
    this.hitStopTimeout = null;
    this.enemyPoints = [
      { x: 0.78, y: 0.28 },
      { x: 0.62, y: 0.22 },
      { x: 0.76, y: 0.4 },
      { x: 0.58, y: 0.35 },
      { x: 0.72, y: 0.18 },
    ];
    this.enemyPointIndex = 0;

    this.starA = this.add.tileSprite(0, 0, width, height, this.makeStarTexture(0x4d6aa8, 2)).setOrigin(0);
    this.starB = this.add.tileSprite(0, 0, width, height, this.makeStarTexture(0x84a1dd, 2)).setOrigin(0);
    this.grid = this.add.tileSprite(0, 0, width, height, this.makeGridTexture()).setOrigin(0).setAlpha(0.26);

    this.enemyMonsters = ["👾", "👹", "🤖", "🦖", "🐙", "🦇", "👻"];
    this.enemyLabels = ["いんべーだー", "おに", "ろぼ", "きょうりゅう", "たこ", "こうもり", "おばけ"];
    this.enemyMonsterIndex = 0;

    this.enemy = this.add.text(width * 0.78, height * 0.28, this.enemyMonsters[0], {
      fontSize: `${Math.max(84, Math.floor(width * 0.08))}px`,
    }).setOrigin(0.5);

    this.enemyFloatTween = this.tweens.add({
      targets: this.enemy,
      y: this.enemy.y + 12,
      duration: 680,
      yoyo: true,
      repeat: -1,
      ease: "Sine.InOut",
    });

    this.player = this.add.text(width * 0.14, height * 0.8, "🛸", {
      fontSize: `${Math.max(56, Math.floor(width * 0.05))}px`,
    }).setOrigin(0.5);

    this.enemyName = this.add.text(width * 0.78, height * 0.18, "てき: いんべーだー", {
      fontFamily: "monospace",
      fontSize: "20px",
      color: "#a9c1ff",
    }).setOrigin(0.5);

    this.hpBg = this.add.rectangle(width * 0.78, height * 0.22, width * 0.24, 16, 0x1a2547).setStrokeStyle(1, 0x5e75b6);
    this.hpBar = this.add.rectangle(this.hpBg.x - this.hpBg.width / 2, this.hpBg.y, this.hpBg.width, 12, 0x44dd77).setOrigin(0, 0.5);

    this.impactText = this.add.text(width * 0.78, height * 0.11, "", {
      fontFamily: "monospace",
      fontSize: "52px",
      color: "#ffffff",
      stroke: "#000000",
      strokeThickness: 8,
    }).setOrigin(0.5).setAlpha(0);

    this.attackWarning = this.add.text(width * 0.5, height * 0.48, "", {
      fontFamily: "monospace",
      fontSize: "56px",
      color: "#ff6585",
      stroke: "#0c0310",
      strokeThickness: 8,
    }).setOrigin(0.5).setAlpha(0);

    this.resizeHandler = () => this.handleResize();
    this.scale.on("resize", this.resizeHandler);
    this.enemyHp = 100;
    sceneRef = this;
    this.moveEnemyToNewSpot(false);
  }

  makeStarTexture(color, size) {
    const key = `star-${color}-${size}`;
    if (this.textures.exists(key)) return key;

    const g = this.make.graphics({ x: 0, y: 0, add: false });
    g.fillStyle(0x000000, 0);
    g.fillRect(0, 0, 256, 256);
    g.fillStyle(color, 1);
    for (let i = 0; i < 120; i += 1) {
      g.fillCircle(Phaser.Math.Between(0, 255), Phaser.Math.Between(0, 255), size);
    }
    g.generateTexture(key, 256, 256);
    g.destroy();
    return key;
  }

  makeGridTexture() {
    const key = "grid-tex";
    if (this.textures.exists(key)) return key;
    const g = this.make.graphics({ x: 0, y: 0, add: false });
    g.clear();
    g.lineStyle(1, 0xffffff, 0.5);
    for (let x = 0; x <= 128; x += 16) g.lineBetween(x, 0, x, 128);
    for (let y = 0; y <= 128; y += 16) g.lineBetween(0, y, 128, y);
    g.generateTexture(key, 128, 128);
    g.destroy();
    return key;
  }

  moveEnemyToNewSpot(animate = true) {
    let nextIndex = Phaser.Math.Between(0, this.enemyPoints.length - 1);
    if (nextIndex === this.enemyPointIndex) nextIndex = (nextIndex + 1) % this.enemyPoints.length;
    this.enemyPointIndex = nextIndex;

    const { width, height } = this.scale;
    const p = this.enemyPoints[nextIndex];
    const nx = width * p.x;
    const ny = height * p.y;

    if (!animate) {
      this.enemy.setPosition(nx, ny);
      this.refreshEnemyFloat();
      this.syncEnemyHud();
      return;
    }

    this.tweens.add({
      targets: this.enemy,
      x: nx,
      y: ny,
      duration: 320,
      ease: "Cubic.Out",
      onUpdate: () => this.syncEnemyHud(),
      onComplete: () => {
        this.refreshEnemyFloat();
        this.syncEnemyHud();
      },
    });
  }

  refreshEnemyFloat() {
    if (this.enemyFloatTween) this.enemyFloatTween.stop();
    this.enemyFloatTween = this.tweens.add({
      targets: this.enemy,
      y: this.enemy.y + 12,
      duration: 680,
      yoyo: true,
      repeat: -1,
      ease: "Sine.InOut",
    });
  }

  syncEnemyHud() {
    this.enemyName.setPosition(this.enemy.x, this.enemy.y - 70);
    this.hpBg.setPosition(this.enemy.x, this.enemy.y - 34);
    this.hpBar.setPosition(this.hpBg.x - this.hpBg.width / 2, this.hpBg.y);
    this.impactText.setPosition(this.enemy.x, this.enemy.y - 116);
  }

  hitStop(target, mode = "enemy") {
    if (this.hitStopTimeout) clearTimeout(this.hitStopTimeout);

    this.cameras.main.shake(130, mode === "enemy" ? 0.005 : 0.012);
    this.tweens.add({
      targets: target,
      x: target.x + Phaser.Math.Between(-8, 8),
      y: target.y + Phaser.Math.Between(-7, 7),
      yoyo: true,
      repeat: 1,
      duration: 28,
    });

    this.tweens.timeScale = 0.05;
    this.hitStopTimeout = setTimeout(() => {
      this.tweens.timeScale = 1;
      this.hitStopTimeout = null;
    }, mode === "enemy" ? 85 : 120);
  }

  playGameOverEffect() {
    this.cameras.main.flash(100, 255, 70, 95, true);
    this.hitStop(this.player, "player");
    this.cameras.main.shake(260, 0.02);
    this.attackWarning.setText("しすてむ だうん");
    this.attackWarning.setAlpha(1);
    this.tweens.add({
      targets: this.attackWarning,
      alpha: 0,
      duration: 900,
      ease: "Quad.Out",
    });
    this.spawnImpactBurst(this.player.x + 10, this.player.y, 0xff4a67);
  }

  playAttackFlash(rating) {
    const tint = rating === "perfect" ? [70, 255, 180] : [90, 220, 255];
    this.cameras.main.flash(80, tint[0], tint[1], tint[2], true);
  }

  playPlayerHitFlash() {
    this.cameras.main.flash(120, 255, 72, 92, true);
  }

  spawnImpactBurst(x, y, color = 0xff77aa) {
    for (let i = 0; i < 8; i += 1) {
      const dot = this.add.circle(x, y, Phaser.Math.Between(3, 6), color, 0.95);
      const a = Phaser.Math.FloatBetween(0, Math.PI * 2);
      const dist = Phaser.Math.Between(40, 95);
      this.tweens.add({
        targets: dot,
        x: x + Math.cos(a) * dist,
        y: y + Math.sin(a) * dist,
        alpha: 0,
        scale: 0.2,
        duration: 380,
        ease: "Cubic.Out",
        onComplete: () => dot.destroy(),
      });
    }
  }

  fireLetter(letter, rating) {
    const { width, height } = this.scale;
    const colors = { perfect: "#61ffb6", good: "#5fe5ff", miss: "#a6b2cb" };
    const damageMap = { perfect: 28, good: 16, miss: 0 };

    const bullet = this.add.text(width * 0.18, height * 0.8, letter, {
      fontFamily: "monospace",
      fontSize: rating === "perfect" ? "72px" : "56px",
      color: colors[rating],
      stroke: "#000",
      strokeThickness: 8,
    }).setOrigin(0.5).setScale(0.5);

    this.tweens.add({
      targets: bullet,
      x: this.enemy.x - 28,
      y: this.enemy.y + 8,
      scale: 1.18,
      alpha: { from: 0.45, to: 1 },
      duration: rating === "perfect" ? 280 : 360,
      ease: rating === "perfect" ? "Back.Out" : "Cubic.Out",
      onComplete: () => bullet.destroy(),
    });

    const damage = damageMap[rating];
    if (damage > 0) {
      this.hitStop(this.enemy, "enemy");
      this.spawnImpactBurst(this.enemy.x - 8, this.enemy.y + 12, rating === "perfect" ? 0x61ffb6 : 0x5fe5ff);
      this.playAttackFlash(rating);
      this.applyDamage(damage);
    }

    this.showImpact(rating, damage);
  }

  fireWordBonus(emoji) {
    const badge = this.add.text(this.player.x + 24, this.player.y - 36, emoji, {
      fontSize: "56px",
      stroke: "#000",
      strokeThickness: 6,
    }).setOrigin(0.5).setAlpha(0.2).setScale(0.5);

    this.tweens.add({
      targets: badge,
      x: this.enemy.x,
      y: this.enemy.y - 10,
      alpha: 1,
      scale: 1,
      duration: 420,
      ease: "Back.Out",
      onComplete: () => {
        this.spawnImpactBurst(this.enemy.x, this.enemy.y, 0xffef77);
        this.applyDamage(22);
        this.attackWarning.setText(`${emoji} ぼーなす!`);
        this.attackWarning.setAlpha(1);
        this.tweens.add({
          targets: this.attackWarning,
          alpha: 0,
          duration: 560,
          ease: "Quad.Out",
        });
        this.tweens.add({
          targets: badge,
          alpha: 0,
          y: badge.y - 24,
          duration: 260,
          onComplete: () => badge.destroy(),
        });
      },
    });
  }

  showImpact(rating, damage) {
    const labels = {
      perfect: `かんぺき -${damage}`,
      good: `いいね -${damage}`,
      miss: "ざんねん +0",
    };
    const colors = { perfect: "#61ffb6", good: "#5fe5ff", miss: "#a6b2cb" };

    this.impactText.setText(labels[rating]);
    this.impactText.setColor(colors[rating]);
    this.impactText.setScale(rating === "perfect" ? 1.22 : 1);
    this.impactText.setAlpha(1);

    this.tweens.add({
      targets: this.impactText,
      y: this.enemyName.y - 20,
      scale: this.impactText.scale + 0.1,
      duration: 110,
      yoyo: true,
      onComplete: () => {
        this.tweens.add({ targets: this.impactText, alpha: 0, duration: 300 });
      },
    });
  }

  applyDamage(damage) {
    this.enemyHp = Math.max(0, this.enemyHp - damage);
    const ratio = this.enemyHp / 100;
    this.hpBar.width = this.hpBg.width * ratio;

    const hpColor = ratio > 0.6 ? 0x44dd77 : ratio > 0.3 ? 0xf5cc48 : 0xff6278;
    this.hpBar.fillColor = hpColor;

    if (this.enemyHp === 0) {
      defeatedCount += 1;
      waveEl.textContent = String(defeatedCount);
      this.enemyMonsterIndex = (this.enemyMonsterIndex + 1) % this.enemyMonsters.length;
      this.enemy.setText(this.enemyMonsters[this.enemyMonsterIndex]);
      this.enemyName.setText(`てき: ${this.enemyLabels[this.enemyMonsterIndex]}`);
      this.enemyHp = 100;
      this.hpBar.width = this.hpBg.width;
      this.hpBar.fillColor = 0x44dd77;
      this.spawnImpactBurst(this.enemy.x, this.enemy.y, 0xffef77);
      this.moveEnemyToNewSpot();
      setFeedback("perfect", "てきをげきは！つぎのてき とうじょう！");
    }
  }

  enemyAttack() {
    if (isGameOver) return;
    const orb = this.add.circle(this.enemy.x - 18, this.enemy.y + 10, 12, 0xff6485, 1);
    this.tweens.add({
      targets: orb,
      x: this.player.x + 8,
      y: this.player.y - 6,
      scale: 2,
      duration: 480,
      ease: "Cubic.In",
      onComplete: () => {
        orb.destroy();
        this.hitStop(this.player, "player");
        this.spawnImpactBurst(this.player.x + 12, this.player.y - 6, 0xff4f6d);
        this.playPlayerHitFlash();
        this.attackWarning.setText("だめーじ！");
        this.attackWarning.setAlpha(1);
        this.tweens.add({
          targets: this.attackWarning,
          alpha: 0,
          duration: 420,
          ease: "Quad.Out",
        });
        setPlayerHp(playerHp - 18);
        sfx.playerHit();
        if (playerHp <= 0) {
          triggerGameOver();
        } else {
          setFeedback("miss", "てきのこうげき！にゅうりょくして きけんどをさげよう");
        }
      },
    });
  }

  resetBattle() {
    this.enemyHp = 100;
    this.hpBar.width = this.hpBg.width;
    this.hpBar.fillColor = 0x44dd77;
    this.enemyPointIndex = 0;
    this.moveEnemyToNewSpot(false);
    this.attackWarning.setAlpha(0);
    this.impactText.setAlpha(0);
  }

  update(_, delta) {
    this.starA.tilePositionY -= 0.05 * delta;
    this.starB.tilePositionY -= 0.1 * delta;
    this.grid.tilePositionY -= 0.04 * delta;

    if (isGameOver) return;

    setInputDanger(inputDanger + delta * 0.008);
    if (inputDanger >= 100) {
      setInputDanger(42);
      this.enemyAttack();
    }
  }

  handleResize() {
    const { width, height } = this.scale;
    this.starA.setSize(width, height);
    this.starB.setSize(width, height);
    this.grid.setSize(width, height);
    this.player.setPosition(width * 0.14, height * 0.8);
    this.hpBg.setSize(width * 0.24, 16);
    this.hpBar.setSize(this.hpBg.width * (this.enemyHp / 100), 12);

    const p = this.enemyPoints[this.enemyPointIndex];
    this.enemy.setPosition(width * p.x, height * p.y);
    this.syncEnemyHud();
    this.attackWarning.setPosition(width * 0.5, height * 0.48);
  }
}

syncAppHeightVar();
window.addEventListener("resize", syncGameSize);
if (window.visualViewport) {
  window.visualViewport.addEventListener("resize", syncGameSize);
  window.visualViewport.addEventListener("scroll", syncGameSize);
}

new Phaser.Game({
  type: Phaser.AUTO,
  parent: "gameRoot",
  width: Math.round(viewportWidth()),
  height: Math.round(viewportHeight()),
  backgroundColor: "#060c1a",
  scene: InvaderScene,
  scale: {
    mode: Phaser.Scale.RESIZE,
    autoCenter: Phaser.Scale.CENTER_BOTH,
  },
});

function handleScore(rating, missionComplete = false) {
  if (rating === "perfect") {
    perfectStreak += 1;
    score += 120 + combo * 4;
    combo += 1;
    setFeedback("perfect", missionComplete ? "かんぺき！ことば せいこう！" : "かんぺき！ちょくげき！");
    sfx.perfect();
  } else if (rating === "good") {
    perfectStreak = 0;
    score += 70 + combo * 2;
    combo += 1;
    setFeedback("good", missionComplete ? "いいね！ことば せいこう！" : "いいね！おしい！");
    sfx.good();
  } else {
    perfectStreak = 0;
    score += 10;
    combo = 0;
    setFeedback("miss", "ざんねん！てきが いきおいづく");
    sfx.miss();
  }

  scoreEl.textContent = String(score);
  comboEl.textContent = String(combo);
  perfectStreakEl.textContent = String(perfectStreak);

  if (!isCleared && score >= clearScoreThreshold) {
    isCleared = true;
    isGameOver = true;
    setFeedback("perfect", `くりあ！${clearScoreThreshold}てん とっぱ！`);
    showClearOverlay();
  }
}

document.addEventListener("keydown", (event) => {
  if (isGameOver) return;

  const input = normalizeInputKey(event);
  if (!input) return;

  sfx.unlock();
  lastInputAt = Date.now();

  const target = currentTarget();
  const rating = judge(input, target);
  updateLog(input);
  renderKeyboard(input);

  consumeDangerByRating(rating);

  if (rating === "miss") {
    handleScore(rating, false);
    if (sceneRef && sceneRef.fireLetter) sceneRef.fireLetter(input, rating);
    targetIndex = 0;
    updateTargetDisplay();
    setTimeout(() => renderKeyboard(), 120);
    return;
  }

  const wasLastStep = targetIndex === targetSequence.length - 1;
  if (sceneRef && sceneRef.fireLetter) sceneRef.fireLetter(input, rating);

  if (wasLastStep) {
    handleScore(rating, targetSequence.length >= 2);
    if (currentMission && sceneRef && sceneRef.fireWordBonus) {
      setFeedback(rating, `${currentMission.word} せいこう！`);
      sceneRef.fireWordBonus(currentMission.emoji);
    }
    setTimeout(() => setNextTarget(), 240);
  } else {
    handleScore(rating, false);
    targetIndex += 1;
    updateTargetDisplay();
    setFeedback(rating, "つぎのきーを つづけてにゅうりょく！");
    renderKeyboard(input);
  }

  setTimeout(() => renderKeyboard(), 120);
});

retryButtonEl.addEventListener("click", () => {
  hideRetryOverlay();
  resetGameState();
});

clearButtonEl.addEventListener("click", () => {
  hideClearOverlay();
  resetGameState();
});

setInputDanger(20);
setPlayerHp(100);
buildKeyboard();
setNextTarget(true);
hideRetryOverlay();
hideClearOverlay();

setInterval(() => {
  if (!isGameOver) renderKeyboard();
}, 180);
