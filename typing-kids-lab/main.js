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
const waveEl = document.getElementById("wave");
const keyboardEl = document.getElementById("keyboard");

let target = "A";
let score = 0;
let combo = 0;
let wave = 1;
let typedLog = [];

const keyButtons = new Map();
const positionMap = new Map();
rows.forEach((row, r) => row.forEach((key, c) => positionMap.set(key, { r, c })));

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

function normalizeInputKey(event) {
  if (event.isComposing) return "";
  if (/^Key[A-Z]$/.test(event.code)) return event.code.replace("Key", "");
  const key = (event.key || "").toUpperCase();
  return /^[A-Z]$/.test(key) ? key : "";
}

function judge(input) {
  const near = neighborsOf(target);
  if (input === target) return "perfect";
  if (near.includes(input)) return "good";
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
}

let sceneRef = null;

class InvaderScene extends Phaser.Scene {
  constructor() {
    super("invader");
  }

  create() {
    const { width, height } = this.scale;

    this.cameras.main.setBackgroundColor("#060c1a");

    this.starA = this.add.tileSprite(0, 0, width, height, this.makeStarTexture(0x4d6aa8, 2)).setOrigin(0);
    this.starB = this.add.tileSprite(0, 0, width, height, this.makeStarTexture(0x84a1dd, 2)).setOrigin(0);
    this.grid = this.add.tileSprite(0, 0, width, height, this.makeGridTexture()).setOrigin(0).setAlpha(0.26);

    this.enemy = this.add.text(width * 0.78, height * 0.28, "👾", {
      fontSize: `${Math.max(84, Math.floor(width * 0.08))}px`,
    }).setOrigin(0.5);

    this.enemyName = this.add.text(width * 0.78, height * 0.18, "INVADER CORE", {
      fontFamily: "monospace",
      fontSize: "20px",
      color: "#a9c1ff",
    }).setOrigin(0.5);

    this.hpBg = this.add.rectangle(width * 0.78, height * 0.22, width * 0.24, 16, 0x1a2547).setStrokeStyle(1, 0x5e75b6);
    this.hpBar = this.add.rectangle(this.hpBg.x - this.hpBg.width / 2, this.hpBg.y, this.hpBg.width, 12, 0x44dd77).setOrigin(0, 0.5);

    this.impactText = this.add.text(width * 0.78, height * 0.11, "", {
      fontFamily: "monospace",
      fontSize: "44px",
      color: "#ffffff",
      stroke: "#000000",
      strokeThickness: 6,
    }).setOrigin(0.5).setAlpha(0);

    this.resizeHandler = () => this.handleResize();
    this.scale.on("resize", this.resizeHandler);
    this.enemyHp = 100;
    sceneRef = this;
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

  fireLetter(letter, rating) {
    const { width, height } = this.scale;
    const colors = { perfect: "#ff5fa2", good: "#5fe5ff", miss: "#a6b2cb" };
    const damageMap = { perfect: 28, good: 16, miss: 0 };

    const bullet = this.add.text(width * 0.14, height * 0.82, letter, {
      fontFamily: "monospace",
      fontSize: "54px",
      color: colors[rating],
      stroke: "#000",
      strokeThickness: 6,
    }).setOrigin(0.5);

    this.tweens.add({
      targets: bullet,
      x: this.enemy.x - 28,
      y: this.enemy.y + 8,
      scale: { from: 0.4, to: 1.1 },
      alpha: { from: 0.3, to: 1 },
      duration: 360,
      ease: "Cubic.Out",
      onComplete: () => bullet.destroy(),
    });

    const damage = damageMap[rating];
    if (damage > 0) {
      this.tweens.add({ targets: this.enemy, scale: 1.18, angle: -6, duration: 60, yoyo: true });
      this.applyDamage(damage);
    }

    this.showImpact(rating, damage);
  }

  showImpact(rating, damage) {
    const labels = {
      perfect: `PERFECT -${damage}`,
      good: `GOOD -${damage}`,
      miss: "MISS +0",
    };
    const colors = { perfect: "#ff5fa2", good: "#5fe5ff", miss: "#a6b2cb" };

    this.impactText.setText(labels[rating]);
    this.impactText.setColor(colors[rating]);
    this.impactText.setScale(0.8);
    this.impactText.setAlpha(1);

    this.tweens.add({
      targets: this.impactText,
      scale: 1.06,
      y: this.enemyName.y - 12,
      duration: 100,
      yoyo: true,
      onComplete: () => {
        this.tweens.add({ targets: this.impactText, alpha: 0, duration: 260 });
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
      wave += 1;
      waveEl.textContent = String(wave);
      this.enemyHp = 100;
      this.hpBar.width = this.hpBg.width;
      this.hpBar.fillColor = 0x44dd77;
      setFeedback("perfect", "WAVE CLEAR! 次の敵が出現！");
    }
  }

  update(_, delta) {
    this.starA.tilePositionY -= 0.05 * delta;
    this.starB.tilePositionY -= 0.09 * delta;
    this.grid.tilePositionY -= 0.03 * delta;
  }

  handleResize() {
    const { width, height } = this.scale;
    this.starA.setSize(width, height);
    this.starB.setSize(width, height);
    this.grid.setSize(width, height);
    this.enemy.setPosition(width * 0.78, height * 0.28);
    this.enemyName.setPosition(width * 0.78, height * 0.18);
    this.hpBg.setPosition(width * 0.78, height * 0.22).setSize(width * 0.24, 16);
    this.hpBar.setPosition(this.hpBg.x - this.hpBg.width / 2, this.hpBg.y).setSize(this.hpBg.width * (this.enemyHp / 100), 12);
    this.impactText.setPosition(width * 0.78, height * 0.11);
  }
}

const game = new Phaser.Game({
  type: Phaser.AUTO,
  parent: "gameRoot",
  width: window.innerWidth,
  height: window.innerHeight,
  backgroundColor: "#060c1a",
  scene: InvaderScene,
  scale: {
    mode: Phaser.Scale.RESIZE,
    autoCenter: Phaser.Scale.CENTER_BOTH,
  },
});


function handleScore(rating) {
  if (rating === "perfect") {
    score += 120 + combo * 4;
    combo += 1;
    setFeedback("perfect", "Perfect! 直撃！");
  } else if (rating === "good") {
    score += 70 + combo * 2;
    combo += 1;
    setFeedback("good", "Good! かすった！");
  } else {
    score += 10;
    combo = 0;
    setFeedback("miss", "Miss! でも経験値+1");
  }

  scoreEl.textContent = String(score);
  comboEl.textContent = String(combo);
}

document.addEventListener("keydown", (event) => {
  const input = normalizeInputKey(event);
  if (!input) return;

  const rating = judge(input);
  updateLog(input);
  renderKeyboard(input);

  handleScore(rating);
  if (sceneRef && sceneRef.fireLetter) sceneRef.fireLetter(input, rating);

  if (rating !== "miss") setTimeout(setNextTarget, 220);
  setTimeout(() => renderKeyboard(), 120);
});

buildKeyboard();
setNextTarget();
