const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, Number(value)));

export function normalizeHex(value) {
  const match = String(value || "").trim().match(/^#?([0-9a-f]{6})$/i);
  return match ? `#${match[1].toLowerCase()}` : null;
}

export function pickerOwnsTarget(trigger, panel, target) {
  return trigger?.contains?.(target) === true || panel?.contains?.(target) === true;
}

export function rgbToHex(red, green, blue) {
  return `#${[red, green, blue].map((value) => Math.round(clamp(value, 0, 255)).toString(16).padStart(2, "0")).join("")}`;
}

export function hexToRgb(value) {
  const hex = normalizeHex(value) || "#000000";
  return {
    r: Number.parseInt(hex.slice(1, 3), 16),
    g: Number.parseInt(hex.slice(3, 5), 16),
    b: Number.parseInt(hex.slice(5, 7), 16),
  };
}

export function rgbToHsl(red, green, blue) {
  const r = clamp(red, 0, 255) / 255, g = clamp(green, 0, 255) / 255, b = clamp(blue, 0, 255) / 255;
  const maximum = Math.max(r, g, b), minimum = Math.min(r, g, b), delta = maximum - minimum;
  const l = (maximum + minimum) / 2;
  let h = 0;
  if (delta) {
    if (maximum === r) h = 60 * (((g - b) / delta) % 6);
    else if (maximum === g) h = 60 * ((b - r) / delta + 2);
    else h = 60 * ((r - g) / delta + 4);
  }
  const s = delta ? delta / (1 - Math.abs(2 * l - 1)) : 0;
  return { h: (h + 360) % 360, s, l };
}

export function hslToRgb(hue, saturation, lightness) {
  const h = ((Number(hue) % 360) + 360) % 360;
  const s = clamp(saturation, 0, 1), l = clamp(lightness, 0, 1);
  const chroma = (1 - Math.abs(2 * l - 1)) * s;
  const x = chroma * (1 - Math.abs((h / 60) % 2 - 1)), match = l - chroma / 2;
  const [r, g, b] = h < 60 ? [chroma, x, 0] : h < 120 ? [x, chroma, 0]
    : h < 180 ? [0, chroma, x] : h < 240 ? [0, x, chroma]
    : h < 300 ? [x, 0, chroma] : [chroma, 0, x];
  return { r: Math.round((r + match) * 255), g: Math.round((g + match) * 255), b: Math.round((b + match) * 255) };
}

function assignStyle(element, values) {
  Object.assign(element.style, values);
  return element;
}

function make(tag, style = {}) { return assignStyle(document.createElement(tag), style); }

export class PaintColorPicker {
  constructor({ color = "#ff0000", onChange, onEyedropper, onClear, panelHost, onOpenChange }) {
    this.color = normalizeHex(color) || "#ff0000";
    this.previous = this.color;
    this.onChange = onChange;
    this.onEyedropper = onEyedropper;
    this.onClear = onClear;
    this.onOpenChange = onOpenChange;
    this.recent = this.loadRecent();
    this.root = make("div", { position: "relative", display: "inline-flex" });
    this.trigger = make("button", {
      width: "34px", height: "28px", padding: "3px", border: "1px solid rgba(255,255,255,.3)",
      borderRadius: "4px", background: "rgba(0,0,0,.25)", cursor: "pointer",
    });
    this.trigger.type = "button";
    this.swatch = make("span", { display: "block", width: "100%", height: "100%", borderRadius: "2px" });
    this.trigger.append(this.swatch);
    this.panel = this.createPanel();
    this.root.append(this.trigger);
    panelHost.append(this.panel);
    this.resizeObserver = new ResizeObserver(() => this.resizeInteractionPlane());
    this.resizeObserver.observe(this.sl);
    this.trigger.addEventListener("click", (event) => {
      event.stopPropagation();
      const opening = this.panel.style.display === "none";
      if (opening) this.previous = this.color;
      this.setOpen(opening);
      this.sync();
    });
    this.documentPointerDown = (event) => {
      if (!pickerOwnsTarget(this.trigger, this.panel, event.target)) this.setOpen(false);
    };
    this.documentKeyDown = (event) => {
      if (event.key !== "Escape" || this.panel.style.display === "none") return;
      event.preventDefault();
      this.setColor(this.previous, false);
      this.setOpen(false);
      this.trigger.focus();
    };
    document.addEventListener("pointerdown", this.documentPointerDown, true);
    document.addEventListener("keydown", this.documentKeyDown, true);
    this.sync();
  }

  loadRecent() {
    try {
      const values = JSON.parse(localStorage.getItem("uc_staged_paint_recent_colors") || "[]");
      return Array.isArray(values) ? values.map(normalizeHex).filter(Boolean).slice(0, 8) : [];
    } catch { return []; }
  }

  saveRecent() {
    this.recent = [this.color, ...this.recent.filter((value) => value !== this.color)].slice(0, 8);
    try { localStorage.setItem("uc_staged_paint_recent_colors", JSON.stringify(this.recent)); } catch {}
    this.syncRecent();
  }

  createPanel() {
    const panel = make("div", {
      flex: "0 0 220px", width: "220px", boxSizing: "border-box", alignSelf: "stretch", minHeight: "0",
      display: "flex", flexDirection: "column", gap: "7px", padding: "9px", border: "1px solid rgba(255,255,255,.3)",
      borderRadius: "7px", color: "#ddd", background: "#20242a", boxShadow: "0 5px 18px rgba(0,0,0,.55)",
    });
    panel.style.display = "none";
    panel.addEventListener("pointerdown", (event) => event.stopPropagation());
    const title = make("div", { fontWeight: "600", opacity: ".9" });
    title.textContent = "HSL color";
    this.sl = make("div", {
      position: "relative", width: "100%", flex: "1 1 150px", minHeight: "80px", borderRadius: "4px", cursor: "crosshair", touchAction: "none",
    });
    this.slCanvas = make("canvas", { position: "absolute", inset: "0", display: "block", width: "100%", height: "100%", borderRadius: "4px" });
    this.slCanvas.width = 202;
    this.slCanvas.height = 150;
    this.slMarker = make("span", {
      position: "absolute", width: "10px", height: "10px", border: "2px solid white", borderRadius: "50%",
      transform: "translate(-50%,-50%)", boxShadow: "0 0 2px #000", pointerEvents: "none",
    });
    this.sl.append(this.slCanvas, this.slMarker);
    this.hue = make("div", {
      position: "relative", height: "16px", borderRadius: "8px", cursor: "pointer", touchAction: "none",
      background: "linear-gradient(to right,#f00,#ff0,#0f0,#0ff,#00f,#f0f,#f00)",
    });
    this.hueMarker = make("span", {
      position: "absolute", top: "-2px", width: "8px", height: "20px", border: "2px solid white", borderRadius: "4px",
      transform: "translateX(-50%)", boxSizing: "border-box", boxShadow: "0 0 2px #000", pointerEvents: "none",
    });
    this.hue.append(this.hueMarker);
    const updatePointer = (target, event) => {
      const rect = target.getBoundingClientRect();
      return { x: clamp((event.clientX - rect.left) / rect.width, 0, 1), y: clamp((event.clientY - rect.top) / rect.height, 0, 1) };
    };
    const track = (target, update) => {
      target.addEventListener("pointerdown", (event) => { target.setPointerCapture(event.pointerId); update(updatePointer(target, event)); });
      target.addEventListener("pointermove", (event) => { if (target.hasPointerCapture(event.pointerId)) update(updatePointer(target, event)); });
    };
    track(this.sl, ({ x, y }) => { this.hsl.s = x; this.hsl.l = 1 - y; this.commitHsl(); });
    track(this.hue, ({ x }) => { this.hsl.h = x * 360; this.commitHsl(); });

    this.hexInput = make("input", { width: "78px" });
    this.hexInput.type = "text"; this.hexInput.maxLength = 7;
    this.rgbInputs = ["R", "G", "B"].map((label) => {
      const input = make("input", { width: "38px" }); input.type = "number"; input.min = "0"; input.max = "255"; input.title = label; return input;
    });
    const fields = make("div", { display: "flex", alignItems: "center", gap: "4px" });
    fields.append(this.hexInput, ...this.rgbInputs);
    const commitFields = (event) => {
      if (event.type === "keydown" && event.key === "Escape") { event.preventDefault(); this.setColor(this.previous, false); return; }
      if (event.type === "keydown" && event.key !== "Enter") return;
      const rgb = this.rgbInputs.map((input) => Number(input.value));
      const next = event.target === this.hexInput
        ? normalizeHex(this.hexInput.value)
        : rgb.every((value) => Number.isFinite(value) && value >= 0 && value <= 255)
          ? rgbToHex(...rgb)
          : null;
      if (next) this.setColor(next, true); else this.sync();
    };
    for (const input of [this.hexInput, ...this.rgbInputs]) {
      input.addEventListener("keydown", commitFields); input.addEventListener("blur", commitFields);
    }
    const comparisons = make("div", { display: "flex", gap: "5px" });
    this.previousSwatch = make("button", { flex: "1", height: "24px", border: "1px solid #777", borderRadius: "3px" });
    this.currentSwatch = make("span", { flex: "1", height: "24px", border: "1px solid #777", borderRadius: "3px" });
    this.previousSwatch.type = "button"; this.previousSwatch.title = "Restore previous color";
    this.previousSwatch.addEventListener("click", () => this.setColor(this.previous, true));
    comparisons.append(this.previousSwatch, this.currentSwatch);
    this.recentRoot = make("div", { display: "flex", minHeight: "20px", gap: "4px", flexWrap: "wrap" });
    const eyedropper = make("button", { height: "28px" });
    eyedropper.type = "button"; eyedropper.textContent = "Pick from composition";
    eyedropper.addEventListener("click", () => { this.setOpen(false); this.onEyedropper?.(); });
    panel.append(title, this.sl, this.hue, comparisons, fields, this.recentRoot, eyedropper);
    if (this.onClear) {
      const clearBtn = make("button", { height: "28px" });
      clearBtn.type = "button";
      clearBtn.textContent = "Clear color (outline only)";
      clearBtn.addEventListener("click", () => { this.setOpen(false); this.onClear?.(); });
      panel.append(clearBtn);
    }
    return panel;
  }

  commitHsl() {
    const rgb = hslToRgb(this.hsl.h, this.hsl.s, this.hsl.l);
    this.setColor(rgbToHex(rgb.r, rgb.g, rgb.b), true, false);
  }

  setColor(value, notify = true, remember = true) {
    const normalized = normalizeHex(value);
    if (!normalized) return false;
    this.color = normalized;
    const rgb = hexToRgb(normalized);
    this.hsl = rgbToHsl(rgb.r, rgb.g, rgb.b);
    this.sync();
    if (remember) this.saveRecent();
    if (notify) this.onChange?.(normalized);
    return true;
  }

  setOpen(open) {
    this.panel.style.display = open ? "flex" : "none";
    this.trigger.setAttribute("aria-expanded", String(open));
    this.onOpenChange?.(open);
  }

  isOpen() { return this.panel.style.display !== "none"; }

  setPanelHeight(height) {
    this.panel.style.height = `${Math.max(1, Math.round(height))}px`;
    this.resizeInteractionPlane();
  }

  resizeInteractionPlane() {
    const width = Math.max(1, Math.round(this.sl.clientWidth || 202));
    const height = Math.max(1, Math.round(this.sl.clientHeight || 150));
    if (this.slCanvas.width === width && this.slCanvas.height === height) return;
    this.slCanvas.width = width;
    this.slCanvas.height = height;
    this.renderedHue = null;
    this.sync();
  }

  renderSlPlane() {
    const hue = Math.round(this.hsl.h * 1000) / 1000;
    if (this.renderedHue === hue) return;
    this.renderedHue = hue;
    const context = this.slCanvas.getContext("2d");
    const image = context.createImageData(this.slCanvas.width, this.slCanvas.height);
    for (let y = 0; y < image.height; y++) for (let x = 0; x < image.width; x++) {
      const rgb = hslToRgb(hue, x / (image.width - 1), 1 - y / (image.height - 1));
      const offset = (y * image.width + x) * 4;
      image.data[offset] = rgb.r; image.data[offset + 1] = rgb.g; image.data[offset + 2] = rgb.b; image.data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }

  syncRecent() {
    if (!this.recentRoot) return;
    this.recentRoot.replaceChildren(...this.recent.map((color) => {
      const button = make("button", { width: "20px", height: "20px", padding: "0", border: "1px solid #888", borderRadius: "3px", background: color });
      button.type = "button"; button.title = color; button.addEventListener("click", () => this.setColor(color, true)); return button;
    }));
  }

  sync() {
    const rgb = hexToRgb(this.color);
    this.hsl ||= rgbToHsl(rgb.r, rgb.g, rgb.b);
    this.swatch.style.background = this.color;
    this.currentSwatch.style.background = this.color;
    this.previousSwatch.style.background = this.previous;
    this.renderSlPlane();
    this.slMarker.style.left = `${this.hsl.s * 100}%`; this.slMarker.style.top = `${(1 - this.hsl.l) * 100}%`;
    this.hueMarker.style.left = `${this.hsl.h / 360 * 100}%`;
    if (document.activeElement !== this.hexInput) this.hexInput.value = this.color;
    [rgb.r, rgb.g, rgb.b].forEach((value, index) => { if (document.activeElement !== this.rgbInputs[index]) this.rgbInputs[index].value = String(value); });
    this.syncRecent();
  }

  dispose() {
    document.removeEventListener("pointerdown", this.documentPointerDown, true);
    document.removeEventListener("keydown", this.documentKeyDown, true);
    this.resizeObserver?.disconnect();
    this.panel.remove();
  }
}
