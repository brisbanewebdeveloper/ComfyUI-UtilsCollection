import { app } from "../../scripts/app.js";
import {
  TABS, TAB_LABELS, TASK_TYPES, VISIBLE_MARKERS, AUDIO_MARKERS, CAMERA_MOTIONS,
  H3_FPS, snapH3Frames, h3SecondsFromFrames, h3StepDuration, formatSeconds, parseSeconds,
  createState, restoreState, addDefinition, removeDefinition, defaultRetentionText,
  createSegment, addSegment, moveItem, extractTags, compilePrompt, timelineWarnings,
} from "./minimax_h3_prompt_model.js";

class H3CanvasPromptEditor {
  constructor(node, name, data) {
    this.node = node;
    this.raw = data[1].default ?? JSON.stringify(createState());
    this.state = null;
    this.error = null;
    this.scroll = 0;
    this.collapsed = new Set();
    this.contentHeight = 0;
    this.viewportY = 0;
    this.viewportHeight = 380;
    this.hitRegions = [];
    this.hoveredRegion = null;
    this.textEditor = null;
    this.abort = new AbortController();
    this.restore(this.raw);

    this.widget = node.addCustomWidget({
      name,
      type: "custom",
      value: this.raw,
      options: { socketless: true },
      computeSize: () => [node.size[0], Math.max(380, node.size[1] - 40)],
      draw: (ctx, _node, width, y) => this.draw(ctx, width, y),
      mouse: (event, position) => this.mouse(event, position),
      serializeValue: () => this.serialize(),
    });
    this.widget.options.socketless = true;

    const getWidgetOnPos = node.getWidgetOnPos;
    node.getWidgetOnPos = (graphX, graphY, includeDisabled) => {
      const x = graphX - node.pos[0];
      const y = graphY - node.pos[1];
      if (this.hitRegions.some(region => this.contains(region, x, y))) return this.widget;
      const found = getWidgetOnPos.call(node, graphX, graphY, includeDisabled);
      return found === this.widget ? undefined : found;
    };
    app.canvas?.canvas?.addEventListener("pointermove", event => this.onPointerMove(event), {
      passive: true, signal: this.abort.signal,
    });
    app.canvas?.canvas?.addEventListener("pointerleave", () => this.clearHoveredTooltip(), {
      passive: true, signal: this.abort.signal,
    });
    app.canvas?.canvas?.addEventListener("wheel", event => this.onWheel(event), {
      capture: true, passive: false, signal: this.abort.signal,
    });
    const onRemoved = node.onRemoved;
    node.onRemoved = (...args) => {
      this.abort.abort();
      this.closeTextEditor();
      node.getWidgetOnPos = getWidgetOnPos;
      onRemoved?.apply(node, args);
    };
    requestAnimationFrame(() => {
      if (!this.abort.signal.aborted) node.setSize([Math.max(480, node.size[0]), Math.max(500, node.size[1])]);
    });
  }

  restore(raw) {
    this.raw = raw;
    try {
      this.state = restoreState(raw);
      this.error = null;
    } catch (error) {
      this.state = null;
      this.error = error.message;
    }
  }

  sync() {
    const value = this.widget?.value;
    if (typeof value === "string" && value !== this.raw) this.restore(value);
  }

  serialize() {
    this.sync();
    if (this.error) throw new Error(this.error);
    compilePrompt(this.state);
    return JSON.stringify(this.state);
  }

  change(update) {
    this.node.graph?.beforeChange(this.node);
    try {
      update();
      this.raw = JSON.stringify(this.state);
      this.widget.value = this.raw;
      app.canvas?.setDirty(true, true);
    } finally {
      this.node.graph?.afterChange(this.node);
    }
  }

  contains(region, x, y) {
    return x >= region.x && x < region.x + region.w && y >= region.y && y < region.y + region.h;
  }

  hit(x, y, w, h, action, tooltip = null) {
    const top = Math.max(y, this.viewportY);
    const bottom = Math.min(y + h, this.viewportY + this.viewportHeight);
    if (bottom > top) this.hitRegions.push({ x, y: top, w, h: bottom - top, action, tooltip });
  }

  box(ctx, x, y, w, h, fill, stroke = "#4b505a", radius = 4) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, radius);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.stroke();
    }
  }

  text(ctx, value, x, y, color = "#ddd", font = "12px sans-serif", maxWidth) {
    ctx.font = font;
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const line = String(value ?? "").replace(/\r?\n/g, " ");
    if (!maxWidth || ctx.measureText(line).width <= maxWidth) {
      ctx.fillText(line, x, y);
      return;
    }
    let end = line.length;
    while (end > 0 && ctx.measureText(line.slice(0, end) + "…").width > maxWidth) end--;
    ctx.fillText(line.slice(0, end) + "…", x, y);
  }

  button(ctx, x, y, w, h, label, action, accent = false, align = "center", active = false, danger = false, tooltip = null) {
    let fill = accent ? "#26394e" : "#202329";
    let stroke = accent ? "#709ecc" : "#59606a";
    let textColor = accent ? "#d7ebff" : "#ddd";

    if (active) {
      fill = "#1d4ed8";
      stroke = "#93c5fd";
      textColor = "#ffffff";
    } else if (danger) {
      fill = "#7f1d1d";
      stroke = "#f87171";
      textColor = "#fecaca";
    }

    this.box(ctx, x, y, w, h, fill, stroke, 4);
    ctx.font = "11px sans-serif";
    const tw = ctx.measureText(label).width;
    const tx = align === "left" ? x + 8 : x + Math.max(4, (w - tw) / 2);
    this.text(ctx, label, tx, y + h / 2, textColor, "11px sans-serif", w - 8);
    this.hit(x, y, w, h, action, tooltip);
  }

  chip(ctx, x, y, label, action, active = false, accent = false, tooltip = null) {
    ctx.font = "11px sans-serif";
    const w = Math.ceil(ctx.measureText(label).width) + 16;
    const h = 24;
    this.button(ctx, x, y, w, h, label, action, accent, "center", active, false, tooltip);
    return w;
  }

  onPointerMove(event) {
    if (this.textEditor) {
      this.clearHoveredTooltip();
      return;
    }
    const canvas = app.canvas;
    if (!canvas?.graph) return;
    const bounds = canvas.canvas.getBoundingClientRect();
    const scale = canvas.ds.scale;
    const graphX = (event.clientX - bounds.left) / scale - canvas.ds.offset[0];
    const graphY = (event.clientY - bounds.top) / scale - canvas.ds.offset[1];
    const x = graphX - this.node.pos[0];
    const y = graphY - this.node.pos[1];

    if (x < 0 || x > (this.node.size[0] || 480) || y < 0 || y > (this.node.size[1] || 500)) {
      this.clearHoveredTooltip();
      return;
    }

    const matched = this.hitRegions.find(r => r.tooltip && this.contains(r, x, y));
    if (matched !== this.hoveredRegion) {
      this.hoveredRegion = matched || null;
      canvas.setDirty(true, false);
    }
  }

  clearHoveredTooltip() {
    if (this.hoveredRegion) {
      this.hoveredRegion = null;
      app.canvas?.setDirty(true, false);
    }
  }

  drawCanvasTooltip(ctx, region, nodeWidth, yTop, nodeHeight) {
    if (!region?.tooltip) return;
    const text = region.tooltip;
    const maxWidth = Math.min(280, nodeWidth - 28);
    const font = "11px sans-serif";
    const lineHeight = 15;

    ctx.save();
    ctx.font = font;

    const words = String(text).split(/\s+/);
    const lines = [];
    let currentLine = "";
    for (let i = 0; i < words.length; i++) {
      const testLine = currentLine ? `${currentLine} ${words[i]}` : words[i];
      if (ctx.measureText(testLine).width > maxWidth && currentLine) {
        lines.push(currentLine);
        currentLine = words[i];
      } else {
        currentLine = testLine;
      }
    }
    if (currentLine) lines.push(currentLine);

    let maxMeasured = 0;
    for (const l of lines) {
      maxMeasured = Math.max(maxMeasured, ctx.measureText(l).width);
    }

    const paddingX = 8;
    const paddingY = 6;
    const boxW = Math.max(60, maxMeasured + paddingX * 2);
    const boxH = lines.length * lineHeight + paddingY * 2;

    let bx = region.x + region.w / 2 - boxW / 2;
    bx = Math.max(10, Math.min(nodeWidth - boxW - 10, bx));

    let by = region.y - boxH - 8;
    if (by < yTop + 4) {
      by = region.y + region.h + 8;
    }

    ctx.shadowColor = "rgba(0, 0, 0, 0.7)";
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 3;

    this.box(ctx, bx, by, boxW, boxH, "#0f1218", "#475569", 4);

    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;

    ctx.fillStyle = "#f1f5f9";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], bx + paddingX, by + paddingY + i * lineHeight);
    }

    ctx.restore();
  }

  field(ctx, label, value, x, y, w, action) {
    this.text(ctx, label, x, y + 6, "#aeb5bf", "11px sans-serif");
    const rect = { x, y: y + 16, w, h: 26 };
    this.button(ctx, rect.x, rect.y, rect.w, rect.h, String(value || "Click to edit..."),
      (event, position) => action(event, position, rect), false, "left");
  }

  editSingleLine(title, value, apply, event) {
    app.canvas.prompt(title, value, text => {
      if (text !== null) this.change(() => apply(text));
    }, event);
  }

  openTextEditor(value, apply, rect, tagSpawner = null) {
    this.closeTextEditor();
    this.clearHoveredTooltip();
    const container = document.createElement("div");
    container.className = "comfy-multiline-container";
    container.dataset.testid = "h3-prompt-editor-container";
    Object.assign(container.style, {
      position: "fixed", zIndex: "1000", boxSizing: "border-box", margin: "0",
      display: "flex", flexDirection: "column", background: "#1f2228",
      border: "1px solid #709ecc", borderRadius: "4px", padding: "4px", gap: "4px",
    });

    const element = document.createElement("textarea");
    element.className = "comfy-multiline-input";
    element.dataset.testid = "h3-prompt-textarea";
    element.value = value;
    element.spellcheck = true;
    Object.assign(element.style, {
      width: "100%", height: "100%", minHeight: "80px", boxSizing: "border-box",
      background: "#121418", color: "#eee", border: "1px solid #3d434d", borderRadius: "3px",
      outline: "none", resize: "none", fontFamily: "Inter, Arial, sans-serif", fontSize: "12px",
      lineHeight: "1.35", padding: "4px",
    });

    if (tagSpawner && tagSpawner.length) {
      const tagBar = document.createElement("div");
      Object.assign(tagBar.style, {
        display: "flex", flexWrap: "wrap", gap: "4px", maxHeight: "50px", overflowY: "auto",
      });
      tagSpawner.forEach(tag => {
        const btn = document.createElement("button");
        btn.textContent = tag;
        Object.assign(btn.style, {
          background: "#26394e", color: "#d7ebff", border: "1px solid #709ecc",
          borderRadius: "3px", fontSize: "11px", padding: "2px 6px", cursor: "pointer",
        });
        btn.onmousedown = e => e.preventDefault();
        btn.onclick = e => {
          e.preventDefault();
          const start = element.selectionStart ?? element.value.length;
          const end = element.selectionEnd ?? element.value.length;
          const prev = element.value;
          element.value = prev.slice(0, start) + tag + prev.slice(end);
          element.selectionStart = element.selectionEnd = start + tag.length;
          element.focus();
          this.change(() => apply(element.value));
        };
        tagBar.appendChild(btn);
      });
      container.appendChild(tagBar);
    }

    container.appendChild(element);
    this.textEditor = { element, container, rect };

    this.outsidePointer = (event) => {
      if (this.textEditor?.container && !this.textEditor.container.contains(event.target)) {
        this.closeTextEditor();
      }
    };
    requestAnimationFrame(() => {
      document.addEventListener("pointerdown", this.outsidePointer, true);
    });

    element.addEventListener("input", () => this.change(() => apply(element.value)));
    element.addEventListener("keydown", event => {
      event.stopPropagation();
      if (event.key === "Escape" || (event.key === "Enter" && (event.ctrlKey || event.metaKey))) {
        event.preventDefault();
        this.closeTextEditor();
      }
    });

    element.addEventListener("blur", (e) => {
      if (this.textEditor?.container && e.relatedTarget && this.textEditor.container.contains(e.relatedTarget)) {
        return;
      }
      this.closeTextEditor();
    });

    document.body.appendChild(container);
    this.positionTextEditor();
    requestAnimationFrame(() => element.focus());
  }

  closeTextEditor() {
    if (this.outsidePointer) {
      document.removeEventListener("pointerdown", this.outsidePointer, true);
      this.outsidePointer = null;
    }
    if (this.textEditor) {
      this.textEditor.container?.remove();
      this.textEditor.element?.remove();
      this.textEditor = null;
    }
  }

  positionTextEditor() {
    if (!this.textEditor || !app.canvas?.canvas) return;
    const { container, rect } = this.textEditor;
    const canvas = app.canvas.canvas;
    const canvasRect = canvas.getBoundingClientRect();
    const scale = app.canvas.ds.scale;
    const x = canvasRect.left + (this.node.pos[0] + rect.x + app.canvas.ds.offset[0]) * scale;
    const y = canvasRect.top + (this.node.pos[1] + rect.y + app.canvas.ds.offset[1]) * scale;
    const w = Math.max(340, rect.w * scale);
    const h = Math.max(140, rect.h * scale + 60);

    Object.assign(container.style, {
      left: `${Math.max(10, Math.min(window.innerWidth - w - 10, x))}px`,
      top: `${Math.max(10, Math.min(window.innerHeight - h - 10, y))}px`,
      width: `${w}px`,
      height: `${h}px`,
    });
  }

  draw(ctx, width, y) {
    if (this.textEditor) this.positionTextEditor();
    this.sync();
    this.hitRegions = [];
    this.viewportY = y;
    this.viewportHeight = Math.max(360, (this.node.size[1] || 480) - y);
    const left = 10;
    const right = width - 10;
    const available = right - left;

    this.box(ctx, 4, y, width - 8, this.viewportHeight, "#181a1f", "#333842", 5);

    if (this.error) {
      this.text(ctx, this.error, left, y + 20, "#ff8080", "bold 12px sans-serif", available);
      return;
    }

    // Top Header: Title + Precision
    this.text(ctx, "MiniMax H3 Prompt Builder", left, y + 14, "#f3f5f7", "bold 12px sans-serif");
    const prec = this.state.precision || 2;
    this.button(ctx, right - 80, y + 4, 80, 20, `${prec} Decimals`, () => {
      this.change(() => {
        this.state.precision = this.state.precision === 2 ? 3 : 2;
      });
    }, true, "center", false, false, "Switches timestamps between two decimal places (00.00s) and three decimal places (00.000s) across the entire prompt.");

    // Tab Navigation Bar
    const tabY = y + 28;
    const tabWidth = Math.floor(available / TABS.length);
    const tabTooltips = {
      subjects: "Define people, creatures, reference pictures, source videos, or audio tracks used in your prompt.",
      summary: "Choose your video task type and write a high-level summary of the overall scene.",
      retention: "Specify how much identity, motion, or audio from your references should carry over into the new video.",
      detailed: "Break down the video into timed intervals detailing what is seen, spoken, and heard.",
      soundscape: "Describe the background ambience, environment noise, and physical action sounds heard across the entire video.",
      music: "Describe the audience-only background music, including instruments, tempo, and rhythm.",
    };

    TABS.forEach((tab, index) => {
      const active = this.state.activeTab === tab;
      const tx = left + index * tabWidth;
      const tw = index === TABS.length - 1 ? (right - tx) : (tabWidth - 2);

      let countStr = "";
      if (tab === "subjects" && this.state.definitions.length) countStr = ` (${this.state.definitions.length})`;
      else if (tab === "retention" && this.state.retention.length) countStr = ` (${this.state.retention.length})`;
      else if (tab === "detailed") countStr = this.state.detailed.hasTimeline ? ` (${this.state.segments.length})` : ` (1)`;

      const label = `${TAB_LABELS[tab].split(" ")[1]}${countStr}`;
      this.button(ctx, tx, tabY, tw, 26, label, () => {
        this.closeTextEditor();
        this.clearHoveredTooltip();
        this.change(() => {
          this.state.activeTab = tab;
          this.scroll = 0;
        });
      }, false, "center", active, false, tabTooltips[tab]);
    });

    const contentTopY = tabY + 32;
    this.viewportY = contentTopY;
    this.viewportHeight = Math.max(200, (y + this.viewportHeight) - contentTopY - 70);

    ctx.save();
    ctx.beginPath();
    ctx.rect(left, contentTopY, available, this.viewportHeight);
    ctx.clip();

    let cy = 0;
    const sy = yCoord => contentTopY + yCoord - this.scroll;
    const tags = extractTags(this.state);
    const availableTags = [...tags.subjects, ...tags.pictures, ...tags.videos, ...tags.audios];

    // --- TAB 1: DEFINITIONS (subject_definitions:) ---
    if (this.state.activeTab === "subjects") {
      this.text(ctx, "subject_definitions: (Define subjects, picture anchors, videos, or audios)", left, sy(cy + 10), "#93c5fd", "bold 12px sans-serif");
      cy += 24;

      // 4 Action Buttons Bar: + Subject, + Picture, + Video, + Audio
      const nextSubId = this.state.definitions.filter(d => d.kind === "subject").length + 1;
      const nextPicId = this.state.definitions.filter(d => d.kind === "picture").length + 1;
      const nextVidId = this.state.definitions.filter(d => d.kind === "video").length + 1;
      const nextAudId = this.state.definitions.filter(d => d.kind === "audio").length + 1;

      const btnW = Math.floor((available - 18) / 4);
      this.button(ctx, left, sy(cy), btnW, 26, `+ <Subject ${nextSubId}>`, () => {
        this.change(() => addDefinition(this.state, "subject", { hasRef: false }));
      }, true, "center", false, false, "Adds a character, creature, or key object to describe appearance, clothing, or link them to a reference image.");

      this.button(ctx, left + btnW + 6, sy(cy), btnW, 26, `+ <Picture ${nextPicId}>`, () => {
        this.change(() => addDefinition(this.state, "picture", { role: "first_frame" }));
      }, false, "center", false, false, "Adds a reference picture definition, such as a starting frame, ending frame, or storyboard composition.");

      this.button(ctx, left + (btnW + 6) * 2, sy(cy), btnW, 26, `+ <Video ${nextVidId}>`, () => {
        this.change(() => addDefinition(this.state, "video", { role: "edit" }));
      }, false, "center", false, false, "Adds a source video reference for video editing, motion transfer, or video continuation.");

      this.button(ctx, left + (btnW + 6) * 3, sy(cy), available - (btnW + 6) * 3, 26, `+ <Audio ${nextAudId}>`, () => {
        this.change(() => addDefinition(this.state, "audio", { role: "timbre" }));
      }, false, "center", false, false, "Adds a reference audio track to copy a speaker's voice timbre or reuse background music.");
      cy += 34;

      if (!this.state.definitions.length) {
        this.text(ctx, "No definitions yet. Click above to add a standalone or referenced entity.", left + 6, sy(cy + 14), "#788290");
        cy += 32;
      }

      this.state.definitions.forEach((item, idx) => {
        const cardY = sy(cy);
        const cardH = 86;
        this.box(ctx, left, cardY, available, cardH, "#22262e", "#444b56", 4);

        // Header badge & Tag
        const kindColors = {
          subject: { bg: "#1e3a5f", border: "#3b82f6", tag: `<Subject ${item.id}>` },
          picture: { bg: "#14532d", border: "#22c55e", tag: `<Picture ${item.id}>` },
          video: { bg: "#701a75", border: "#c084fc", tag: `<Video ${item.id}>` },
          audio: { bg: "#4a1d96", border: "#a855f7", tag: `<Audio ${item.id}>` },
        };
        const cfg = kindColors[item.kind] || kindColors.subject;
        this.box(ctx, left + 8, cardY + 8, 88, 20, cfg.bg, cfg.border, 3);
        this.text(ctx, cfg.tag, left + 14, cardY + 18, "#ffffff", "bold 11px monospace");

        // Action buttons (Remove / Up / Down)
        this.button(ctx, right - 28, cardY + 6, 24, 22, "×", () => {
          this.change(() => removeDefinition(this.state, idx));
        }, false, "center", false, true, "Removes this reference definition.");
        this.button(ctx, right - 54, cardY + 6, 22, 22, "↑", () => {
          this.change(() => moveItem(this.state.definitions, idx, -1));
        }, false, "center", false, false, "Moves this item up in order.");
        this.button(ctx, right - 78, cardY + 6, 22, 22, "↓", () => {
          this.change(() => moveItem(this.state.definitions, idx, 1));
        }, false, "center", false, false, "Moves this item down in order.");

        // Config Controls per Kind
        if (item.kind === "subject") {
          // Reference Switcher: Standalone (No Ref) vs Referenced
          const noRefW = this.chip(ctx, left + 104, cardY + 7, item.hasRef ? "[With Ref]" : "[Standalone (No Ref)]", () => {
            this.change(() => { item.hasRef = !item.hasRef; });
          }, !item.hasRef, item.hasRef, item.hasRef ? "This character takes visual identity and clothing from a reference Picture or Video." : "This character is described entirely through text without requiring an image reference.");

          if (item.hasRef) {
            const pic1W = this.chip(ctx, left + 110 + noRefW, cardY + 7, "<Pic 1>", () => {
              this.change(() => { item.refType = "picture"; item.refIndex = 1; });
            }, item.refType === "picture" && item.refIndex === 1);

            const pic2W = this.chip(ctx, left + 114 + noRefW + pic1W, cardY + 7, "<Pic 2>", () => {
              this.change(() => { item.refType = "picture"; item.refIndex = 2; });
            }, item.refType === "picture" && item.refIndex === 2);

            const vid1W = this.chip(ctx, left + 118 + noRefW + pic1W + pic2W, cardY + 7, "<Vid 1>", () => {
              this.change(() => { item.refType = "video"; item.refIndex = 1; });
            }, item.refType === "video" && item.refIndex === 1);

            this.chip(ctx, left + 122 + noRefW + pic1W + pic2W + vid1W, cardY + 7, "Idx #", event => {
              this.editSingleLine("Reference Index Number", String(item.refIndex || 1), val => {
                const n = parseInt(val, 10);
                if (n > 0) item.refIndex = n;
              }, event);
            });
          }

          // Field: details/traits
          const prefixLabel = item.hasRef
            ? `<Subject ${item.id}> is fully referenced in <${item.refType === "video" ? "Video" : "Picture"} ${item.refIndex}>:`
            : `<Subject ${item.id}> is:`;
          this.text(ctx, prefixLabel, left + 10, cardY + 40, "#cbd5e1", "11px monospace", available - 20);

          this.field(ctx, "Visual Characteristics & Clothing", item.text, left + 8, cardY + 42, available - 16, (_ev, _pos, rect) => {
            this.openTextEditor(item.text, val => { item.text = val; }, rect, availableTags);
          });
        }
        else if (item.kind === "picture") {
          // Role selector for Standalone Picture
          const roles = [
            { id: "first_frame", label: "First Frame (00.00s)", tip: "Locks this picture as the exact starting frame at 00.00s." },
            { id: "final_frame", label: "Final Frame", tip: "Locks this picture as the exact ending frame at the end of the video." },
            { id: "storyboard", label: "Storyboard", tip: "Uses this picture as a camera angle and composition reference for shots." },
            { id: "custom", label: "Custom", tip: "Custom image reference role." },
          ];
          let rx = left + 104;
          roles.forEach(r => {
            rx += this.chip(ctx, rx, cardY + 7, r.label, () => {
              this.change(() => { item.role = r.id; });
            }, item.role === r.id, false, r.tip) + 4;
          });

          if (item.role === "custom") {
            this.field(ctx, "Picture Anchor Role Description", item.text, left + 8, cardY + 42, available - 16, (_ev, _pos, rect) => {
              this.openTextEditor(item.text, val => { item.text = val; }, rect, availableTags);
            });
          } else {
            let roleSummary = `<Picture ${item.id}> is the fixed first frame anchor at 00.00s.`;
            if (item.role === "final_frame") roleSummary = `<Picture ${item.id}> is the fixed final frame anchor at video endpoint.`;
            else if (item.role === "storyboard") roleSummary = `<Picture ${item.id}> is a storyboard reference for [Shot 1] and [Shot 2].`;
            this.text(ctx, roleSummary, left + 10, cardY + 48, "#94a3b8", "11px monospace", available - 20);
          }
        }
        else if (item.kind === "video") {
          // Role selector for Standalone Video
          const vRoles = [
            { id: "edit", label: "Edit Source", tip: "Uses this video as the source video to edit or modify." },
            { id: "continue", label: "Continuation", tip: "Extends this video forward in time from its ending." },
            { id: "structure", label: "Pacing & Motion", tip: "Uses this video as a guide for camera motion and rhythm without copying characters." },
            { id: "custom", label: "Custom", tip: "Custom video reference role." },
          ];
          let vx = left + 104;
          vRoles.forEach(r => {
            vx += this.chip(ctx, vx, cardY + 7, r.label, () => {
              this.change(() => { item.role = r.id; });
            }, item.role === r.id, false, r.tip) + 4;
          });

          if (item.role === "custom") {
            this.field(ctx, "Video Role Description", item.text, left + 8, cardY + 42, available - 16, (_ev, _pos, rect) => {
              this.openTextEditor(item.text, val => { item.text = val; }, rect, availableTags);
            });
          } else {
            let vSummary = `<Video ${item.id}> is the source video for the target video edit.`;
            if (item.role === "continue") vSummary = `<Video ${item.id}> is the source video for continuation.`;
            else if (item.role === "structure") vSummary = `<Video ${item.id}> provides camera movement, cuts, and temporal structure.`;
            this.text(ctx, vSummary, left + 10, cardY + 48, "#94a3b8", "11px monospace", available - 20);
          }
        }
        else if (item.kind === "audio") {
          // Role selector for Standalone Audio
          const aRoles = [
            { id: "timbre", label: "Voice Timbre", tip: "Copies the vocal sound and tone of this audio onto a speaking character." },
            { id: "full", label: "Full Track", tip: "Reuses this audio track as the complete final sound of the video." },
            { id: "music", label: "Music & Rhythm", tip: "Uses this audio as a reference for background music style and rhythm." },
            { id: "custom", label: "Custom", tip: "Custom audio reference role." },
          ];
          let ax = left + 104;
          aRoles.forEach(r => {
            ax += this.chip(ctx, ax, cardY + 7, r.label, () => {
              this.change(() => { item.role = r.id; });
            }, item.role === r.id, false, r.tip) + 4;
          });

          if (item.role === "timbre") {
            this.chip(ctx, ax, cardY + 7, `Speaker: <Subj ${item.targetSubject || 1}>`, event => {
              this.editSingleLine("Target Subject ID for Voice Timbre", String(item.targetSubject || 1), val => {
                const n = parseInt(val, 10);
                if (n > 0) item.targetSubject = n;
              }, event);
            });
            const timbreSummary = `<Audio ${item.id}> is the voice-timbre reference for <Subject ${item.targetSubject || 1}> (S${item.targetSubject || 1}).`;
            this.text(ctx, timbreSummary, left + 10, cardY + 48, "#94a3b8", "11px monospace", available - 20);
          } else if (item.role === "custom") {
            this.field(ctx, "Audio Role Description", item.text, left + 8, cardY + 42, available - 16, (_ev, _pos, rect) => {
              this.openTextEditor(item.text, val => { item.text = val; }, rect, availableTags);
            });
          } else {
            let aSummary = `<Audio ${item.id}> is reused as the target video's complete final audio track.`;
            if (item.role === "music") aSummary = `<Audio ${item.id}> is the music-style and rhythm reference.`;
            this.text(ctx, aSummary, left + 10, cardY + 48, "#94a3b8", "11px monospace", available - 20);
          }
        }

        cy += cardH + 8;
      });
    }

    // --- TAB 2: SUMMARY ---
    else if (this.state.activeTab === "summary") {
      this.text(ctx, "summary: (One task-prefixed paragraph describing final target)", left, sy(cy + 10), "#93c5fd", "bold 12px sans-serif");
      cy += 24;

      this.text(ctx, "Task Types (Principle 4 allowed values):", left, sy(cy + 8), "#aeb5bf", "11px sans-serif");
      cy += 18;

      let chipX = left;
      TASK_TYPES.forEach(task => {
        const active = (this.state.summary.taskTypes || []).includes(task);
        ctx.font = "11px sans-serif";
        const cw = Math.ceil(ctx.measureText(task).width) + 16;
        if (chipX + cw > right) {
          chipX = left;
          cy += 28;
        }
        this.chip(ctx, chipX, sy(cy), task, () => {
          this.change(() => {
            const list = this.state.summary.taskTypes || [];
            if (list.includes(task)) {
              this.state.summary.taskTypes = list.filter(t => t !== task);
            } else {
              list.push(task);
              this.state.summary.taskTypes = list;
            }
          });
        }, active, true);
        chipX += cw + 6;
      });
      cy += 32;

      // Compiled Task Prefix Preview
      const prefix = (this.state.summary.taskTypes || []).length
        ? `[${this.state.summary.taskTypes.join(" + ")}]`
        : "[no task type selected]";
      this.text(ctx, `Prefix: ${prefix}`, left, sy(cy + 8), "#94a3b8", "11px monospace", available);
      cy += 20;

      // Quick Tag Insertion Bar
      if (availableTags.length) {
        this.text(ctx, "Insert tag at cursor:", left, sy(cy + 6), "#aeb5bf", "11px sans-serif");
        cy += 16;
        let tagX = left;
        availableTags.forEach(tag => {
          ctx.font = "11px monospace";
          const tw = Math.ceil(ctx.measureText(tag).width) + 14;
          if (tagX + tw > right) {
            tagX = left;
            cy += 26;
          }
          this.chip(ctx, tagX, sy(cy), tag, () => {
            this.change(() => {
              this.state.summary.text = (this.state.summary.text ? this.state.summary.text + " " : "") + tag;
            });
          });
          tagX += tw + 4;
        });
        cy += 30;
      }

      this.field(ctx, "Summary Description", this.state.summary.text, left, sy(cy), available, (_ev, _pos, rect) => {
        this.openTextEditor(this.state.summary.text, val => { this.state.summary.text = val; }, rect, availableTags);
      });
      cy += 50;
    }

    // --- TAB 3: RETENTION ANALYSIS ---
    else if (this.state.activeTab === "retention") {
      this.text(ctx, "retention_analysis: (Format: <label>: <marker> - <descriptor>)", left, sy(cy + 10), "#93c5fd", "bold 12px sans-serif");
      cy += 24;

      this.button(ctx, left, sy(cy), 170, 26, "+ Add Retention Label", event => {
        this.editSingleLine("Tracked Label (e.g. <Subject 1> or <Video 1>)", "<Video 1>", label => {
          if (label?.trim()) {
            this.state.retention.push({
              label: label.trim(),
              marker: "attribute_transfer",
              text: defaultRetentionText(label.trim(), "attribute_transfer"),
            });
          }
        }, event);
      }, true, "center", false, false, "Adds a tracked reference label to define its retention or transfer rules.");

      this.button(ctx, left + 178, sy(cy), 120, 26, "Clear All (T2V)", () => {
        this.change(() => { this.state.retention = []; });
      }, false, "center", false, false, "Clears retention analysis (cleanly omitted when generating without references).");
      cy += 32;

      if (!this.state.retention.length) {
        this.text(ctx, "retention_analysis is empty (cleanly omitted for standalone T2VA/I2VA prompts).", left + 6, sy(cy + 14), "#788290");
        cy += 32;
      }

      this.state.retention.forEach((item, idx) => {
        const cardY = sy(cy);
        const cardH = 88;
        this.box(ctx, left, cardY, available, cardH, "#22262e", "#444b56", 4);

        // Label Badge
        this.box(ctx, left + 8, cardY + 8, 90, 20, "#1e293b", "#3b82f6", 3);
        this.text(ctx, item.label || "<Label>", left + 14, cardY + 18, "#ffffff", "bold 11px monospace");

        // Marker Selector (cycles on click)
        const isAudio = item.label?.startsWith("<Audio");
        const markers = isAudio ? AUDIO_MARKERS : VISIBLE_MARKERS;
        const markerLabel = `Marker: ${item.marker}`;
        ctx.font = "11px sans-serif";
        const mw = Math.ceil(ctx.measureText(markerLabel).width) + 16;
        this.button(ctx, left + 106, cardY + 7, mw, 22, markerLabel, () => {
          this.change(() => {
            const nextIdx = (markers.indexOf(item.marker) + 1) % markers.length;
            item.marker = markers[nextIdx];
          });
        }, true, "center", false, false, "Cycles through retention markers: attribute transfer, partially preserved, fully preserved, or weak reference.");

        // Autofill default button
        this.button(ctx, left + 112 + mw, cardY + 7, 72, 22, "Autofill", () => {
          this.change(() => {
            item.text = defaultRetentionText(item.label, item.marker);
          });
        }, false, "center", false, false, "Fills in recommended retention description text for this marker.");

        // Remove button
        this.button(ctx, right - 28, cardY + 6, 24, 22, "×", () => {
          this.change(() => {
            this.state.retention.splice(idx, 1);
          });
        }, false, "center", false, true, "Removes this retention item.");

        // Formatted preview line
        this.text(ctx, `${item.label}: ${item.marker} -`, left + 10, cardY + 40, "#94a3b8", "11px monospace", available - 20);

        // Descriptor text field
        this.field(ctx, "Relationship Descriptor", item.text, left + 8, cardY + 42, available - 16, (_ev, _pos, rect) => {
          this.openTextEditor(item.text, val => { item.text = val; }, rect, availableTags);
        });

        cy += cardH + 8;
      });
    }

    // --- TAB 4: DETAILED DESCRIPTION / TIMELINE ---
    else if (this.state.activeTab === "detailed") {
      this.text(ctx, "detailed_description: & Timeline", left, sy(cy + 10), "#93c5fd", "bold 12px sans-serif");
      cy += 24;

      // Mode Switch: Timeline vs Continuous
      const hasTimeline = this.state.detailed.hasTimeline !== false;
      this.button(ctx, left, sy(cy), 130, 26, "Timeline Mode", () => {
        this.change(() => { this.state.detailed.hasTimeline = true; });
      }, false, "center", hasTimeline, false, "Divides your video into timed chronological segments with separate visual, dialogue, and sound controls.");

      this.button(ctx, left + 136, sy(cy), 150, 26, "Continuous (No Timeline)", () => {
        this.change(() => { this.state.detailed.hasTimeline = false; });
      }, false, "center", !hasTimeline, false, "Allows writing freely in continuous paragraphs using shot labels like [Shot 1] and camera cuts.");
      cy += 34;

      if (!hasTimeline) {
        this.field(ctx, "Continuous Detailed Description", this.state.detailed.continuousText, left, sy(cy), available, (_ev, _pos, rect) => {
          this.openTextEditor(this.state.detailed.continuousText, val => { this.state.detailed.continuousText = val; }, rect, availableTags);
        });
        cy += 60;
      } else {
        // Timeline Duration Controls
        const curDur = Number(this.state.detailed.segmentDuration) || 2.333;
        const curFrames = snapH3Frames(curDur * H3_FPS);
        const durLabel = `${curDur.toFixed(2)}s (${curFrames}f)`;

        this.text(ctx, "Segment Step Duration:", left, sy(cy + 12), "#aeb5bf", "11px sans-serif");

        // Decrement button (-17 frames)
        this.button(ctx, left + 130, sy(cy), 26, 24, "-", () => {
          this.change(() => {
            this.state.detailed.segmentDuration = h3StepDuration(curDur, -1);
          });
        }, false, "center", false, false, "Shortens segment duration by 17 frames (about 0.71 seconds), matching H3 native steps.");

        // Duration display/edit
        this.button(ctx, left + 160, sy(cy), 110, 24, durLabel, event => {
          this.editSingleLine("Step Duration in Seconds", String(curDur.toFixed(2)), val => {
            const s = parseFloat(val);
            if (s > 0) this.state.detailed.segmentDuration = s;
          }, event);
        }, true, "center", false, false, "Current default segment duration. Click to edit manually.");

        // Increment button (+17 frames)
        this.button(ctx, left + 274, sy(cy), 26, 24, "+", () => {
          this.change(() => {
            this.state.detailed.segmentDuration = h3StepDuration(curDur, 1);
          });
        }, false, "center", false, false, "Lengthens segment duration by 17 frames (about 0.71 seconds), matching H3 native steps.");

        // Add Segment Button
        this.button(ctx, left + 310, sy(cy), available - 310, 24, `+ Add Segment (${curDur.toFixed(2)}s)`, () => {
          this.change(() => addSegment(this.state));
        }, true, "center", false, false, "Adds a new timed scene segment starting right after the previous one.");
        cy += 36;

        // Segments List
        this.state.segments.forEach((seg, sIdx) => {
          const collapsed = this.collapsed.has(sIdx);
          const cardH = collapsed ? 34 : 260;
          const cardY = sy(cy);
          this.box(ctx, left, cardY, available, cardH, "#22262e", "#444b56", 4);

          // Header
          const segHeader = `Segment ${sIdx + 1} [${seg.start} - ${seg.end}]`;
          this.text(ctx, segHeader, left + 8, cardY + 16, "#f3f5f7", "bold 11px monospace");

          this.button(ctx, right - 130, cardY + 5, 46, 22, collapsed ? "Open" : "Close", () => {
            if (this.collapsed.has(sIdx)) this.collapsed.delete(sIdx);
            else this.collapsed.add(sIdx);
            app.canvas?.setDirty(true, true);
          }, false, "center", false, false, "Collapses or expands this segment card.");
          this.button(ctx, right - 80, cardY + 5, 22, 22, "⧉", () => {
            this.change(() => this.state.segments.splice(sIdx + 1, 0, JSON.parse(JSON.stringify(seg))));
          }, false, "center", false, false, "Duplicates this segment.");
          this.button(ctx, right - 54, cardY + 5, 22, 22, "↑", () => {
            this.change(() => moveItem(this.state.segments, sIdx, -1));
          }, false, "center", false, false, "Moves this segment earlier in time.");
          this.button(ctx, right - 28, cardY + 5, 22, 22, "×", () => {
            this.change(() => this.state.segments.splice(sIdx, 1));
          }, false, "center", false, true, "Deletes this scene segment.");

          if (!collapsed) {
            let scy = cardY + 36;

            // Start & End editing
            const halfW = (available - 20) / 2;
            this.button(ctx, left + 8, scy, halfW, 22, `Start: ${seg.start}`, event => {
              this.editSingleLine("Segment Start Time", String(seg.start), val => { seg.start = val; }, event);
            });
            this.button(ctx, left + 12 + halfW, scy, halfW, 22, `End: ${seg.end}`, event => {
              this.editSingleLine("Segment End Time", String(seg.end), val => { seg.end = val; }, event);
            });
            scy += 26;

            // Quick Pill Bar: Shot toggle + Camera motion + Subjects
            let qx = left + 8;
            const shotLabel = seg.hasShot ? `[Shot ${seg.shot || sIdx + 1}] ✓` : `[+ Shot]`;
            qx += this.chip(ctx, qx, scy, shotLabel, () => {
              this.change(() => { seg.hasShot = !seg.hasShot; });
            }, seg.hasShot, true, "Toggles whether this segment introduces an instant camera cut or smoothly continues previous movement.") + 6;

            // Camera Motion quick insert
            ["Push In", "Pan Left", "Tilt Up", "Static Shot"].forEach(cam => {
              if (qx + 60 < right) {
                qx += this.chip(ctx, qx, scy, cam, () => {
                  this.change(() => {
                    seg.visual = (seg.visual ? seg.visual + ", " : "") + cam;
                  });
                }) + 4;
              }
            });

            // Quick subject tag inserts
            (tags.subjects || []).slice(0, 3).forEach(st => {
              if (qx + 65 < right) {
                qx += this.chip(ctx, qx, scy, st, () => {
                  this.change(() => {
                    seg.visual = (seg.visual ? seg.visual + " " : "") + st;
                  });
                }) + 4;
              }
            });
            scy += 28;

            // [VISUAL] Field
            this.field(ctx, "[VISUAL]: Chronological visual & camera description", seg.visual, left + 8, scy, available - 16, (_ev, _pos, rect) => {
              this.openTextEditor(seg.visual, val => { seg.visual = val; }, rect, availableTags);
            });
            scy += 48;

            // [SPEECH] Channel
            const spkActive = Boolean(seg.speech?.enabled);
            this.button(ctx, left + 8, scy + 8, 80, 22, `[SPEECH]`, () => {
              this.change(() => {
                seg.speech = seg.speech || {};
                seg.speech.enabled = !spkActive;
              });
            }, false, "center", spkActive, false, "Enables spoken dialogue, speaker identity, and spoken language for this segment.");

            if (spkActive) {
              this.button(ctx, left + 92, scy + 8, 50, 22, seg.speech.speaker || "S1", event => {
                this.editSingleLine("Speaker ID (e.g. S1)", seg.speech.speaker || "S1", val => { seg.speech.speaker = val; }, event);
              }, false, "center", false, false, "Speaker ID for this line (e.g. S1 or S2).");
              this.button(ctx, left + 146, scy + 8, 60, 22, seg.speech.language || "English", event => {
                this.editSingleLine("Language", seg.speech.language || "English", val => { seg.speech.language = val; }, event);
              }, false, "center", false, false, "Spoken language name inside the dialogue tag.");
              this.field(ctx, "Spoken Words", seg.speech.text, left + 210, scy - 8, available - 218, (_ev, _pos, rect) => {
                this.openTextEditor(seg.speech.text, val => { seg.speech.text = val; }, rect, availableTags);
              });
            }
            scy += 34;

            // [SOUNDS] Channel
            const sndActive = Boolean(seg.sounds?.enabled);
            this.button(ctx, left + 8, scy + 8, 80, 22, `[SOUNDS]`, () => {
              this.change(() => {
                seg.sounds = seg.sounds || {};
                seg.sounds.enabled = !sndActive;
              });
            }, false, "center", sndActive, false, "Enables synchronized physical action sounds, impacts, and ambient effects for this segment.");

            if (sndActive) {
              this.field(ctx, "Ambience & Sound Effects", seg.sounds.text, left + 92, scy - 8, available - 100, (_ev, _pos, rect) => {
                this.openTextEditor(seg.sounds.text, val => { seg.sounds.text = val; }, rect, availableTags);
              });
            }
            scy += 34;

            // [MUSIC] Channel
            const musActive = Boolean(seg.music?.enabled);
            this.button(ctx, left + 8, scy + 8, 80, 22, `[MUSIC]`, () => {
              this.change(() => {
                seg.music = seg.music || {};
                seg.music.enabled = !musActive;
              });
            }, false, "center", musActive, false, "Enables background music notes specific to this segment.");

            if (musActive) {
              this.field(ctx, "Diegetic Music", seg.music.text, left + 92, scy - 8, available - 100, (_ev, _pos, rect) => {
                this.openTextEditor(seg.music.text, val => { seg.music.text = val; }, rect, availableTags);
              });
            }
          }

          cy += cardH + 8;
        });
      }
    }

    // --- TAB 5: SOUNDSCAPE ---
    else if (this.state.activeTab === "soundscape") {
      this.text(ctx, "overall_soundscape: (Continuous ambient sound paragraph)", left, sy(cy + 10), "#93c5fd", "bold 12px sans-serif");
      cy += 24;

      this.field(ctx, "Overall Soundscape Description", this.state.overall_soundscape, left, sy(cy), available, (_ev, _pos, rect) => {
        this.openTextEditor(this.state.overall_soundscape, val => { this.state.overall_soundscape = val; }, rect, availableTags);
      });
      cy += 60;
    }

    // --- TAB 6: MUSIC ---
    else if (this.state.activeTab === "music") {
      this.text(ctx, "non_diegetic_music: (One to three English sentences or N/A)", left, sy(cy + 10), "#93c5fd", "bold 12px sans-serif");
      cy += 24;

      const isNA = (this.state.non_diegetic_music || "").trim() === "N/A";
      this.button(ctx, left, sy(cy), 90, 26, "Set N/A", () => {
        this.change(() => { this.state.non_diegetic_music = "N/A"; });
      }, false, "center", isNA, false, "Sets background music to N/A when no music should play.");
      cy += 32;

      this.field(ctx, "Background Score / Music Description", this.state.non_diegetic_music, left, sy(cy), available, (_ev, _pos, rect) => {
        this.openTextEditor(this.state.non_diegetic_music, val => { this.state.non_diegetic_music = val; }, rect, availableTags);
      });
      cy += 60;
    }

    this.contentHeight = cy;
    const maxScroll = Math.max(0, cy - this.viewportHeight);
    if (this.scroll > maxScroll) {
      this.scroll = maxScroll;
      app.canvas?.setDirty(true, true);
    }
    if (maxScroll > 0) {
      const trackY = contentTopY + 2;
      const trackHeight = this.viewportHeight - 4;
      const thumbHeight = Math.max(24, trackHeight * this.viewportHeight / cy);
      const thumbY = trackY + (trackHeight - thumbHeight) * this.scroll / maxScroll;
      this.box(ctx, width - 8, trackY, 4, trackHeight, "#17191d", null, 2);
      this.box(ctx, width - 8, thumbY, 4, thumbHeight, "#8290a0", null, 2);
      this.hit(width - 12, trackY, 10, trackHeight, (_event, position) => {
        const ratio = (position[1] - trackY - thumbHeight / 2) / (trackHeight - thumbHeight);
        this.scroll = Math.max(0, Math.min(maxScroll, ratio * maxScroll));
        app.canvas?.setDirty(true, true);
      });
    }
    ctx.restore();

    // Bottom Preview Area (Fixed below content viewport)
    const previewY = y + this.viewportHeight + (contentTopY - y) + 4;
    this.box(ctx, left, previewY, available, 60, "#121418", "#2d323b", 3);
    const warnings = timelineWarnings(this.state);
    const statusText = warnings.length ? `⚠ ${warnings[0]}` : `Prompt compiled: ${compilePrompt(this.state).length} chars`;
    const statusColor = warnings.length ? "#fbbf24" : "#4ade80";
    this.text(ctx, statusText, left + 6, previewY + 12, statusColor, "11px sans-serif", available - 12);

    const promptPreview = compilePrompt(this.state) || "Prompt preview will appear here...";
    const previewLine = promptPreview.replace(/\n+/g, " ❚ ");
    this.text(ctx, previewLine, left + 6, previewY + 34, "#94a3b8", "11px monospace", available - 12);

    // Canvas Draw-Loop Tooltip (Offset upwards on vertical axis)
    if (this.hoveredRegion?.tooltip) {
      this.drawCanvasTooltip(ctx, this.hoveredRegion, width, y, this.viewportHeight);
    }
  }

  mouse(event, position) {
    if (event.button !== 0 || !/up$/.test(event.type)) return true;
    this.clearHoveredTooltip();
    const region = this.hitRegions.find(item => this.contains(item, position[0], position[1]));
    region?.action(event, position);
    return true;
  }

  onWheel(event) {
    if (this.textEditor) this.closeTextEditor();
    this.clearHoveredTooltip();
    if (event.ctrlKey || event.metaKey || !this.state) return;
    const canvas = app.canvas;
    if (!canvas?.graph || this.contentHeight <= this.viewportHeight) return;
    const bounds = canvas.canvas.getBoundingClientRect();
    const graphX = (event.clientX - bounds.left) / canvas.ds.scale - canvas.ds.offset[0];
    const graphY = (event.clientY - bounds.top) / canvas.ds.scale - canvas.ds.offset[1];
    if (canvas.graph.getNodeOnPos(graphX, graphY, canvas.visible_nodes) !== this.node) return;
    const x = graphX - this.node.pos[0];
    const y = graphY - this.node.pos[1];
    if (x < 4 || x > this.node.size[0] - 4 || y < this.viewportY || y > this.viewportY + this.viewportHeight) return;
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.viewportHeight : 1);
    const next = Math.max(0, Math.min(this.contentHeight - this.viewportHeight, this.scroll + delta));
    if (next === this.scroll) return;
    this.scroll = next;
    event.preventDefault();
    event.stopImmediatePropagation();
    canvas.setDirty(true, true);
  }
}

app.registerExtension({
  name: "UtilsCollection.MiniMaxH3PromptBuilder",
  getCustomWidgets() {
    return {
      UC_MINIMAX_H3_PROMPT_BUILDER(node, name, data) {
        const editor = new H3CanvasPromptEditor(node, name, data);
        return { widget: editor.widget, minWidth: 480, minHeight: 500 };
      },
    };
  },
});
