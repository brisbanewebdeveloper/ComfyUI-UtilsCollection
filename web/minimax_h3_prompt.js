import { app } from "../../scripts/app.js";
import {
  CHANNELS, HEADER_CHOICES, addSegment, compilePrompt, createState,
  moveItem, restoreState, timelineWarnings,
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
    this.viewportHeight = 360;
    this.hitRegions = [];
    this.textEditor = null;
    this.abort = new AbortController();
    this.restore(this.raw);

    this.widget = node.addCustomWidget({
      name,
      type: "custom",
      value: this.raw,
      options: { socketless: true },
      computeSize: () => [node.size[0], 360],
      draw: (ctx, _node, width, y) => this.draw(ctx, width, y),
      mouse: (event, position) => this.mouse(event, position),
      serializeValue: () => this.serialize(),
    });
    this.widget.options.socketless = true;

    // Leave everything outside painted controls to LiteGraph's node drag and menu handling.
    const getWidgetOnPos = node.getWidgetOnPos;
    node.getWidgetOnPos = (graphX, graphY, includeDisabled) => {
      const x = graphX - node.pos[0];
      const y = graphY - node.pos[1];
      if (this.hitRegions.some(region => this.contains(region, x, y))) return this.widget;
      const found = getWidgetOnPos.call(node, graphX, graphY, includeDisabled);
      return found === this.widget ? undefined : found;
    };
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
      if (!this.abort.signal.aborted) node.setSize([Math.max(410, node.size[0]), Math.max(440, node.size[1])]);
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

  hit(x, y, w, h, action) {
    const top = Math.max(y, this.viewportY);
    const bottom = Math.min(y + h, this.viewportY + this.viewportHeight);
    if (bottom > top) this.hitRegions.push({ x, y: top, w, h: bottom - top, action });
  }

  box(ctx, x, y, w, h, fill, stroke = "#4b505a", radius = 5) {
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

  button(ctx, x, y, w, h, label, action, accent = false, align = "center") {
    this.box(ctx, x, y, w, h, accent ? "#26394e" : "#202329", accent ? "#709ecc" : "#59606a", 4);
    ctx.font = "12px sans-serif";
    const tw = ctx.measureText(label).width;
    this.text(ctx, label, align === "left" ? x + 8 : x + Math.max(6, (w - tw) / 2),
      y + h / 2, accent ? "#d7ebff" : "#ddd", "12px sans-serif", w - 12);
    this.hit(x, y, w, h, action);
  }

  field(ctx, label, value, x, y, w, action) {
    this.text(ctx, label, x, y + 7, "#aeb5bf");
    const rect = { x, y: y + 17, w, h: 27 };
    this.button(ctx, rect.x, rect.y, rect.w, rect.h, String(value || "Click to edit"),
      (event, position) => action(event, position, rect), false, "left");
  }

  editSingleLine(title, value, apply, event) {
    app.canvas.prompt(title, value, text => this.change(() => apply(text)), event);
  }

  openTextEditor(value, apply, rect) {
    this.closeTextEditor();
    const element = document.createElement("textarea");
    element.className = "comfy-multiline-input";
    element.dataset.testid = "h3-prompt-textarea";
    element.value = value;
    element.spellcheck = true;
    Object.assign(element.style, {
      position: "fixed", zIndex: "1000", boxSizing: "border-box", margin: "0",
      border: "1px solid #709ecc", borderRadius: "4px", outline: "none", resize: "none",
      fontFamily: "Inter, Arial, sans-serif", lineHeight: "1.35",
    });
    this.textEditor = { element, rect };
    element.addEventListener("input", () => this.change(() => apply(element.value)));
    element.addEventListener("keydown", event => {
      event.stopPropagation();
      if (event.key === "Escape" || (event.key === "Enter" && (event.ctrlKey || event.metaKey))) {
        event.preventDefault();
        this.closeTextEditor();
      }
    });
    for (const type of ["pointerdown", "pointermove", "pointerup", "click", "dblclick"]) {
      element.addEventListener(type, event => event.stopPropagation());
    }
    element.addEventListener("contextmenu", event => event.stopPropagation());
    element.addEventListener("wheel", event => {
      if (element.scrollHeight > element.clientHeight) {
        event.stopPropagation();
        return;
      }
      event.preventDefault();
      app.canvas?.processMouseWheel(event);
    });
    element.addEventListener("blur", () => this.closeTextEditor());
    document.body.append(element);
    this.positionTextEditor();
    requestAnimationFrame(() => element.focus());
  }

  positionTextEditor() {
    if (!this.textEditor) return;
    const canvas = app.canvas?.canvas;
    if (!canvas) return this.closeTextEditor();
    const { element, rect } = this.textEditor;
    const bounds = canvas.getBoundingClientRect();
    const { scale, offset } = app.canvas.ds;
    const left = bounds.left + (this.node.pos[0] + rect.x + offset[0]) * scale;
    const top = bounds.top + (this.node.pos[1] + rect.y + offset[1]) * scale;
    Object.assign(element.style, {
      left: `${left}px`, top: `${top}px`, width: `${rect.w * scale}px`,
      height: `${Math.max(84, rect.h) * scale}px`,
      fontSize: `${12 * scale}px`,
    });
  }

  closeTextEditor() {
    if (!this.textEditor) return;
    this.textEditor.element.remove();
    this.textEditor = null;
    app.canvas?.setDirty(true, true);
  }

  draw(ctx, width, y) {
    this.sync();
    this.viewportY = y;
    this.viewportHeight = Math.max(300, this.node.size[1] - y - 8);
    this.hitRegions = [];
    const left = 12;
    const right = width - 12;
    const available = right - left;
    const top = y;
    const sy = contentY => top + contentY - this.scroll;
    let cy = 9;
    ctx.save();
    ctx.beginPath();
    ctx.rect(4, y, width - 8, this.viewportHeight);
    ctx.clip();
    this.box(ctx, 5, y, width - 10, this.viewportHeight, "#26292f", null, 5);

    this.text(ctx, "MiniMax H3 prompt", left, sy(cy + 14), "#eee", "bold 14px sans-serif");
    if (this.state) this.button(ctx, right - 96, sy(cy), 96, 28,
      this.state.precision + " decimals",
      () => this.change(() => { this.state.precision = this.state.precision === 2 ? 3 : 2; }));
    cy += 39;
    this.text(ctx, "Prompt headers", left, sy(cy + 7), "#e4e7eb", "bold 12px sans-serif");
    cy += 22;
    this.text(ctx, "Describe subjects and overall intent.", left, sy(cy + 7), "#aeb5bf");
    cy += 24;

    if (this.error) {
      this.text(ctx, this.error, left, sy(cy + 10), "#ffb0b0", "12px sans-serif", available);
      cy += 40;
    } else {
      this.state.headers.forEach((header, index) => {
        const cardY = sy(cy);
        this.box(ctx, left, cardY, available, 112, "#2b2e35", "#50545d");
        this.text(ctx, header.name || "Custom header", left + 9, cardY + 17,
          "#e8e8e8", "bold 12px sans-serif", available - 168);
        this.button(ctx, right - 123, cardY + 4, 27, 26, "↑",
          () => this.change(() => moveItem(this.state.headers, index, -1)));
        this.button(ctx, right - 92, cardY + 4, 27, 26, "↓",
          () => this.change(() => moveItem(this.state.headers, index, 1)));
        this.button(ctx, right - 61, cardY + 4, 61, 26, "Remove",
          () => this.change(() => this.state.headers.splice(index, 1)));
        this.field(ctx, "Header name", header.name, left + 9, cardY + 35, available - 18,
          event => this.editSingleLine("Header name", header.name,
            value => { header.name = value; }, event));
        this.field(ctx, "Text", header.text, left + 9, cardY + 69, available - 18,
          (_event, _position, rect) => this.openTextEditor(header.text,
            value => { header.text = value; }, rect));
        cy += 119;
      });

      let chipX = left;
      for (const name of [...HEADER_CHOICES, ""]) {
        const label = name ? name.replaceAll("_", " ") : "Custom +";
        ctx.font = "12px sans-serif";
        const chipWidth = Math.min(available, Math.ceil(ctx.measureText(label).width) + 19);
        if (chipX + chipWidth > right) {
          chipX = left;
          cy += 32;
        }
        this.button(ctx, chipX, sy(cy), chipWidth, 27, label,
          event => {
            if (name) this.change(() => this.state.headers.push({ name, text: "" }));
            else this.editSingleLine("Custom header name", "", value => {
              if (value.trim()) this.state.headers.push({ name: value.trim(), text: "" });
            }, event);
          });
        chipX += chipWidth + 5;
      }
      cy += 43;

      this.text(ctx, "Timeline", left, sy(cy + 7), "#e4e7eb", "bold 12px sans-serif");
      cy += 22;
      this.text(ctx, "Add segments to describe what happens over time.", left, sy(cy + 7), "#aeb5bf");
      cy += 24;
      this.state.segments.forEach((segment, index) => {
        const collapsed = this.collapsed.has(index);
        const cardHeight = collapsed ? 36 : 222;
        const cardY = sy(cy);
        this.box(ctx, left, cardY, available, cardHeight, "#2b2e35", "#50545d");
        this.text(ctx, "Segment " + (index + 1), left + 9, cardY + 18, "#eee", "bold 12px sans-serif");
        this.button(ctx, right - 181, cardY + 4, 49, 27, collapsed ? "Open" : "Close",
          () => {
            if (this.collapsed.has(index)) this.collapsed.delete(index);
            else this.collapsed.add(index);
            app.canvas?.setDirty(true, true);
          });
        this.button(ctx, right - 128, cardY + 4, 27, 27, "⧉",
          () => this.change(() => this.state.segments.splice(index + 1, 0, structuredClone(segment))));
        this.button(ctx, right - 97, cardY + 4, 27, 27, "↑",
          () => this.change(() => moveItem(this.state.segments, index, -1)));
        this.button(ctx, right - 66, cardY + 4, 27, 27, "↓",
          () => this.change(() => moveItem(this.state.segments, index, 1)));
        this.button(ctx, right - 35, cardY + 4, 35, 27, "×",
          () => this.change(() => this.state.segments.splice(index, 1)));
        if (!collapsed) {
          const half = (available - 23) / 2;
          this.field(ctx, "Start", segment.start, left + 9, cardY + 37, half,
            event => this.editSingleLine("Segment start", segment.start,
              value => { segment.start = value; }, event));
          this.field(ctx, "End", segment.end, left + 14 + half, cardY + 37, half,
            event => this.editSingleLine("Segment end", segment.end,
              value => { segment.end = value; }, event));
          CHANNELS.forEach((channel, channelIndex) => {
            const rowY = cardY + 85 + channelIndex * 33;
            const entry = segment.channels[channel];
            this.button(ctx, left + 9, rowY, 91, 27,
              (entry.enabled ? "✓ " : "○ ") + channel.toUpperCase(),
              () => this.change(() => { entry.enabled = !entry.enabled; }), entry.enabled);
            if (entry.enabled) this.button(ctx, left + 105, rowY, available - 114, 27,
              entry.text || "Click to describe", (_event, _position) => this.openTextEditor(entry.text,
                value => { entry.text = value; }, { x: left + 105, y: rowY, w: available - 114, h: 27 }),
              false, "left");
          });
        }
        cy += cardHeight + 7;
      });
      this.button(ctx, left, sy(cy), available, 29, "Add segment +",
        () => this.change(() => addSegment(this.state)), true);
      cy += 44;
      this.text(ctx, "Prompt preview", left, sy(cy + 7), "#e4e7eb", "bold 12px sans-serif");
      cy += 22;
      let prompt = "";
      let message = "Empty fields stay out of output.";
      let messageColor = "#aeb5bf";
      try {
        prompt = compilePrompt(this.state);
        const warnings = timelineWarnings(this.state);
        if (warnings.length) {
          message = warnings.join(" ");
          messageColor = "#f0cb79";
        }
      } catch (error) {
        message = error.message;
        messageColor = "#ffb0b0";
      }
      this.text(ctx, message, left, sy(cy + 7), messageColor, "12px sans-serif", available);
      cy += 24;
      const lines = (prompt || "Prompt appears here as you write.").split("\n");
      const shown = lines.slice(0, 8);
      const previewHeight = Math.max(43, shown.length * 16 + 14);
      this.box(ctx, left, sy(cy), available, previewHeight, "#1f2228", "#50545d");
      shown.forEach((line, index) => this.text(ctx, line || " ", left + 8, sy(cy + 16 + index * 16),
        "#ddd", "12px monospace", available - 16));
      if (lines.length > shown.length) this.text(ctx, "…", right - 18, sy(cy + previewHeight - 10));
      cy += previewHeight + 9;
    }

    this.contentHeight = cy;
    const maxScroll = Math.max(0, cy - this.viewportHeight);
    if (this.scroll > maxScroll) {
      this.scroll = maxScroll;
      app.canvas?.setDirty(true, true);
    }
    if (maxScroll > 0) {
      const trackY = y + 4;
      const trackHeight = this.viewportHeight - 8;
      const thumbHeight = Math.max(28, trackHeight * this.viewportHeight / cy);
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
    this.positionTextEditor();
  }

  mouse(event, position) {
    if (event.button !== 0 || !/up$/.test(event.type)) return true;
    const region = this.hitRegions.find(item => this.contains(item, position[0], position[1]));
    region?.action(event, position);
    return true;
  }

  onWheel(event) {
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
        return { widget: editor.widget, minWidth: 410, minHeight: 360 };
      },
    };
  },
});
