import { app } from "../../scripts/app.js";
import {
  CHANNELS, HEADER_CHOICES, addSegment, compilePrompt, createState,
  moveItem, restoreState, timelineWarnings,
} from "./minimax_h3_prompt_model.js";

const STYLE = `
:host { display:block; width:100%; height:100%; color:var(--input-text,#ddd); font:12px/1.35 sans-serif; }
* { box-sizing:border-box; }
.editor { width:100%; height:100%; overflow:auto; overscroll-behavior:contain; padding:8px; scrollbar-width:thin; }
.top, .toolbar, .row, .card-title, .actions { display:flex; align-items:center; gap:5px; flex-wrap:wrap; }
.top { justify-content:space-between; margin-bottom:9px; }
.title { font-size:14px; font-weight:700; }
.hint { color:var(--descrip-text,#aaa); margin:3px 0 7px; }
.section { margin-bottom:12px; }
.section-title { font-weight:700; font-size:12px; margin:0 0 5px; }
.toolbar { margin:4px 0 7px; }
.card { border:1px solid var(--border-color,#555); border-radius:6px; padding:7px; margin:0 0 6px; background:var(--comfy-menu-bg,#26292e); min-width:0; }
.card-title { margin-bottom:5px; }
.card-title strong { margin-right:auto; }
.actions { margin-left:auto; }
.field { display:flex; flex-direction:column; gap:3px; margin:4px 0; min-width:0; flex:1 1 100%; }
.field span { color:var(--descrip-text,#bbb); }
.row .field { flex:1 1 130px; }
.time-row .field { flex:1 1 95px; }
input, textarea, select, button { color:var(--input-text,#ddd); background:var(--comfy-input-bg,#30343b); border:1px solid var(--border-color,#555b64); border-radius:4px; font:inherit; padding:4px 6px; min-width:0; }
input, textarea { width:100%; }
textarea { resize:vertical; min-height:54px; line-height:1.4; }
button, select { cursor:pointer; min-height:26px; }
button:hover { border-color:#89b5df; }
button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible { outline:2px solid #89b5df; outline-offset:1px; }
.small { padding:2px 5px; min-height:23px; }
.primary { border-color:#709ecc; }
.channel { margin-top:6px; padding-top:5px; border-top:1px solid var(--border-color,#555); }
.channel label { display:flex; align-items:center; gap:5px; cursor:pointer; }
.channel input[type=checkbox] { width:auto; margin:0; accent-color:#89b5df; }
.channel textarea { margin-top:5px; }
.muted { color:var(--descrip-text,#aaa); }
.preview { white-space:pre-wrap; overflow-wrap:anywhere; max-height:180px; overflow:auto; background:var(--comfy-input-bg,#202228); border:1px solid var(--border-color,#555); border-radius:4px; padding:7px; margin:5px 0; user-select:text; }
.message { margin:4px 0; }
.error { color:#ffb0b0; }
.warning { color:#f0cb79; }
`;

class H3PromptEditor {
  constructor(node, name, data) {
    this.node = node;
    this.document = globalThis.document;
    this.window = this.document.defaultView;
    this.raw = data[1].default ?? JSON.stringify(createState());
    this.disposed = false;
    this.collapsed = new Set();
    this.abort = new this.window.AbortController();
    this.element = this.el("div");
    Object.assign(this.element.style, { width: "100%", height: "100%", minWidth: "0", minHeight: "0" });
    const shadow = this.element.attachShadow({ mode: "open" });
    const style = this.el("style");
    style.textContent = STYLE;
    this.editor = this.el("div", "editor");
    this.editor.setAttribute("aria-label", "MiniMax H3 prompt editor");
    shadow.append(style, this.editor);
    this.restore(this.raw);
    this.widget = node.addDOMWidget(name, "uc_h3_prompt_editor", this.element, {
      socketless: true,
      hideOnZoom: false,
      margin: 4,
      getMinHeight: () => 360,
      getValue: () => this.error ? this.raw : JSON.stringify(this.state),
      setValue: value => this.restore(value),
    });
    this.widget.options.socketless = true;
    this.widget.serializeValue = () => {
      if (this.error) throw new Error(this.error);
      if (this.disposed) throw new Error("Prompt editor was removed.");
      compilePrompt(this.state);
      return JSON.stringify(this.state);
    };
    const remove = this.widget.onRemove;
    this.widget.onRemove = (...args) => {
      this.abort.abort();
      this.disposed = true;
      remove?.apply(this.widget, args);
    };
    const options = { signal: this.abort.signal };
    this.editor.addEventListener("click", event => this.onClick(event), options);
    this.editor.addEventListener("input", event => this.onInput(event), options);
    this.editor.addEventListener("change", event => this.onChange(event), options);
    this.editor.addEventListener("keydown", event => {
      if (event.target.matches("input,textarea,select,button")) event.stopPropagation();
    }, options);
    this.editor.addEventListener("pointerdown", event => {
      if (event.button !== 1 && event.target.closest("input,textarea,select,button,label")) event.stopPropagation();
    }, options);
    this.editor.addEventListener("wheel", event => {
      if (event.ctrlKey || event.metaKey) return;
      const canScroll = event.deltaY < 0 ? this.editor.scrollTop > 0
        : this.editor.scrollTop + this.editor.clientHeight < this.editor.scrollHeight;
      if (canScroll) event.stopPropagation();
    }, options);
    this.window.requestAnimationFrame(() => {
      if (!this.disposed) node.setSize([Math.max(410, node.size[0]), Math.max(440, node.size[1])]);
    });
  }

  el(tag, className, text) {
    const element = this.document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  button(text, action, index, title = text) {
    const button = this.el("button", "small", text);
    button.type = "button";
    button.dataset.action = action;
    if (index !== undefined) button.dataset.index = String(index);
    button.title = title;
    button.setAttribute("aria-label", title);
    return button;
  }

  field(label, value, scope, index, key, multiline = false) {
    const wrapper = this.el("label", "field");
    wrapper.append(this.el("span", "", label));
    const input = this.el(multiline ? "textarea" : "input");
    if (!multiline) input.type = "text";
    input.value = value;
    input.dataset.scope = scope;
    input.dataset.index = String(index);
    input.dataset.key = key;
    input.setAttribute("aria-label", label);
    if (multiline) input.rows = 2;
    wrapper.append(input);
    return wrapper;
  }

  restore(raw) {
    if (this.disposed) return;
    this.raw = raw;
    try {
      this.state = restoreState(raw);
      this.error = null;
      this.render();
    } catch (error) {
      this.error = error.message;
      this.editor.replaceChildren(this.el("p", "error", this.error));
    }
  }

  render(focus = null) {
    const scroll = this.editor.scrollTop;
    const content = this.el("div");
    const top = this.el("div", "top");
    top.append(this.el("span", "title", "MiniMax H3 prompt"));
    const precision = this.el("select");
    precision.dataset.scope = "precision";
    precision.setAttribute("aria-label", "Timestamp precision");
    for (const digits of [2, 3]) {
      const option = this.el("option", "", `${digits} decimals`);
      option.value = String(digits);
      precision.append(option);
    }
    precision.value = String(this.state.precision);
    top.append(precision);
    content.append(top);

    const headers = this.el("section", "section");
    headers.append(this.el("h3", "section-title", "Prompt headers"));
    headers.append(this.el("p", "hint", "Describe subjects and overall intent. Add only headers you need."));
    this.state.headers.forEach((header, index) => {
      const card = this.el("div", "card");
      const bar = this.el("div", "card-title");
      bar.append(this.el("strong", "", header.name || `Header ${index + 1}`));
      const actions = this.el("div", "actions");
      actions.append(this.button("↑", "header-up", index, `Move header ${index + 1} up`),
        this.button("↓", "header-down", index, `Move header ${index + 1} down`),
        this.button("Remove", "header-remove", index, `Remove header ${index + 1}`));
      bar.append(actions);
      card.append(bar, this.field("Header name", header.name, "header", index, "name"),
        this.field("Text", header.text, "header", index, "text", true));
      headers.append(card);
    });
    const headerTools = this.el("div", "toolbar");
    for (const name of HEADER_CHOICES) {
      const button = this.button(name.replaceAll("_", " "), "add-header", undefined, `Add ${name} header`);
      button.dataset.name = name;
      headerTools.append(button);
    }
    headerTools.append(this.button("Custom header +", "add-header", undefined, "Add a custom header"));
    headers.append(headerTools);
    content.append(headers);

    const timeline = this.el("section", "section");
    timeline.append(this.el("h3", "section-title", "Timeline"));
    if (!this.state.segments.length) timeline.append(this.el("p", "hint", "Add a segment to describe what happens over time."));
    this.state.segments.forEach((segment, index) => {
      const card = this.el("div", "card");
      const bar = this.el("div", "card-title");
      bar.append(this.el("strong", "", `Segment ${index + 1}`));
      const actions = this.el("div", "actions");
      actions.append(this.button(this.collapsed.has(index) ? "Expand" : "Collapse", "segment-collapse", index),
        this.button("Duplicate", "segment-duplicate", index),
        this.button("↑", "segment-up", index, `Move segment ${index + 1} up`),
        this.button("↓", "segment-down", index, `Move segment ${index + 1} down`),
        this.button("Remove", "segment-remove", index, `Remove segment ${index + 1}`));
      bar.append(actions);
      card.append(bar);
      if (!this.collapsed.has(index)) {
        const times = this.el("div", "row time-row");
        times.append(this.field("Start (seconds or MM:SS)", segment.start, "segment", index, "start"),
          this.field("End (seconds or MM:SS)", segment.end, "segment", index, "end"));
        card.append(times);
        for (const name of CHANNELS) {
          const channel = segment.channels[name];
          const row = this.el("div", "channel");
          const label = this.el("label");
          const checkbox = this.el("input");
          checkbox.type = "checkbox";
          checkbox.checked = channel.enabled;
          checkbox.dataset.scope = "channel-toggle";
          checkbox.dataset.index = String(index);
          checkbox.dataset.key = name;
          label.append(checkbox, this.el("strong", "", `[${name.toUpperCase()}]`));
          row.append(label);
          if (channel.enabled) row.append(this.field(`${name} description`, channel.text, "channel", index, name, true));
          card.append(row);
        }
      }
      timeline.append(card);
    });
    const add = this.button("Add segment +", "add-segment");
    add.classList.add("primary");
    timeline.append(add);
    content.append(timeline);

    const previewSection = this.el("section", "section");
    previewSection.append(this.el("h3", "section-title", "Prompt preview"));
    this.message = this.el("div", "message");
    this.preview = this.el("pre", "preview");
    previewSection.append(this.message, this.preview);
    content.append(previewSection);
    this.editor.replaceChildren(content);
    this.editor.scrollTop = scroll;
    this.updatePreview();
    if (focus) {
      const element = [...this.editor.querySelectorAll("[data-scope]")].find(item =>
        item.dataset.scope === focus.scope && item.dataset.index === String(focus.index) && item.dataset.key === focus.key);
      element?.focus({ preventScroll: true });
    }
  }

  updatePreview() {
    try {
      const prompt = compilePrompt(this.state);
      this.preview.textContent = prompt || "Prompt appears here as you write.";
      const warnings = timelineWarnings(this.state);
      this.message.className = warnings.length ? "message warning" : "message muted";
      this.message.textContent = warnings.join(" ") || "Empty fields stay out of output.";
    } catch (error) {
      this.preview.textContent = "";
      this.message.className = "message error";
      this.message.textContent = error.message;
    }
  }

  change(update, focus = null) {
    this.node.graph?.beforeChange(this.node);
    try {
      update();
      if (focus !== false) this.render(focus);
      else this.updatePreview();
    } finally {
      this.node.graph?.afterChange(this.node);
      app.canvas?.setDirty(true, true);
    }
  }

  onInput(event) {
    const { scope, index, key } = event.target.dataset ?? {};
    if (!["header", "segment", "channel"].includes(scope)) return;
    const number = Number(index);
    this.change(() => {
      if (scope === "header") this.state.headers[number][key] = event.target.value;
      else if (scope === "segment") this.state.segments[number][key] = event.target.value;
      else this.state.segments[number].channels[key].text = event.target.value;
    }, false);
  }

  onChange(event) {
    const { scope, index, key } = event.target.dataset ?? {};
    if (scope === "precision") this.change(() => { this.state.precision = Number(event.target.value); });
    if (scope === "channel-toggle") this.change(() => {
      this.state.segments[Number(index)].channels[key].enabled = event.target.checked;
    }, { scope: "channel", index, key });
  }

  onClick(event) {
    const button = event.target.closest?.("button[data-action]");
    if (!button || !this.editor.contains(button)) return;
    event.stopPropagation();
    const action = button.dataset.action;
    const index = Number(button.dataset.index);
    if (action === "add-header") {
      this.change(() => this.state.headers.push({ name: button.dataset.name ?? "", text: "" }),
        { scope: "header", index: this.state.headers.length, key: "name" });
    } else if (action === "add-segment") {
      this.change(() => addSegment(this.state), { scope: "segment", index: this.state.segments.length, key: "start" });
    } else if (action === "segment-collapse") {
      if (this.collapsed.has(index)) this.collapsed.delete(index);
      else this.collapsed.add(index);
      this.render();
    } else if (action.startsWith("header-")) {
      this.change(() => {
        if (action === "header-remove") this.state.headers.splice(index, 1);
        else moveItem(this.state.headers, index, action === "header-up" ? -1 : 1);
      });
    } else if (action.startsWith("segment-")) {
      this.change(() => {
        if (action === "segment-remove") this.state.segments.splice(index, 1);
        else if (action === "segment-duplicate") this.state.segments.splice(index + 1, 0, structuredClone(this.state.segments[index]));
        else moveItem(this.state.segments, index, action === "segment-up" ? -1 : 1);
        this.collapsed.clear();
      });
    }
  }
}

app.registerExtension({
  name: "UtilsCollection.MiniMaxH3PromptBuilder",
  getCustomWidgets() {
    return {
      UC_MINIMAX_H3_PROMPT_BUILDER(node, name, data) {
        const editor = new H3PromptEditor(node, name, data);
        return { widget: editor.widget, minWidth: 410, minHeight: 360 };
      },
    };
  },
});
