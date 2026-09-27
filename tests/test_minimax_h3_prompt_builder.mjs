import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  CHANNELS, HEADER_CHOICES, addSegment, compilePrompt, createState, formatTime, moveItem,
  restoreState, timelineWarnings,
} from "../web/minimax_h3_prompt_model.js";

test("headers and enabled channels compile into H3 text", () => {
  const state = createState();
  state.headers.push({ name: "summary", text: "A chase." });
  state.headers.unshift({ name: "non_diegetic_music", text: "Low strings." });
  state.headers.push({ name: "overall_soundscape", text: "Traffic." });
  addSegment(state);
  state.segments[0].channels.visual.text = "Camera tracks runner.";
  state.segments[0].channels.music.text = "Saved draft.";
  assert.equal(compilePrompt(state), "summary:\nA chase.\n\ndetailed_description:\nTimeline:\n[00:00.00-00:05.00]:\n[VISUAL]: Camera tracks runner.\n\noverall_soundscape:\nTraffic.\n\nnon_diegetic_music:\nLow strings.");
  state.segments[0].channels.music.enabled = true;
  assert.match(compilePrompt(state), /\[MUSIC\]: Saved draft\.[\s\S]*non_diegetic_music:\nLow strings\.$/);
  assert.equal(restoreState(JSON.stringify(state)).segments[0].channels.music.text, "Saved draft.");
});

test("new segments continue from previous end; gaps warn without blocking", () => {
  const state = createState();
  addSegment(state);
  addSegment(state);
  assert.deepEqual(state.segments.map(segment => [segment.start, segment.end]), [["0", "5"], ["00:05.00", "00:10.00"]]);
  state.segments.forEach(segment => { segment.channels.visual.text = "Motion."; });
  state.segments[1].start = "7";
  assert.deepEqual(timelineWarnings(state), ["Segment 2 starts after previous segment."]);
  assert.match(compilePrompt(state), /\[00:07\.00-00:10\.00\]/);
  moveItem(state.segments, 1, -1);
  assert.equal(state.segments[0].start, "7");
  assert.equal(formatTime(7125, 3), "00:07.125");
});

test("canvas editor keeps blank space draggable, scrolls, and overlays multiline text", () => {
  const source = readFileSync(new URL("../web/minimax_h3_prompt.js", import.meta.url), "utf8");
  const canvasSource = source.slice(source.indexOf("class H3CanvasPromptEditor"));
  let wheel;
  let widget;
  let singleLinePrompt;
  const appended = [];
  const created = [];
  const document = {
    body: {
      append(element) { appended.push(element); },
    },
    createElement(tagName) {
      created.push(tagName);
      const listeners = new Map();
      return {
        tagName: tagName.toUpperCase(), style: {}, dataset: {}, value: "",
        scrollHeight: 120, clientHeight: 84,
        addEventListener(type, listener) {
          listeners.set(type, [...(listeners.get(type) ?? []), listener]);
        },
        emit(type, event = {}) {
          for (const listener of listeners.get(type) ?? []) {
            listener({
              key: "", ctrlKey: false, metaKey: false,
              preventDefault() {}, stopPropagation() {}, ...event,
            });
          }
        },
        focus() { this.focused = true; },
        remove() { this.removed = true; },
      };
    },
  };
  const node = {
    pos: [0, 0], size: [410, 440],
    graph: { beforeChange() {}, afterChange() {} },
    addCustomWidget(value) { widget = value; return value; },
    getWidgetOnPos() { return widget; },
    setSize(size) { this.size = size; },
  };
  const canvas = {
    canvas: {
      addEventListener(type, listener) { if (type === "wheel") wheel = listener; },
      getBoundingClientRect() { return { left: 0, top: 0 }; },
    },
    ds: { scale: 1, offset: [0, 0] },
    graph: { getNodeOnPos() { return node; } },
    setDirty() {},
    prompt(...args) { singleLinePrompt = args; },
  };
  const app = { canvas, registerExtension() {} };
  const Editor = new Function(
    "app", "CHANNELS", "HEADER_CHOICES", "addSegment", "compilePrompt", "createState",
    "moveItem", "restoreState", "timelineWarnings", "requestAnimationFrame", "document",
    canvasSource + "\nreturn H3CanvasPromptEditor;",
  )(app, CHANNELS, HEADER_CHOICES, addSegment, compilePrompt, createState,
    moveItem, restoreState, timelineWarnings, callback => callback(), document);
  const state = createState();
  state.headers.push({ name: "summary", text: "Before." });
  for (let i = 0; i < 4; i++) {
    addSegment(state);
    state.segments[i].channels.visual.text = "Movement.";
  }
  const editor = new Editor(node, "prompt_state", [null, { default: JSON.stringify(state) }]);
  const ctx = {
    beginPath() {}, roundRect() {}, fill() {}, stroke() {}, rect() {}, clip() {},
    save() {}, restore() {}, fillText() {},
    measureText(value) { return { width: value.length * 7 }; },
  };
  widget.draw(ctx, node, 410, 60);
  assert.equal(node.getWidgetOnPos(200, 83), undefined);
  assert.equal(node.getWidgetOnPos(320, 80), widget);
  widget.mouse({ button: 0, type: "pointerup" }, [320, 80]);
  assert.equal(JSON.parse(widget.serializeValue()).precision, 3);
  widget.mouse({ button: 0, type: "pointerup" }, [30, 220]);
  assert.equal(singleLinePrompt[0], "Header name");
  singleLinePrompt[2]("renamed_summary");
  assert.equal(editor.state.headers[0].name, "renamed_summary");
  widget.mouse({ button: 0, type: "pointerup" }, [30, 250]);
  const textEditor = editor.textEditor.element;
  assert.equal(textEditor.tagName, "TEXTAREA");
  assert.equal(textEditor.className, "comfy-multiline-input");
  assert.equal(appended.length, 1);
  assert.deepEqual(created, ["textarea"]);
  textEditor.value = "Updated.";
  textEditor.emit("input");
  assert.equal(editor.state.headers[0].text, "Updated.");
  textEditor.emit("blur");
  assert.equal(editor.textEditor, null);
  assert.equal(textEditor.removed, true);
  assert.ok(editor.contentHeight > editor.viewportHeight);
  let consumed = false;
  wheel({
    clientX: 200, clientY: 130, deltaY: 80, deltaMode: 0,
    ctrlKey: false, metaKey: false,
    preventDefault() { consumed = true; },
    stopImmediatePropagation() {},
  });
  assert.ok(editor.scroll > 0);
  assert.equal(consumed, true);
  assert.equal(JSON.parse(widget.serializeValue()).segments.length, 4);
});
