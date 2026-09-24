import test from "node:test";
import assert from "node:assert/strict";
import {
  addSegment, compilePrompt, createState, formatTime, moveItem,
  restoreState, timelineWarnings,
} from "../web/minimax_h3_prompt_model.js";

test("headers and enabled channels compile into H3 text", () => {
  const state = createState();
  state.headers.push({ name: "summary", text: "A chase." });
  addSegment(state);
  state.segments[0].channels.visual.text = "Camera tracks runner.";
  state.segments[0].channels.music.text = "Saved draft.";
  assert.equal(compilePrompt(state), "summary:\nA chase.\n\nTimeline:\n[00:00.00-00:05.00]:\n[VISUAL]: Camera tracks runner.");
  state.segments[0].channels.music.enabled = true;
  assert.match(compilePrompt(state), /\[MUSIC\]: Saved draft\.$/);
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
