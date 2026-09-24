export const CHANNELS = ["visual", "speech", "sounds", "music"];
export const HEADER_CHOICES = ["subject_definitions", "summary", "retention_analysis", "overall_soundscape", "non_diegetic_music"];

export function createState() {
  return { version: 1, precision: 2, headers: [], segments: [] };
}

export function createSegment(start = "0", end = "5") {
  return {
    start, end,
    channels: Object.fromEntries(CHANNELS.map(name => [name, { enabled: name === "visual", text: "" }])),
  };
}

export function restoreState(raw) {
  let state;
  try { state = JSON.parse(raw); }
  catch { throw new Error("Saved prompt state is invalid JSON."); }
  if (!state || state.version !== 1 || ![2, 3].includes(state.precision)
      || !Array.isArray(state.headers) || !Array.isArray(state.segments)) {
    throw new Error("Saved prompt state has an unsupported format.");
  }
  for (const header of state.headers) {
    if (!header || typeof header.name !== "string" || typeof header.text !== "string") {
      throw new Error("Saved header has an invalid name or text.");
    }
  }
  for (const segment of state.segments) {
    if (!segment || typeof segment.start !== "string" || typeof segment.end !== "string" || !segment.channels) {
      throw new Error("Saved segment has invalid times or channels.");
    }
    for (const name of CHANNELS) {
      const channel = segment.channels[name];
      if (!channel || typeof channel.enabled !== "boolean" || typeof channel.text !== "string") {
        throw new Error(`Saved segment has an invalid ${name} field.`);
      }
    }
  }
  return state;
}

export function parseTime(value, precision, label = "Time") {
  const match = /^(?:(\d+):)?(\d+)(?:\.(\d{1,3}))?$/.exec(value.trim());
  if (!match) throw new Error(`${label}: enter seconds or MM:SS.`);
  const [, minutes, seconds, fraction = ""] = match;
  if (minutes !== undefined && Number(seconds) >= 60) throw new Error(`${label}: seconds must be below 60 in MM:SS.`);
  if (fraction.length > precision) throw new Error(`${label}: select ${fraction.length}-decimal precision or shorten the time.`);
  const total = (Number(minutes ?? 0) * 60 + Number(seconds)) * 1000 + Number(fraction.padEnd(3, "0"));
  if (!Number.isSafeInteger(total)) throw new Error(`${label}: time is too large.`);
  return total;
}

export function formatTime(milliseconds, precision) {
  const minutes = Math.floor(milliseconds / 60000);
  const seconds = Math.floor(milliseconds % 60000 / 1000);
  const fraction = String(milliseconds % 1000).padStart(3, "0").slice(0, precision);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${fraction}`;
}

export function addSegment(state) {
  const previous = state.segments.at(-1);
  if (!previous) state.segments.push(createSegment());
  else {
    let start = "", end = "";
    try {
      const milliseconds = parseTime(previous.end, state.precision, "Previous end");
      start = formatTime(milliseconds, state.precision);
      end = formatTime(milliseconds + 5000, state.precision);
    } catch { /* Leave times empty until previous end is corrected. */ }
    state.segments.push(createSegment(start, end));
  }
  return state.segments.length - 1;
}

export function moveItem(items, index, change) {
  const target = index + change;
  if (target < 0 || target >= items.length) return false;
  [items[index], items[target]] = [items[target], items[index]];
  return true;
}

export function compilePrompt(state) {
  if (!state || state.version !== 1 || ![2, 3].includes(state.precision)
      || !Array.isArray(state.headers) || !Array.isArray(state.segments)) {
    throw new Error("Saved prompt state has an unsupported format.");
  }
  const blocks = [];
  state.headers.forEach((header, index) => {
    const body = header.text.trim();
    if (!body) return;
    const name = header.name.trim().replace(/:$/, "");
    if (!name || /[\r\n:]/.test(name)) throw new Error(`Header ${index + 1}: enter one label without a colon.`);
    blocks.push(`${name}:\n${body}`);
  });
  const timeline = [];
  state.segments.forEach((segment, index) => {
    const lines = CHANNELS.filter(name => segment.channels[name].enabled && segment.channels[name].text.trim())
      .map(name => `[${name.toUpperCase()}]: ${segment.channels[name].text.trim()}`);
    if (!lines.length) return;
    const start = parseTime(segment.start, state.precision, `Segment ${index + 1} start`);
    const end = parseTime(segment.end, state.precision, `Segment ${index + 1} end`);
    if (end <= start) throw new Error(`Segment ${index + 1}: end must be later than start.`);
    timeline.push(`[${formatTime(start, state.precision)}-${formatTime(end, state.precision)}]:\n${lines.join("\n")}`);
  });
  if (timeline.length) blocks.push(`Timeline:\n${timeline.join("\n\n")}`);
  return blocks.join("\n\n");
}

export function timelineWarnings(state) {
  const warnings = [];
  let previousEnd = null;
  state.segments.forEach((segment, index) => {
    if (!CHANNELS.some(name => segment.channels[name].enabled && segment.channels[name].text.trim())) return;
    try {
      const start = parseTime(segment.start, state.precision);
      const end = parseTime(segment.end, state.precision);
      if (end <= start) return;
      if (previousEnd !== null && start !== previousEnd) {
        warnings.push(`Segment ${index + 1} ${start > previousEnd ? "starts after" : "overlaps"} previous segment.`);
      }
      previousEnd = end;
    } catch { /* The preview reports invalid times. */ }
  });
  return warnings;
}
