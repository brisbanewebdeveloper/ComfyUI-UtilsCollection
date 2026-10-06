export const TABS = ["subjects", "summary", "retention", "detailed", "soundscape", "music"];

export const TAB_LABELS = {
  subjects: "1. Definitions",
  summary: "2. Summary",
  retention: "3. Retention",
  detailed: "4. Timeline",
  soundscape: "5. Soundscape",
  music: "6. Music",
};

export const TASK_TYPES = [
  "reference generation",
  "keyframe completion",
  "video editing",
  "video continuation",
  "audio reuse",
  "audio reference",
];

export const VISIBLE_MARKERS = [
  "attribute_transfer",
  "partially_preserved",
  "fully_preserved",
  "weak_reference",
];

export const AUDIO_MARKERS = [
  "fully_copy",
  "partially_copy",
  "reference",
  "weak_reference",
];

export const CAMERA_MOTIONS = [
  "Push In",
  "Pull Out",
  "Pan Left",
  "Pan Right",
  "Tilt Up",
  "Tilt Down",
  "Tracking Shot",
  "Static Shot",
  "Zoom In",
  "Zoom Out",
  "Truck Left",
  "Truck Right",
  "Arc Shot",
  "POV",
];

export const CHANNELS = ["visual", "speech", "sounds", "music"];

export const H3_FPS = 24;

export function snapH3Frames(frames) {
  const f = Math.max(5, Math.round(frames));
  return f + ((((5 - (f % 17)) % 17) + 17) % 17);
}

export function h3SecondsFromFrames(frames) {
  return snapH3Frames(frames) / H3_FPS;
}

export function h3StepDuration(currentSeconds, stepDelta = 1) {
  const currentFrames = snapH3Frames(Math.max(0.2, Number(currentSeconds) || 2.333) * H3_FPS);
  const nextFrames = Math.max(5, currentFrames + stepDelta * 17);
  return nextFrames / H3_FPS;
}

export function formatSeconds(seconds, precision = 2) {
  const sec = Math.max(0, Number(seconds) || 0);
  const width = precision + 3;
  return `${sec.toFixed(precision).padStart(width, "0")}s`;
}

export function parseSeconds(value) {
  if (typeof value === "number" && Number.isFinite(value)) return Math.max(0, value);
  let str = String(value ?? "").trim();
  if (str.endsWith("s")) str = str.slice(0, -1).trim();
  if (str.includes(":")) {
    const parts = str.split(":");
    return Math.max(0, (Number(parts[0]) || 0) * 60 + (Number(parts[1]) || 0));
  }
  const parsed = parseFloat(str);
  return Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
}

export function createState() {
  return {
    version: 2,
    precision: 2,
    activeTab: "subjects",
    definitions: [], // All tracked definitions: subjects, pictures, videos, audios
    summary: {
      taskTypes: ["reference generation"],
      text: "",
    },
    retention: [],
    detailed: {
      hasTimeline: true,
      continuousText: "",
      segmentDuration: 2.333, // 56 frames
    },
    segments: [],
    overall_soundscape: "",
    non_diegetic_music: "N/A",
  };
}

export function restoreState(raw) {
  let state;
  try {
    state = typeof raw === "string" ? JSON.parse(raw) : raw;
  } catch {
    throw new Error("Saved prompt state is invalid JSON.");
  }
  if (!state || typeof state !== "object") {
    throw new Error("Saved prompt state must be an object.");
  }

  const precision = [2, 3].includes(state.precision) ? state.precision : 2;
  const activeTab = TABS.includes(state.activeTab) ? state.activeTab : "subjects";

  // Support definitions array, or fallback to legacy subjects array
  const rawDefs = Array.isArray(state.definitions) ? state.definitions : (Array.isArray(state.subjects) ? state.subjects : []);
  const definitions = rawDefs.map((d, idx) => {
    const kind = ["subject", "picture", "video", "audio"].includes(d.kind) ? d.kind : (d.type === "audio" ? "audio" : "subject");
    return {
      id: Number(d.id) || idx + 1,
      kind,
      hasRef: typeof d.hasRef === "boolean" ? d.hasRef : (d.refType && d.refType !== "none"),
      refType: ["picture", "video", "audio", "none"].includes(d.refType) ? d.refType : "picture",
      refIndex: Number(d.refIndex) || 1,
      role: String(d.role ?? "custom"),
      targetSubject: Number(d.targetSubject) || 1,
      text: String(d.text ?? ""),
      raw: typeof d.raw === "string" ? d.raw : "",
    };
  });

  const summary = {
    taskTypes: Array.isArray(state.summary?.taskTypes) ? state.summary.taskTypes.filter(t => TASK_TYPES.includes(t)) : ["reference generation"],
    text: String(state.summary?.text ?? ""),
  };

  const retention = Array.isArray(state.retention) ? state.retention.map(r => ({
    label: String(r.label ?? "").trim(),
    marker: String(r.marker ?? "attribute_transfer").trim(),
    text: String(r.text ?? r.descriptor ?? "").trim(),
  })) : [];

  const detailed = {
    hasTimeline: state.detailed?.hasTimeline !== false,
    continuousText: String(state.detailed?.continuousText ?? ""),
    segmentDuration: Math.max(0.2, Number(state.detailed?.segmentDuration) || 2.333),
  };

  const segments = Array.isArray(state.segments) ? state.segments.map((seg, idx) => ({
    start: seg.start !== undefined ? String(seg.start) : formatSeconds(idx * detailed.segmentDuration, precision),
    end: seg.end !== undefined ? String(seg.end) : formatSeconds((idx + 1) * detailed.segmentDuration, precision),
    hasShot: seg.hasShot !== false,
    shot: Number(seg.shot) || idx + 1,
    visual: String(seg.visual ?? (seg.channels?.visual?.text ?? "")),
    speech: {
      enabled: Boolean(seg.speech?.enabled ?? seg.channels?.speech?.enabled),
      speaker: String(seg.speech?.speaker ?? "S1"),
      language: String(seg.speech?.language ?? "English"),
      text: String(seg.speech?.text ?? (seg.channels?.speech?.text ?? "")),
    },
    sounds: {
      enabled: Boolean(seg.sounds?.enabled ?? seg.channels?.sounds?.enabled),
      text: String(seg.sounds?.text ?? (seg.channels?.sounds?.text ?? "")),
    },
    music: {
      enabled: Boolean(seg.music?.enabled ?? seg.channels?.music?.enabled),
      text: String(seg.music?.text ?? (seg.channels?.music?.text ?? "")),
    },
  })) : [];

  return {
    version: 2,
    precision,
    activeTab,
    definitions,
    summary,
    retention,
    detailed,
    segments,
    overall_soundscape: String(state.overall_soundscape ?? ""),
    non_diegetic_music: String(state.non_diegetic_music ?? "N/A"),
  };
}

export function addDefinition(state, kind = "subject", options = {}) {
  const existingIds = state.definitions.filter(d => d.kind === kind).map(d => d.id || 0);
  const nextId = existingIds.length ? Math.max(...existingIds) + 1 : 1;
  const item = {
    id: nextId,
    kind,
    hasRef: Boolean(options.hasRef),
    refType: options.refType || "picture",
    refIndex: options.refIndex || 1,
    role: options.role || "custom",
    targetSubject: options.targetSubject || 1,
    text: options.text || "",
    raw: "",
  };
  state.definitions.push(item);

  // Auto-sync retention entry if appropriate
  let label = `<Subject ${nextId}>`;
  let marker = "attribute_transfer";
  if (kind === "picture") {
    label = `<Picture ${nextId}>`;
    marker = "fully_preserved";
  } else if (kind === "video") {
    label = `<Video ${nextId}>`;
    marker = "partially_preserved";
  } else if (kind === "audio") {
    label = `<Audio ${nextId}>`;
    marker = options.role === "full" ? "fully_copy" : "reference";
  }

  if (item.hasRef || kind !== "subject") {
    if (!state.retention.some(r => r.label === label)) {
      state.retention.push({
        label,
        marker,
        text: defaultRetentionText(label, marker),
      });
    }
  }
  return state.definitions.length - 1;
}

export function removeDefinition(state, index) {
  if (index >= 0 && index < state.definitions.length) {
    const removed = state.definitions.splice(index, 1)[0];
    const prefix = removed.kind === "picture" ? "Picture" : removed.kind === "video" ? "Video" : removed.kind === "audio" ? "Audio" : "Subject";
    const label = `<${prefix} ${removed.id}>`;
    state.retention = state.retention.filter(r => r.label !== label);
  }
}

export function defaultRetentionText(label, marker) {
  switch (marker) {
    case "attribute_transfer":
      return `transfers identity and appearance to ${label} while retaining source motion and camera progression`;
    case "partially_preserved":
      return `identity and appearance are preserved while action, environment, and motion are newly generated`;
    case "fully_preserved":
      return `every defined visible characteristic and role is fully preserved`;
    case "weak_reference":
      return `broad visible similarity in style, category, and composition is followed`;
    case "fully_copy":
      return `complete source signal serves as complete final audio track`;
    case "partially_copy":
      return `selected layers are copied while other audio elements are newly synchronized`;
    case "reference":
      return `timbre, vocal delivery, and cadence are referenced`;
    default:
      return "";
  }
}

export function createSegment(startSec = 0, durationSec = 2.333, precision = 2, shot = 1) {
  const start = formatSeconds(startSec, precision);
  const end = formatSeconds(startSec + durationSec, precision);
  return {
    start,
    end,
    hasShot: true,
    shot,
    visual: "",
    speech: { enabled: false, speaker: "S1", language: "English", text: "" },
    sounds: { enabled: false, text: "" },
    music: { enabled: false, text: "" },
  };
}

export function addSegment(state) {
  const precision = state.precision || 2;
  const duration = Number(state.detailed?.segmentDuration) || 2.333;
  let startSec = 0;
  let shot = 1;

  if (state.segments.length > 0) {
    const last = state.segments[state.segments.length - 1];
    startSec = parseSeconds(last.end);
    shot = (last.shot || 1) + (last.hasShot ? 1 : 0);
  }

  const seg = createSegment(startSec, duration, precision, shot);
  state.segments.push(seg);
  return state.segments.length - 1;
}

export function moveItem(items, index, change) {
  const target = index + change;
  if (target < 0 || target >= items.length) return false;
  [items[index], items[target]] = [items[target], items[index]];
  return true;
}

export function extractTags(state) {
  const subjects = [];
  const pictures = new Set();
  const videos = new Set();
  const audios = new Set();

  for (const d of state.definitions || []) {
    if (d.kind === "subject") {
      subjects.push(`<Subject ${d.id}>`);
      if (d.hasRef) {
        if (d.refType === "picture") pictures.add(`<Picture ${d.refIndex}>`);
        if (d.refType === "video") videos.add(`<Video ${d.refIndex}>`);
      }
    } else if (d.kind === "picture") {
      pictures.add(`<Picture ${d.id}>`);
    } else if (d.kind === "video") {
      videos.add(`<Video ${d.id}>`);
    } else if (d.kind === "audio") {
      audios.add(`<Audio ${d.id}>`);
    }
  }

  return {
    subjects,
    pictures: Array.from(pictures),
    videos: Array.from(videos),
    audios: Array.from(audios),
  };
}

export function compileDefinitionLine(item) {
  if (item.raw && item.raw.trim()) return item.raw.trim();
  const id = item.id || 1;
  const desc = item.text?.trim() || "";

  if (item.kind === "subject") {
    if (item.hasRef) {
      const refTag = `<${item.refType === "video" ? "Video" : "Picture"} ${item.refIndex || 1}>`;
      return `<Subject ${id}> is fully referenced in ${refTag}:${desc ? ` ${desc}` : ""}`;
    }
    if (desc.startsWith("is ") || desc.startsWith("are ")) {
      return `<Subject ${id}> ${desc}`;
    }
    return `<Subject ${id}> is ${desc || "a primary visual subject in the scene."}`;
  }

  if (item.kind === "picture") {
    if (item.role === "first_frame") {
      return `<Picture ${id}> is the fixed first frame anchor at 00.00s.`;
    }
    if (item.role === "final_frame") {
      return `<Picture ${id}> is the fixed final frame anchor at video endpoint.`;
    }
    if (item.role === "storyboard") {
      return `<Picture ${id}> is a storyboard reference for [Shot 1] and [Shot 2], defining viewpoint and subject placement.`;
    }
    return `<Picture ${id}>:${desc ? ` ${desc}` : ""}`;
  }

  if (item.kind === "video") {
    if (item.role === "edit") {
      return `<Video ${id}> is the source video for the target video edit.`;
    }
    if (item.role === "continue") {
      return `<Video ${id}> is the source video for continuation.`;
    }
    if (item.role === "structure") {
      return `<Video ${id}> provides camera movement, cuts, and temporal structure.`;
    }
    return `<Video ${id}>:${desc ? ` ${desc}` : ""}`;
  }

  if (item.kind === "audio") {
    if (item.role === "timbre") {
      const spk = item.targetSubject || 1;
      return `<Audio ${id}> is the voice-timbre reference for <Subject ${spk}> (S${spk}).`;
    }
    if (item.role === "full") {
      return `<Audio ${id}> is reused as the target video's complete final audio track.`;
    }
    if (item.role === "music") {
      return `<Audio ${id}> is the music-style and rhythm reference.`;
    }
    return `<Audio ${id}>:${desc ? ` ${desc}` : ""}`;
  }

  return "";
}

export function compilePrompt(state) {
  if (!state || typeof state !== "object") return "";
  const precision = [2, 3].includes(state.precision) ? state.precision : 2;
  const blocks = [];

  // 1. subject_definitions:
  const subLines = (state.definitions || []).map(compileDefinitionLine).filter(Boolean);
  if (subLines.length) {
    blocks.push(`subject_definitions:\n${subLines.join("\n")}`);
  }

  // 2. summary:
  const summaryText = state.summary?.text?.trim() || "";
  const taskTypes = Array.isArray(state.summary?.taskTypes) ? state.summary.taskTypes : [];
  if (taskTypes.length || summaryText) {
    const ordered = [];
    if (taskTypes.includes("video editing")) ordered.push("video editing");
    for (const t of taskTypes) {
      if (t !== "video editing" && !ordered.includes(t)) ordered.push(t);
    }
    const prefix = ordered.length ? `[${ordered.join(" + ")}]` : "";
    const body = `${prefix} ${summaryText}`.trim();
    if (body) blocks.push(`summary:\n${body}`);
  }

  // 3. retention_analysis:
  const retLines = [];
  for (const item of state.retention || []) {
    const label = item.label?.trim();
    const marker = item.marker?.trim();
    const desc = item.text?.trim();
    if (label && marker) {
      retLines.push(desc ? `${label}: ${marker} - ${desc}` : `${label}: ${marker}`);
    }
  }
  if (retLines.length) {
    blocks.push(`retention_analysis:\n${retLines.join("\n")}`);
  }

  // 4. detailed_description:
  const hasTimeline = state.detailed?.hasTimeline !== false;
  const continuousText = state.detailed?.continuousText?.trim() || "";
  const segments = state.segments || [];

  if (hasTimeline && segments.length) {
    const timelineBlocks = [];
    segments.forEach((seg, index) => {
      const startStr = formatSeconds(parseSeconds(seg.start), precision);
      const endStr = formatSeconds(parseSeconds(seg.end), precision);
      const lines = [];

      const visual = seg.visual?.trim() || "";
      const hasShot = seg.hasShot !== false;
      const shotNum = seg.shot || index + 1;
      if (visual || hasShot) {
        const shotPrefix = (hasShot && !visual.startsWith(`[Shot ${shotNum}]`)) ? `[Shot ${shotNum}] ` : "";
        lines.push(`[VISUAL]: ${shotPrefix}${visual}`.trim());
      }

      if (seg.speech?.enabled) {
        const spk = seg.speech.speaker?.trim() || "S1";
        const lang = seg.speech.language?.trim() || "English";
        const txt = seg.speech.text?.trim() || "";
        const spkPrefix = `(${spk}) `;
        const dBody = txt ? `<d>[${lang}] ${txt}</d>` : "";
        lines.push(`[SPEECH]: ${spkPrefix}${dBody}`.trim());
      }

      if (seg.sounds?.enabled && seg.sounds.text?.trim()) {
        lines.push(`[SOUNDS]: ${seg.sounds.text.trim()}`);
      }

      if (seg.music?.enabled && seg.music.text?.trim()) {
        lines.push(`[MUSIC]: ${seg.music.text.trim()}`);
      }

      if (lines.length) {
        timelineBlocks.push(`[${startStr}-${endStr}]:\n${lines.join("\n")}`);
      }
    });

    if (timelineBlocks.length) {
      blocks.push(`detailed_description:\nTimeline:\n${timelineBlocks.join("\n\n")}`);
    }
  } else if (continuousText) {
    blocks.push(`detailed_description:\n${continuousText}`);
  }

  // 5. overall_soundscape:
  const soundscape = state.overall_soundscape?.trim();
  if (soundscape) {
    blocks.append ? blocks.append(`overall_soundscape:\n${soundscape}`) : blocks.push(`overall_soundscape:\n${soundscape}`);
  }

  // 6. non_diegetic_music:
  const music = state.non_diegetic_music?.trim();
  if (music && blocks.length) {
    blocks.push(`non_diegetic_music:\n${music}`);
  } else if (music && music !== "N/A") {
    blocks.push(`non_diegetic_music:\n${music}`);
  }

  return blocks.join("\n\n");
}

export function timelineWarnings(state) {
  const warnings = [];
  if (state.detailed?.hasTimeline === false) return warnings;
  let previousEnd = null;
  (state.segments || []).forEach((seg, index) => {
    try {
      const start = parseSeconds(seg.start);
      const end = parseSeconds(seg.end);
      if (end <= start) {
        warnings.push(`Segment ${index + 1}: end (${seg.end}) must be after start (${seg.start}).`);
      }
      if (previousEnd !== null && Math.abs(start - previousEnd) > 0.05) {
        warnings.push(`Segment ${index + 1} does not touch previous segment (${formatSeconds(previousEnd, state.precision)} vs ${formatSeconds(start, state.precision)}).`);
      }
      previousEnd = end;
    } catch {
      // ignore
    }
  });
  return warnings;
}
