export const BASE_TASKS = ["T2VA", "I2VA", "FL2VA", "L2VA"];

export const BASE_TABS = ["task", "description", "soundscape", "music"];

export const BASE_TAB_LABELS = {
  task: "1. Task & Align",
  description: "2. Description",
  soundscape: "3. Soundscape",
  music: "4. Music",
};

export const VISUAL_STYLES = [
  "Live-action, cinematic",
  "2D-animated",
  "3D CG",
  "Claymation",
  "Watercolor",
  "Vintage film",
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

export function createBaseState() {
  return {
    version: 1,
    task: "T2VA",
    duration: 5.0,
    finalShot: 1,
    autoFinalShot: true,
    precision: 2,
    activeTab: "task",
    description: {
      mode: "timeline",
      continuousText: "",
      segmentDuration: 2.333,
    },
    segments: [],
    overall_soundscape: "",
    non_diegetic_music: "N/A",
  };
}

export function restoreBaseState(raw) {
  let state;
  try {
    state = typeof raw === "string" ? JSON.parse(raw) : raw;
  } catch {
    throw new Error("Saved base prompt state is invalid JSON.");
  }
  if (!state || typeof state !== "object") {
    throw new Error("Saved base prompt state must be an object.");
  }

  const task = BASE_TASKS.includes(state.task) ? state.task : "T2VA";
  const precision = [2, 3].includes(state.precision) ? state.precision : 2;
  const activeTab = BASE_TABS.includes(state.activeTab) ? state.activeTab : "task";
  const duration = Math.max(0.2, Number(state.duration) || 5.0);
  const finalShot = Math.max(1, Number(state.finalShot) || 1);
  const autoFinalShot = state.autoFinalShot !== false;

  const descData = state.description || {};
  const description = {
    mode: ["timeline", "continuous"].includes(descData.mode) ? descData.mode : "timeline",
    continuousText: String(descData.continuousText ?? ""),
    segmentDuration: Math.max(0.2, Number(descData.segmentDuration) || 2.333),
  };

  const segments = Array.isArray(state.segments) ? state.segments.map((seg, idx) => ({
    start: seg.start !== undefined ? String(seg.start) : formatSeconds(idx * description.segmentDuration, precision),
    end: seg.end !== undefined ? String(seg.end) : formatSeconds((idx + 1) * description.segmentDuration, precision),
    hasShot: seg.hasShot !== false,
    shot: Number(seg.shot) || idx + 1,
    visual: String(seg.visual ?? ""),
    speech: {
      enabled: Boolean(seg.speech?.enabled),
      speaker: String(seg.speech?.speaker ?? "S1"),
      language: String(seg.speech?.language ?? "English"),
      text: String(seg.speech?.text ?? ""),
    },
    sounds: {
      enabled: Boolean(seg.sounds?.enabled),
      text: String(seg.sounds?.text ?? ""),
    },
    music: {
      enabled: Boolean(seg.music?.enabled),
      text: String(seg.music?.text ?? ""),
    },
  })) : [];

  return {
    version: 1,
    task,
    duration,
    finalShot,
    autoFinalShot,
    precision,
    activeTab,
    description,
    segments,
    overall_soundscape: String(state.overall_soundscape ?? ""),
    non_diegetic_music: String(state.non_diegetic_music ?? "N/A"),
  };
}

export function getFinalShot(state) {
  if (!state.autoFinalShot) return Math.max(1, Number(state.finalShot) || 1);
  let maxShot = 1;
  const segments = state.segments || [];
  for (let idx = 0; idx < segments.length; idx++) {
    const seg = segments[idx];
    if (seg.shot != null) {
      maxShot = Math.max(maxShot, Number(seg.shot) || 1);
    } else if (seg.hasShot) {
      maxShot = Math.max(maxShot, idx + 1);
    }
  }
  return maxShot;
}

export function createBaseSegment(startSec, durationSec, precision = 2, shot = 1) {
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

export function addBaseSegment(state) {
  const precision = state.precision || 2;
  const duration = Number(state.description?.segmentDuration) || 2.333;
  let startSec = 0;
  let shot = 1;

  if (state.segments.length > 0) {
    const last = state.segments[state.segments.length - 1];
    startSec = parseSeconds(last.end);
    shot = (Number(last.shot) || 1) + (last.hasShot ? 1 : 0);
  }

  const seg = createBaseSegment(startSec, duration, precision, shot);
  state.segments.push(seg);
  return seg;
}

export function compileBasePrompt(state) {
  if (!state || typeof state !== "object") return "";
  const precision = [2, 3].includes(state.precision) ? state.precision : 2;
  const task = String(state.task || "T2VA").toUpperCase();
  const duration = Math.max(0.2, Number(state.duration) || 5.0);
  const finalShot = getFinalShot(state);
  const durStr = duration.toFixed(2);

  let instruction = "";
  if (task === "I2VA") {
    instruction = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.";
  } else if (task === "FL2VA") {
    instruction = `How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot ${finalShot}) aligns with the ${durStr}-second mark of the target video.`;
  } else if (task === "L2VA") {
    instruction = `How the reference pictures align with the target video — <Picture 1> (from [Shot ${finalShot}]) aligns with the ${durStr}-second mark of the target video.`;
  }

  const blocks = [];

  // Field 1: integrated_multimodal_description:
  const isTimeline = state.description?.mode !== "continuous";
  const continuousText = state.description?.continuousText?.trim() || "";
  const segments = state.segments || [];

  if (isTimeline && segments.length) {
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
      blocks.push(`integrated_multimodal_description:\nTimeline:\n${timelineBlocks.join("\n\n")}`);
    }
  } else if (continuousText) {
    blocks.push(`integrated_multimodal_description:\n${continuousText}`);
  }

  // Field 2: overall_soundscape:
  const soundscape = state.overall_soundscape?.trim() || "";
  if (soundscape) {
    blocks.push(`overall_soundscape:\n${soundscape}`);
  }

  // Field 3: non_diegetic_music:
  const music = state.non_diegetic_music?.trim() || "";
  if (music && blocks.length) {
    blocks.push(`non_diegetic_music:\n${music}`);
  } else if (music && music !== "N/A") {
    blocks.push(`non_diegetic_music:\n${music}`);
  }

  if (!blocks.length && !instruction) return "";

  const parts = [];
  if (instruction) parts.push(instruction);
  if (blocks.length) parts.push(blocks.join("\n\n"));
  return parts.join("\n\n");
}

export function baseTimelineWarnings(state) {
  const warnings = [];
  const segments = state.segments || [];
  for (let idx = 1; idx < segments.length; idx++) {
    const prev = segments[idx - 1];
    const curr = segments[idx];
    const prevEnd = parseSeconds(prev.end);
    const currStart = parseSeconds(curr.start);
    if (Math.abs(prevEnd - currStart) > 0.01) {
      if (currStart > prevEnd) {
        warnings.push(`Segment ${idx + 1} starts after previous segment ends.`);
      } else {
        warnings.push(`Segment ${idx + 1} overlaps with previous segment.`);
      }
    }
  }
  return warnings;
}
