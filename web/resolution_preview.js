import { app } from "../../scripts/app.js";
import { h3VideoLengthFromSeconds, h3ReferenceFrameRange } from "./h3_video_length.js";
import {
  clampResolutionPreviewSize,
  resolutionPreviewMinimumSize,
} from "./resolution_preview_layout.js";

const VIDEO_MIDDLE_BAND_RESOLUTIONS = {
  "7:3": [[896, 384], [1120, 480], [1280, 544]],
  "16:9": [[768, 416], [864, 480], [1024, 576], [1152, 672]],
  "16:10": [[768, 480], [1024, 640]],
  "4:3": [[640, 480], [768, 576], [864, 672], [1024, 768]],
};

function gcd(left, right) {
  while (right) [left, right] = [right, left % right];
  return left;
}

function lcm(left, right) {
  return (left * right) / gcd(left, right);
}

function ratioFromValue(value) {
  const match = String(value).match(/^(\d+):(\d+)/);
  return match ? [Number(match[1]), Number(match[2])] : null;
}

function widgetValue(node, name) {
  return node.widgets?.find((widget) => widget.name === name)?.value;
}

function widgetIsLinked(node, name) {
  return node.inputs?.some((input) => input.widget?.name === name && input.link != null) ?? false;
}

function compareKeys(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return left[index] - right[index];
  }
  return 0;
}

function regularResolution(ratioWidth, ratioHeight, megapixels, multiple, minimum) {
  const scale = Math.sqrt((megapixels * 1024 * 1024) / (ratioWidth * ratioHeight));
  let width = Math.round((ratioWidth * scale) / multiple) * multiple;
  let height = Math.round((ratioHeight * scale) / multiple) * multiple;
  if (width < minimum || height < minimum) {
    const widthStep = multiple / gcd(ratioWidth, multiple);
    const heightStep = multiple / gcd(ratioHeight, multiple);
    const ratioStep = lcm(widthStep, heightStep);
    const minimumRatio = Math.ceil(Math.max(minimum / ratioWidth, minimum / ratioHeight));
    const ratioScale = Math.ceil(minimumRatio / ratioStep) * ratioStep;
    width = ratioWidth * ratioScale;
    height = ratioHeight * ratioScale;
  }
  return [width, height];
}

function videoResolution(ratioWidth, ratioHeight, megapixels, multiple, minimum) {
  const divisor = gcd(ratioWidth, ratioHeight);
  ratioWidth /= divisor;
  ratioHeight /= divisor;
  const landscape = ratioWidth >= ratioHeight;
  const landscapeRatio = landscape
    ? [ratioWidth, ratioHeight]
    : [ratioHeight, ratioWidth];
  const targetPixels = megapixels * 1024 * 1024;
  const middleBand = VIDEO_MIDDLE_BAND_RESOLUTIONS[landscapeRatio.join(":")];
  if (megapixels >= 0.3 && megapixels <= 0.8 && middleBand) {
    const candidates = middleBand
      .map(([width, height]) => (landscape ? [width, height] : [height, width]))
      .filter(([width, height]) => (
        width % multiple === 0
        && height % multiple === 0
        && width >= minimum
        && height >= minimum
      ));
    if (candidates.length) {
      return candidates.reduce((best, candidate) => (
        Math.abs(candidate[0] * candidate[1] - targetPixels)
          < Math.abs(best[0] * best[1] - targetPixels)
          ? candidate
          : best
      ));
    }
  }

  const anchorRatio = landscape ? ratioWidth : ratioHeight;
  const companionRatio = landscape ? ratioHeight : ratioWidth;
  const anchorStep = lcm(multiple, anchorRatio);
  const minimumAnchor = Math.ceil(minimum / anchorStep) * anchorStep;
  const maximum = 8192;
  let best;
  let bestKey;
  for (let anchor = minimumAnchor; anchor <= maximum; anchor += anchorStep) {
    const idealCompanion = (anchor * companionRatio) / anchorRatio;
    const companions = new Set([
      Math.floor(idealCompanion / multiple) * multiple,
      Math.ceil(idealCompanion / multiple) * multiple,
    ]);
    for (const companion of companions) {
      if (companion < minimum || companion > maximum) continue;
      const [width, height] = landscape ? [anchor, companion] : [companion, anchor];
      const megapixelError = Math.abs(width * height - targetPixels) / targetPixels;
      const ratioError = Math.abs((width / height) / (ratioWidth / ratioHeight) - 1);
      const key = [megapixelError, ratioError, width, height];
      if (!bestKey || compareKeys(key, bestKey) < 0) {
        best = [width, height];
        bestKey = key;
      }
    }
  }
  return best;
}

const PREVIEW_FONT = "12px sans-serif";
const previewMeasureContext = document.createElement("canvas").getContext("2d");
previewMeasureContext.font = PREVIEW_FONT;

function fitPreview(node) {
  const minimum = node.computeSize();
  if (node.size[0] < minimum[0]) node.setSize([minimum[0], node.size[1]]);
  node.setDirtyCanvas(true, true);
}

function connectedOrigin(node, inputName) {
  const slot = (node.inputs || []).findIndex((input) => (
    input.name === inputName || input.name?.endsWith(`.${inputName}`) || input.label === inputName
  ));
  const linkId = slot >= 0 ? node.inputs?.[slot]?.link : null;
  const link = linkId != null ? node.graph?.links?.[linkId] : null;
  return link ? { link, node: node.graph?._nodes_by_id?.[link.origin_id] } : null;
}

function resolveConnectedImageDimensions(node, inputName = "image") {
  const origin = connectedOrigin(node, inputName);
  if (!origin?.node || [2, 4].includes(origin.node.mode)) return null;
  const originNode = origin.node;

  // 1. Direct preview element on the origin node (e.g. LoadImage, LoadImageWithAlpha)
  const candidate = originNode.imgs?.[0];
  if (candidate?.complete && (candidate.naturalWidth || candidate.width)) {
    return [candidate.naturalWidth || candidate.width, candidate.naturalHeight || candidate.height];
  }
  if (candidate && !candidate.complete) {
    candidate.addEventListener("load", () => {
      updatePreview(node);
    }, { once: true });
  }

  // 2. Latest execution outputs from server
  const descriptor = app.nodeOutputs?.[String(origin.link.origin_id)]?.images?.[0];
  if (descriptor?.width && descriptor?.height) {
    return [descriptor.width, descriptor.height];
  }

  // 3. Fallback: check if origin node has explicit width/height widgets
  const w = originNode.widgets?.find((w) => w.name === "width")?.value;
  const h = originNode.widgets?.find((w) => w.name === "height")?.value;
  if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
    return [Number(w), Number(h)];
  }

  return null;
}

function updatePreview(node, backendValue) {
  if (node.__ucH3ReferenceVideo) {
    const range = backendValue ?? h3ReferenceFrameRange(
      widgetIsLinked(node, "start_at_timestamp") ? null : Number(widgetValue(node, "start_at_timestamp")),
      widgetIsLinked(node, "duration_seconds") ? null : Number(widgetValue(node, "duration_seconds")),
      node.__ucH3SourceSeconds ?? null,
      widgetIsLinked(node, "segment_count") ? null : Number(widgetValue(node, "segment_count") ?? 0),
      widgetIsLinked(node, "segment_index") ? null : Number(widgetValue(node, "segment_index") ?? 0),
    );
    node.__ucResolutionPreview = `start frame ${range.start ?? "…"} · end frame ${range.end ?? "…"} · ${range.length ?? "…"} frames${range.padding ? ` (${range.padding} padded)` : ""}`;
    fitPreview(node);
    return;
  }
  if (backendValue !== undefined) {
    node.__ucResolutionPreview = String(Array.isArray(backendValue) ? backendValue[0] : backendValue);
  } else {
    let ratio = ratioFromValue(widgetValue(node, "aspect_ratio"));
    if (node.__ucVideoLengthPicker) {
      const cropMethod = widgetValue(node, "crop_method");
      if (cropMethod !== "center") {
        const connectedDims = resolveConnectedImageDimensions(node, "image") || resolveConnectedImageDimensions(node, "video");
        if (connectedDims && connectedDims[0] > 0 && connectedDims[1] > 0) {
          ratio = [connectedDims[0], connectedDims[1]];
        }
      }
    }
    const megapixels = Number(widgetValue(node, "megapixels"));
    const multiple = Number(widgetValue(node, "multiple"));
    const minimum = Number(widgetValue(node, "minimum")) || 256;
    if (!ratio || !megapixels || !multiple) return;
    const [width, height] = node.__ucVideoResolutionSelector
      ? videoResolution(...ratio, megapixels, multiple, minimum)
      : regularResolution(...ratio, megapixels, multiple, minimum);
    const length = (node.__ucVideoResolutionSelector || node.__ucVideoLengthPicker) && !widgetIsLinked(node, "duration_seconds")
      ? h3VideoLengthFromSeconds(Number(widgetValue(node, "duration_seconds")))
      : null;
    const duration = node.__ucVideoLengthPicker && !widgetIsLinked(node, "duration_seconds")
      ? Number(widgetValue(node, "duration_seconds"))
      : null;
    node.__ucResolutionPreview = length === null
      ? `${width}×${height}`
      : duration !== null
        ? `${width}×${height} · ${length} frames · ${duration.toFixed(2)} s`
        : `${width}×${height} · ${length} frames`;
  }
  fitPreview(node);
}

app.registerExtension({
  name: "ComfyUI.UtilsCollection.ResolutionPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["UC_ResolutionSelectorExtended", "UC_VideoResolutionSelector", "UC_VideoResolutionAndLengthPicker", "UC_MiniMaxH3RefVid"].includes(nodeData.name)) return;

    const computeSize = nodeType.prototype.computeSize;
    nodeType.prototype.computeSize = function (out) {
      const baseSize = computeSize?.call(this, out ? [...out] : undefined) || [...(out || this.size || [0, 0])];
      const textWidth = previewMeasureContext.measureText(this.__ucResolutionPreview || "").width;
      return resolutionPreviewMinimumSize(baseSize, textWidth);
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      clampResolutionPreviewSize(size, this.computeSize(), this.flags?.collapsed);
      return onResize?.apply(this, arguments);
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.__ucVideoResolutionSelector = nodeData.name === "UC_VideoResolutionSelector";
      this.__ucVideoLengthPicker = nodeData.name === "UC_VideoResolutionAndLengthPicker";
      this.__ucH3ReferenceVideo = nodeData.name === "UC_MiniMaxH3RefVid";
      this.__ucResolutionPreview = "";
      if (this.__ucVideoLengthPicker) {
        const durationWidget = this.widgets?.find((w) => w.name === "duration_seconds");
        if (durationWidget) {
          const originalCallback = durationWidget.callback;
          durationWidget.callback = function (value) {
            const k = Math.max(0, Math.round((Number(value) - 5 / 24) / (17 / 24)));
            const snappedFrames = 5 + 17 * k;
            const snappedSeconds = Number((snappedFrames / 24).toFixed(5));
            durationWidget.value = snappedSeconds;
            originalCallback?.call(this, snappedSeconds);
          };
        }
      }
      const minimum = this.computeSize();
      this.setSize([
        Math.max(this.size[0], minimum[0]),
        Math.max(this.size[1], minimum[1]),
      ]);
      updatePreview(this);
      return result;
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      onDrawForeground?.apply(this, arguments);
      if (this.flags.collapsed || !this.__ucResolutionPreview) return;
      ctx.save();
      ctx.fillStyle = "#bbb";
      ctx.font = PREVIEW_FONT;
      ctx.textAlign = "center";
      ctx.fillText(this.__ucResolutionPreview, this.size[0] / 2, this.size[1] - 9);
      ctx.restore();
    };

    const onWidgetChanged = nodeType.prototype.onWidgetChanged;
    nodeType.prototype.onWidgetChanged = function (name) {
      const result = onWidgetChanged?.apply(this, arguments);
      if (["aspect_ratio", "crop_method", "megapixels", "multiple", "minimum", "duration_seconds", "start_at_timestamp", "segment_count", "segment_index"].includes(name)) updatePreview(this);
      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      if (this.__ucH3ReferenceVideo) {
        const range = message?.h3_reference_range?.[0];
        if (range) {
          this.__ucH3SourceSeconds = range.source_seconds;
          updatePreview(this, { start: range.start_frame, end: range.source_end_frame ?? range.start_frame + range.length - 1, length: range.length, padding: range.padded_frames ?? 0 });
        }
        return;
      }
      updatePreview(this, message?.resolution);
    };

    if (["UC_MiniMaxH3RefVid", "UC_VideoResolutionAndLengthPicker"].includes(nodeData.name)) {
      const onConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function (type, slot) {
        const result = onConnectionsChange?.apply(this, arguments);
        if (type === 1) {
          if (this.inputs?.[slot]?.name === "video") this.__ucH3SourceSeconds = null;
          updatePreview(this);
          const origin = connectedOrigin(this, this.inputs?.[slot]?.name);
          if (origin?.node) {
            const originNode = origin.node;
            const origOnWidget = originNode.onWidgetChanged;
            const self = this;
            originNode.onWidgetChanged = function () {
              const res = origOnWidget?.apply(this, arguments);
              updatePreview(self);
              return res;
            };
          }
        }
        return result;
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        updatePreview(this);
        return result;
      };
    }
  },
});
