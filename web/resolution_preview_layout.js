export const RESOLUTION_PREVIEW_HEIGHT = 26;
export const RESOLUTION_PREVIEW_MIN_WIDTH = 220;

export function resolutionPreviewMinimumSize(baseSize) {
  return [
    Math.max(Number(baseSize?.[0]) || 0, RESOLUTION_PREVIEW_MIN_WIDTH),
    (Number(baseSize?.[1]) || 0) + RESOLUTION_PREVIEW_HEIGHT,
  ];
}

export function clampResolutionPreviewSize(size, minimum, collapsed = false) {
  if (!size || collapsed) return size;
  size[0] = Math.max(size[0], minimum[0]);
  size[1] = Math.max(size[1], minimum[1]);
  return size;
}
