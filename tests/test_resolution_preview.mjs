import test from "node:test";
import assert from "node:assert/strict";

import {
  clampResolutionPreviewSize,
  resolutionPreviewMinimumSize,
} from "../web/resolution_preview_layout.js";


test("resolution preview reserves a bottom band in computed minimum size", () => {
  assert.deepEqual(resolutionPreviewMinimumSize([180, 220]), [220, 246]);
  assert.deepEqual(resolutionPreviewMinimumSize([320, 180]), [320, 206]);
});


test("resolution preview prevents manual resize below its computed minimum", () => {
  const size = [160, 190];
  assert.equal(clampResolutionPreviewSize(size, [220, 246]), size);
  assert.deepEqual(size, [220, 246]);
});


test("collapsed nodes keep their collapsed size", () => {
  const size = [90, 30];
  clampResolutionPreviewSize(size, [220, 246], true);
  assert.deepEqual(size, [90, 30]);
});
