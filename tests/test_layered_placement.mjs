import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

import {
  DEFAULT_PLACEMENT,
  drawNormalizedBox,
  drawRect,
  moveNormalizedBox,
  moveRect,
  moveResolvedPlacement,
  normalizeBoxRegion,
  parseBoxEditorData,
  parsePlacementData,
  placementToRect,
  projectQuadPoint,
  rectToPlacement,
  resizeNormalizedBox,
  resizeRectFromDelta,
  rotationFromPointer,
  serializeBoxEditorData,
  serializePlacementData,
  stepPlacementValue,
  deformQuadCentroidLocked,
  dragQuadCorner,
  isValidQuad,
  normalizeRotation,
  PAINT_LAYER_KEY,
  pythonRound,
  resolveLayerGeometry,
  resolvedWorkspaceBounds,
} from "../web/placement_geometry.js";

const transformFixtures = JSON.parse(fs.readFileSync(
  new URL("./fixtures/staged_transform_geometry.json", import.meta.url), "utf8",
));

test("frontend rounding matches Python ties-to-even placement rounding", () => {
  assert.deepEqual(
    [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5].map(pythonRound),
    [-2, -2, 0, 0, 2, 2],
  );
});

test("paint placement metadata round trips without changing paint-free workflows", () => {
  const legacy = parsePlacementData('{"version":2,"layer_order":["foreground_0"],"layers":{}}');
  assert.equal(legacy.paint_layer, undefined);
  assert.equal(
    serializePlacementData(legacy),
    '{"version":2,"workspace_padding":0.5,"layer_order":["foreground_0"],"layers":{}}',
  );
  const parsed = parsePlacementData(JSON.stringify({
    version: 3,
    layer_order: ["foreground_0", PAINT_LAYER_KEY],
    layers: {},
    paint_layer: {
      included: false, asset_id: "paint-id", owner_node_id: "12", revision: 4,
      width: 640, height: 480,
      asset: { filename: "paint.png", subfolder: "clipspace", type: "input" },
    },
  }));
  assert.equal(parsed.paint_layer.asset.filename, "paint.png");
  assert.equal(parsed.paint_layer.included, false);
  assert.deepEqual(JSON.parse(serializePlacementData(parsed)).layer_order, ["foreground_0", PAINT_LAYER_KEY]);
});

test("resolved transform geometry matches backend pixel-space fixtures", () => {
  for (const fixture of transformFixtures) {
    const geometry = resolveLayerGeometry({
      backgroundWidth: fixture.background[0],
      backgroundHeight: fixture.background[1],
      sourceWidth: fixture.source[0],
      sourceHeight: fixture.source[1],
      placement: {
        scale: fixture.scale,
        center_x: fixture.center[0],
        center_y: fixture.center[1],
        rotation: fixture.rotation,
        corners: fixture.corners,
      },
      workspacePadding: fixture.padding,
    });
    assert.deepEqual([geometry.source.width, geometry.source.height], fixture.expected.source, fixture.name);
    assert.deepEqual([geometry.transformed.width, geometry.transformed.height], fixture.expected.output, fixture.name);
    assert.deepEqual([geometry.offset.x, geometry.offset.y], fixture.expected.offset, fixture.name);
    const visible = geometry.visible;
    assert.deepEqual([
      visible.destination.y,
      visible.destination.y + visible.destination.height,
      visible.destination.x,
      visible.destination.x + visible.destination.width,
      visible.source.y,
      visible.source.y + visible.source.height,
      visible.source.x,
      visible.source.x + visible.source.width,
    ], fixture.expected.visible, fixture.name);
    geometry.transformed.points.forEach((point, index) => {
      close(point[0], fixture.expected.points[index][0], 1e-9);
      close(point[1], fixture.expected.points[index][1], 1e-9);
    });
  }
});

test("moving a transformed layer clamps its expanded backend frame", () => {
  const placement = {
    scale: 0.5, center_x: 0.5, center_y: 0.5, rotation: 37,
    corners: [[-1, -1], [1, -1], [1, 1], [-1, 1]],
  };
  const input = {
    backgroundWidth: 1000, backgroundHeight: 800,
    sourceWidth: 400, sourceHeight: 200, placement, workspacePadding: 0.5,
  };
  const geometry = resolveLayerGeometry(input);
  const moved = moveResolvedPlacement(1000, 800, geometry, placement, 0, 1000, 0.5);
  const resolved = resolveLayerGeometry({ ...input, placement: moved });
  assert.equal(resolved.frame.y, 499);
  assert.equal(resolved.frame.height, 401);
  assert.equal(resolved.frame.y + resolved.frame.height, 900);
});

test("workspace framing includes resolved transformed extents", () => {
  const bounds = resolvedWorkspaceBounds(100, 80, 0.5, [
    { x: -30, y: 10, width: 40, height: 100 },
  ]);
  assert.deepEqual(bounds, { left: -30, top: -10, right: 112.5, bottom: 110 });
});

test("corner rotation follows pointer angle without snapping", () => {
  const rotation = rotationFromPointer(
    12.5,
    { x: 100, y: 100 },
    { x: 140, y: 100 },
    { x: 100, y: 140 },
  );
  close(rotation, 102.5);
  close(
    rotationFromPointer(
      170,
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: -1, y: -0.01 },
    ),
    -9.427061302316474,
  );
});

const close = (actual, expected, tolerance = 1e-9) => assert.ok(
  Math.abs(actual - expected) <= tolerance,
  `${actual} was not within ${tolerance} of ${expected}`,
);

test("landscape, portrait, and square map logical axes correctly", () => {
  const landscape = placementToRect(200, 100, 1, {
    scale: 0.5,
    long_axis_shift: 1,
    short_axis_shift: -1,
  });
  assert.deepEqual(landscape, { x: 150, y: 0, width: 50, height: 50 });

  const portrait = placementToRect(100, 200, 1, {
    scale: 0.5,
    long_axis_shift: 1,
    short_axis_shift: -1,
  });
  assert.deepEqual(portrait, { x: 0, y: 150, width: 50, height: 50 });

  const square = placementToRect(100, 100, 2, {
    scale: 0.5,
    long_axis_shift: 1,
    short_axis_shift: -1,
  });
  assert.deepEqual(square, { x: 50, y: 0, width: 50, height: 25 });
});

test("oversized placement uses negative travel and round trips", () => {
  const placement = { scale: 2, long_axis_shift: 0.6, short_axis_shift: -0.4 };
  const rect = placementToRect(160, 100, 0.5, placement);
  assert.ok(rect.height > 100);
  assert.ok(rect.y < 0);
  const roundTrip = rectToPlacement(160, 100, rect, placement);
  close(roundTrip.scale, placement.scale);
  const rebuilt = placementToRect(160, 100, 0.5, roundTrip);
  close(rebuilt.x, rect.x);
  close(rebuilt.y, rect.y);
});

test("legacy zero-travel placement migrates to an equivalent center", () => {
  const prior = { scale: 1, long_axis_shift: 0.75, short_axis_shift: -0.25 };
  const rect = placementToRect(100, 200, 1, prior);
  const result = rectToPlacement(100, 200, rect, prior);
  assert.deepEqual(placementToRect(100, 200, 1, result), rect);
});

test("drawing in opposite directions produces the same normalized bounds", () => {
  const forward = drawRect({ x: 10, y: 20 }, { x: 110, y: 120 }, 2);
  const reverse = drawRect({ x: 110, y: 120 }, { x: 10, y: 20 }, 2);
  assert.deepEqual(forward, reverse);
  close(forward.width / forward.height, 2);
});

test("moving clamps both contained and oversized rectangles", () => {
  assert.deepEqual(
    moveRect(100, 100, { x: 25, y: 25, width: 50, height: 50 }, 100, -100),
    { x: 50, y: 0, width: 50, height: 50 },
  );
  assert.deepEqual(
    moveRect(100, 100, { x: -50, y: -20, width: 200, height: 140 }, 100, -100),
    { x: 0, y: -40, width: 200, height: 140 },
  );
  assert.deepEqual(
    moveRect(100, 100, { x: -50, y: -20, width: 200, height: 140 }, 100, -100, 0.5),
    { x: 0, y: -40, width: 200, height: 140 },
  );
  assert.deepEqual(
    moveRect(100, 100, { x: 25, y: 25, width: 50, height: 50 }, -100, 100, 1),
    { x: -25, y: 75, width: 50, height: 50 },
  );
});

test("corner resizing stays proportional and respects scale limits", () => {
  const resized = resizeRectFromDelta(
    { x: 25, y: 25, width: 50, height: 25 },
    "se",
    40,
    20,
    2,
    5,
    80,
  );
  close(resized.width / resized.height, 2);
  assert.equal(Math.max(resized.width, resized.height), 80);
  assert.equal(resized.x, 25);
  assert.equal(resized.y, 25);
});

test("placement JSON is normalized and deterministically ordered", () => {
  const parsed = parsePlacementData('{"version":1,"workspace_padding":2,"layer_order":["foreground_10","foreground_2","foreground_10"],"layers":{"foreground_10":{"scale":2},"foreground_2":{}}}');
  assert.deepEqual(parsed.layers.foreground_2, { scale: 0.9, long_axis_shift: 0, short_axis_shift: 0, flip_horizontal: false, flip_vertical: false });
  assert.equal(parsed.workspace_padding, 1);
  assert.deepEqual(parsed.layer_order, ["foreground_10", "foreground_2"]);
  assert.equal(
    serializePlacementData(parsed),
    '{"version":1,"workspace_padding":1,"layer_order":["foreground_10","foreground_2"],"layers":{"foreground_2":{"scale":0.9,"long_axis_shift":0,"short_axis_shift":0,"flip_horizontal":false,"flip_vertical":false},"foreground_10":{"scale":2,"long_axis_shift":0,"short_axis_shift":0,"flip_horizontal":false,"flip_vertical":false}}}',
  );
  assert.deepEqual(parsePlacementData("not json"), { version: 2, workspace_padding: 0.5, layer_order: [], layers: {} });
});

test("version 2 stores normalized centers outside the output canvas", () => {
  const placement = rectToPlacement(100, 80, { x: -20, y: 60, width: 40, height: 20 });
  assert.deepEqual(placement, { scale: 0.5, center_x: 0, center_y: 0.875, flip_horizontal: false, flip_vertical: false });
  assert.deepEqual(placementToRect(100, 80, 2, placement), { x: -20, y: 60, width: 40, height: 20 });
  const encoded = serializePlacementData({
    version: 2,
    workspace_padding: 1,
    layer_order: [],
    layers: { foreground_0: placement },
  });
  assert.equal(encoded, '{"version":2,"workspace_padding":1,"layer_order":[],"layers":{"foreground_0":{"scale":0.5,"center_x":0,"center_y":0.875,"flip_horizontal":false,"flip_vertical":false}}}');
});

test("horizontal flip survives placement normalization and geometry edits", () => {
  const parsed = parsePlacementData('{"version":2,"layers":{"foreground_0":{"flip_horizontal":true}}}');
  assert.equal(parsed.layers.foreground_0.flip_horizontal, true);
  const rect = placementToRect(100, 80, 2, parsed.layers.foreground_0);
  assert.equal(rectToPlacement(100, 80, rect, parsed.layers.foreground_0).flip_horizontal, true);
});

test("vertical flip survives placement normalization and geometry edits", () => {
  const parsed = parsePlacementData('{"version":2,"layers":{"foreground_0":{"flip_vertical":true}}}');
  assert.equal(parsed.layers.foreground_0.flip_vertical, true);
  const rect = placementToRect(100, 80, 2, parsed.layers.foreground_0);
  assert.equal(rectToPlacement(100, 80, rect, parsed.layers.foreground_0).flip_vertical, true);
});

test("layer arrow controls step cleanly and respect placement limits", () => {
  assert.equal(stepPlacementValue("scale", 0.9, 0.01), 0.91);
  assert.equal(stepPlacementValue("center_x", 0.2734, 0.002), 0.2754);
  assert.equal(stepPlacementValue("center_x", 0.2734, 0.005), 0.2784);
  assert.equal(stepPlacementValue("scale", 0.05, -0.01), 0.05);
  assert.equal(stepPlacementValue("center_x", 0.5, -0.01), 0.49);
  assert.equal(stepPlacementValue("center_y", 10, 0.01), 10);
});

test("version 3 normalizes face transforms deterministically", () => {
  const parsed = parsePlacementData('{"version":3,"layers":{"foreground_0_face_0":{"rotation":181,"included":false}}}');
  assert.equal(parsed.layers.foreground_0_face_0.rotation, -179);
  assert.equal(parsed.layers.foreground_0_face_0.included, false);
  assert.equal(normalizeRotation(-181), 179);
  assert.match(serializePlacementData(parsed), /"version":3/);
});

test("version 3 normalizes ordinary foreground transforms deterministically", () => {
  const parsed = parsePlacementData('{"version":3,"layers":{"foreground_0":{"rotation":-181,"corners":[[-0.8,-1],[1,-0.9],[0.9,1],[-1,0.8]]}}}');
  assert.equal(parsed.layers.foreground_0.rotation, 179);
  assert.deepEqual(parsed.layers.foreground_0.corners, [[-0.8, -1], [1, -0.9], [0.9, 1], [-1, 0.8]]);
  assert.equal(parsed.layers.foreground_0.included, true);
  assert.equal(
    serializePlacementData(parsed),
    '{"version":3,"workspace_padding":0.5,"layer_order":[],"layers":{"foreground_0":{"scale":0.9,"center_x":0.5,"center_y":0.5,"flip_horizontal":false,"flip_vertical":false,"rotation":179,"corners":[[-0.8,-1],[1,-0.9],[0.9,1],[-1,0.8]],"included":true}}}',
  );
});

test("layer locks round trip additively without changing unlocked workflows", () => {
  const locked = parsePlacementData('{"version":3,"layers":{"foreground_0":{"locked":true}}}');
  assert.equal(locked.layers.foreground_0.locked, true);
  assert.match(serializePlacementData(locked), /"locked":true/);
  const unlocked = parsePlacementData('{"version":3,"layers":{"foreground_0":{"locked":false}}}');
  assert.equal(unlocked.layers.foreground_0.locked, undefined);
  assert.doesNotMatch(serializePlacementData(unlocked), /"locked"/);
});

test("rect edits preserve every face-specific transform field", () => {
  const prior = {
    scale: 0.25, center_x: 0.5, center_y: 0.5,
    flip_horizontal: true, flip_vertical: false,
    rotation: 37, corners: [[-0.8, -1], [1, -0.9], [0.9, 1], [-1, 0.8]],
    included: false,
  };
  const edited = rectToPlacement(100, 100, { x: 20, y: 25, width: 40, height: 40 }, prior);
  assert.equal(edited.rotation, 37);
  assert.deepEqual(edited.corners, prior.corners);
  assert.equal(edited.included, false);
});

test("quad corner and weighted deformation preserve valid centroid-locked geometry", () => {
  const quad = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
  const dragged = dragQuadCorner(quad, 0, -0.7, -0.8);
  assert.equal(isValidQuad(dragged), true);
  const deformed = deformQuadCentroidLocked(quad, [0.7, 0.7], 0.2, -0.1);
  assert.equal(isValidQuad(deformed), true);
  const centroid = (points) => points.reduce((sum, p) => [sum[0] + p[0] / 4, sum[1] + p[1] / 4], [0, 0]);
  assert.ok(centroid(deformed).every((v) => Math.abs(v) < 1e-12));
});

test("interior deformation turns the quad toward the drag direction without moving its centroid", () => {
  const quad = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
  const deformed = deformQuadCentroidLocked(quad, [0.8, 0], 0.3, 0);
  assert.equal(isValidQuad(deformed), true);
  const leftHeight = deformed[3][1] - deformed[0][1];
  const rightHeight = deformed[2][1] - deformed[1][1];
  assert.ok(rightHeight < leftHeight);
  const movement = deformed.reduce((sum, point, index) => [
    sum[0] + point[0] - quad[index][0],
    sum[1] + point[1] - quad[index][1],
  ], [0, 0]);
  assert.ok(movement.every((value) => Math.abs(value) < 1e-12));
});

test("projective quad mapping reaches corners and maps finite interior coordinates", () => {
  const quad = [[-1, -1], [0.7, -0.8], [0.8, 0.6], [-0.9, 1]];
  for (const [actual, expected] of [
    [projectQuadPoint(quad, 0, 0), quad[0]],
    [projectQuadPoint(quad, 1, 0), quad[1]],
    [projectQuadPoint(quad, 1, 1), quad[2]],
    [projectQuadPoint(quad, 0, 1), quad[3]],
  ]) {
    close(actual[0], expected[0]);
    close(actual[1], expected[1]);
  }
  assert.ok(projectQuadPoint(quad, 0.5, 0.5).every(Number.isFinite));
});

test("bounding box editor state round-trips and clamps move/resize/draw geometry", () => {
  const drawn = drawNormalizedBox({ x: 0.7, y: 0.8 }, { x: 0.2, y: 0.3 }, 0, "text");
  assert.deepEqual(
    { x: drawn.x, y: drawn.y, w: drawn.w, h: drawn.h, type: drawn.type },
    { x: 0.2, y: 0.3, w: 0.5, h: 0.5, type: "text" },
  );
  const moved = moveNormalizedBox(drawn, 0.4, -0.5);
  assert.equal(moved.x, 0.5);
  assert.equal(moved.y, 0);
  const resized = resizeNormalizedBox(moved, "nw", -0.2, 0.1);
  assert.equal(resized.x, 0.3);
  assert.equal(resized.y, 0.1);
  assert.equal(resized.w, 0.7);
  assert.equal(resized.h, 0.4);

  const serialized = serializeBoxEditorData({
    boxes: [{ ...resized, desc: "sign", text: "OPEN", included: false, locked: true }],
    last_incoming: null,
  });
  const parsed = parseBoxEditorData(serialized);
  assert.equal(parsed.boxes.length, 1);
  assert.equal(parsed.boxes[0].desc, "sign");
  assert.equal(parsed.boxes[0].text, "OPEN");
  assert.equal(parsed.boxes[0].included, false);
  assert.equal(parsed.boxes[0].locked, true);
  assert.deepEqual(parsed.boxes[0].palette, []);
  assert.equal(parsed.boxes[0].outline_color, "#ff6b6b");
  const secondBox = normalizeBoxRegion({ x: 0.1, y: 0.1, w: 0.2, h: 0.2 }, 1);
  assert.equal(secondBox.outline_color, "#4ecdc4");
  assert.equal(normalizeBoxRegion(secondBox, 0).outline_color, "#4ecdc4");
  assert.deepEqual(normalizeBoxRegion({ palette: ["#FF6B6B"] }).palette, ["#ff6b6b"]);
  assert.ok(normalizeBoxRegion({ x: -1, y: -1, w: 2, h: 2 }).w <= 1);
});
