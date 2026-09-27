import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../web/h3_autogrow.js", import.meta.url), "utf8")
  .replace('import { app } from "../../scripts/app.js";',
    "const app = { registerExtension(extension) { globalThis.h3AutogrowExtension = extension; } };");
const { trimH3AutogrowInputs } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

function nodeWithInputs(inputs) {
  return {
    inputs: inputs.map(([name, connected]) => ({ name, link: connected ? 1 : null })),
    comfyDynamic: { autogrow: { reference_images: {}, fusion_images: {} } },
    size: [220, 400],
    removeInput(index) { this.inputs.splice(index, 1); },
    computeSize() { return [180, 30 + this.inputs.length * 20]; },
    setSize(size) { this.size = size; },
  };
}

test("workflow load keeps connections and one empty H3 socket per group", () => {
  const node = nodeWithInputs([
    ["clip", true],
    ["reference_images.reference_image_1", false],
    ["fusion_images.fusion_image_1", true],
    ["fusion_images.fusion_image_2", true],
    ["fusion_images.fusion_image_2", false],
    ["fusion_images.fusion_image_3", false],
    ["fusion_images.fusion_image_3", false],
    ["fusion_images.fusion_image_4", false],
  ]);
  node.type = "UC_AdvancedMiniMaxH3ImageToVideo";

  globalThis.h3AutogrowExtension.loadedGraphNode(node);
  assert.deepEqual(node.inputs.map((input) => [input.name, input.link != null]), [
    ["clip", true],
    ["reference_images.reference_image_1", false],
    ["fusion_images.fusion_image_1", true],
    ["fusion_images.fusion_image_2", true],
    ["fusion_images.fusion_image_3", false],
  ]);
  assert.equal(trimH3AutogrowInputs(node), false);
});
