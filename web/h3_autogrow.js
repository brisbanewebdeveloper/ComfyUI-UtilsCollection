import { app } from "../../scripts/app.js";

const NODE_IDS = new Set([
  "UC_AdvancedMiniMaxH3ImageToVideo",
  "UC_MiniMaxH3ClipContinuationEncoder",
  "UC_AdvMiniMaxH3ImageToVideoTokenFusion",
  "UC_AdvMiniMaxH3ImageToVideoTemporalFusion",
  "UC_AdvMiniMaxH3ImageToVideoTemporalTokenFusion",
]);

const GROUPS = [
  ["reference_images", "reference_image_"],
  ["fusion_images", "fusion_image_"],
];

export function trimH3AutogrowInputs(node) {
  let removed = false;
  for (const [group, prefix] of GROUPS) {
    if (!node.comfyDynamic?.autogrow?.[group]) continue;
    const start = `${group}.${prefix}`;
    const slots = node.inputs.flatMap((input, index) => {
      if (!input.name.startsWith(start)) return [];
      const suffix = input.name.slice(start.length);
      if (!/^[1-9]\d*$/.test(suffix)) return [];
      return [{ index, ordinal: Number(suffix), connected: input.link != null }];
    });
    const next = 1 + Math.max(0, ...slots.filter((slot) => slot.connected).map((slot) => slot.ordinal));
    let keptPlaceholder = false;
    for (const slot of slots.reverse()) {
      if (slot.connected) continue;
      if (slot.ordinal === next && !keptPlaceholder) {
        keptPlaceholder = true;
        continue;
      }
      node.removeInput(slot.index);
      removed = true;
    }
  }
  if (removed) {
    const [width, height] = node.computeSize();
    node.setSize([Math.max(node.size[0], width), height]);
    node.graph?.setDirtyCanvas?.(true, true);
  }
  return removed;
}

app.registerExtension({
  name: "ComfyUI.UtilsCollection.H3Autogrow",
  loadedGraphNode(node) {
    if (NODE_IDS.has(node.comfyClass) || NODE_IDS.has(node.type)) {
      trimH3AutogrowInputs(node);
    }
  },
});
