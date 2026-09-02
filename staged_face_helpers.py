import logging
import os

import numpy as np
import torch

from .model_assets import require_huggingface_model
from .background_replace_helpers import _expanded_box, _ordered_ring, _polygon_mask
from .composite_helpers import (
    _expand_mask,
    _feather_mask,
    _ordered_single_foregrounds,
    _resize_composite_mask,
    background_removal_with_alpha,
)
from .staged_compositor_helpers import bounded_editor_preview, _stage_layered_foregrounds


def load_face_model():
    import comfy.utils
    from comfy_extras.nodes_mediapipe import FaceLandmarkerModel

    filename = "mediapipe_face_fp32.safetensors"
    path = require_huggingface_model(
        "detection",
        filename,
        "Comfy-Org/mediapipe",
        f"detection/{filename}",
    )
    path = os.path.normcase(os.path.abspath(path))
    try:
        model = FaceLandmarkerModel(comfy.utils.load_torch_file(path, safe_load=True))
    except Exception as exc:
        raise ValueError(f"Comfy Core could not load {filename} as a face detection model.") from exc
    return model


def _box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-6)


def _same_face(a, b):
    if _box_iou(a["bbox_xyxy"], b["bbox_xyxy"]) >= 0.3:
        return True
    boxes = [np.asarray(face["bbox_xyxy"], dtype=np.float32) for face in (a, b)]
    centers = [(box[:2] + box[2:]) * 0.5 for box in boxes]
    scales = [max(box[2] - box[0], box[3] - box[1], 1.0) for box in boxes]
    return np.linalg.norm(centers[0] - centers[1]) <= 0.35 * max(scales) and (
        min(scales) / max(scales) >= 0.45
    )


def _merge_cluster(cluster):
    weights = np.asarray([
        max(float(face.get("score", 0.0)), 1e-4)
        * max(float(face.get("presence", 1.0)), 1e-4)
        for face in cluster
    ], dtype=np.float32)
    weights /= weights.sum()
    boxes = np.stack([np.asarray(face["bbox_xyxy"], dtype=np.float32) for face in cluster])
    # Use the union for extraction so a partial high-confidence observation cannot
    # cut away regions supported by another threshold pass.
    merged = dict(max(cluster, key=lambda face: float(face.get("score", 0.0))))
    merged["bbox_xyxy"] = np.array([
        boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
    ], dtype=np.float32)
    merged["center_xy"] = np.sum(
        ((boxes[:, :2] + boxes[:, 2:]) * 0.5) * weights[:, None], axis=0
    )
    merged["score"] = float(max(float(face.get("score", 0.0)) for face in cluster))
    merged["observations"] = len(cluster)
    return merged


def _run_detection_batch(model, images_uint8, maximum_faces, threshold):
    return model.detect_batch(
        images_uint8, num_faces=max(1, int(maximum_faces)) * 3,
        score_thresh=threshold, variant="full",
    )


def _merged_faces(candidates, maximum_faces):
    clusters = []
    for face in sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        matching = next((cluster for cluster in clusters if any(_same_face(face, other) for other in cluster)), None)
        if matching is None:
            clusters.append([face])
        else:
            matching.append(face)
    merged = [_merge_cluster(cluster) for cluster in clusters]
    merged.sort(key=lambda face: (float(face["center_xy"][1]), float(face["center_xy"][0])))
    return merged[:max(1, int(maximum_faces))]


def detect_faces_adaptive(model, image_uint8, maximum_faces, threshold):
    results, failed = detect_many_or_warn(
        model, [("", image_uint8)], maximum_faces, threshold
    )
    if failed:
        raise RuntimeError("MediaPipe face detection failed.")
    return results[""]


def detect_many_or_warn(model, images, maximum_faces, threshold):
    threshold = min(max(float(threshold), 0.01), 1.0)
    thresholds = []
    for value in (threshold, max(0.05, round(threshold * 0.7, 4)), max(0.05, round(threshold * 0.5, 4))):
        if value not in thresholds:
            thresholds.append(value)

    results = {socket: [] for socket, _ in images}
    pending = list(images)
    failed = set()
    for pass_threshold in thresholds:
        if not pending:
            break
        try:
            batches = _run_detection_batch(
                model, [image for _, image in pending], maximum_faces, pass_threshold
            )
            pass_results = list(zip(pending, batches))
        except Exception:
            pass_results = []
            for item in pending:
                try:
                    batches = _run_detection_batch(
                        model, [item[1]], maximum_faces, pass_threshold
                    )
                    pass_results.append((item, batches[0]))
                except Exception:
                    failed.add(item[0])
                    logging.warning(
                        "MediaPipe face detection failed for %s; retaining ordinary foreground only.",
                        item[0], exc_info=True,
                    )

        next_pending = []
        for (socket, image), candidates in pass_results:
            if socket in failed:
                continue
            if candidates:
                results[socket] = _merged_faces(candidates, maximum_faces)
            else:
                next_pending.append((socket, image))
        pending = next_pending
    return results, failed


def detect_or_warn(model, image_uint8, maximum_faces, threshold, socket):
    results, failed = detect_many_or_warn(
        model, [(socket, image_uint8)], maximum_faces, threshold
    )
    return results[socket], socket in failed


def face_removal_with_alpha(
    image,
    background_removal_model,
    face_detection_model,
    background_options,
    face_options,
):
    """Return padded straight-RGBA face crops and their exact alpha masks."""
    rgba, foreground_alpha = background_removal_with_alpha(
        image, background_removal_model, background_options
    )
    rgb = rgba[..., :3]
    processing_resolution = max(
        0, int(background_options.get("mask_processing_resolution", 0) or 0)
    )
    detection_inputs = []
    detection_shapes = []
    for image_index, source in enumerate(rgb):
        detection_rgb = bounded_editor_preview(
            source.unsqueeze(0), processing_resolution
        )
        detection_shapes.append(detection_rgb.shape[1:3])
        detection_inputs.append(
            (
                str(image_index),
                detection_rgb.mul(255)
                .add(0.5)
                .clamp(0, 255)
                .to(torch.uint8)
                .cpu()
                .numpy()[0],
            )
        )
    detected_by_image, failed_images = detect_many_or_warn(
        face_detection_model,
        detection_inputs,
        face_options["maximum_faces"],
        face_options["detection_threshold"],
    )

    ring = _ordered_ring(face_detection_model.connection_sets["face_oval"])
    crops = []
    for image_index, source in enumerate(rgb):
        key = str(image_index)
        if key in failed_images:
            continue
        source_height, source_width = source.shape[:2]
        detection_height, detection_width = detection_shapes[image_index]
        scale_x = source_width / detection_width
        scale_y = source_height / detection_height
        for face in detected_by_image[key]:
            scaled_box = np.asarray(face["bbox_xyxy"], dtype=np.float32) * np.asarray(
                [scale_x, scale_y, scale_x, scale_y], dtype=np.float32
            )
            x1, y1, x2, y2 = _expanded_box(
                scaled_box,
                face_options["bbox_expansion"],
                source_width,
                source_height,
            )
            landmarks = face["landmarks_xy"] * np.asarray(
                [scale_x, scale_y], dtype=np.float32
            )
            points = landmarks[ring] - np.asarray([x1, y1], dtype=np.float32)
            face_alpha = _polygon_mask(
                y2 - y1, x2 - x1, points, source.device, source.dtype
            )
            face_alpha = face_alpha * foreground_alpha[
                image_index, y1:y2, x1:x2
            ].to(face_alpha)
            face_alpha = _expand_mask(
                face_alpha, face_options["mask_expansion"]
            ).clamp(0, 1)
            feather_radius = max(0, int(face_options["face_feather_radius"]))
            if feather_radius:
                face_alpha = _feather_mask(face_alpha, -feather_radius)
            if torch.any(face_alpha > 0):
                crops.append((source[y1:y2, x1:x2], face_alpha))

    if not crops:
        raise ValueError("No face was detected in the input image batch.")

    output_height = max(crop.shape[0] for crop, _ in crops)
    output_width = max(crop.shape[1] for crop, _ in crops)
    output_rgba = []
    output_masks = []
    for crop, alpha in crops:
        height, width = crop.shape[:2]
        top = (output_height - height) // 2
        left = (output_width - width) // 2
        padded_rgb = crop.new_zeros((output_height, output_width, 3))
        padded_alpha = alpha.new_zeros((output_height, output_width))
        padded_rgb[top : top + height, left : left + width] = crop
        padded_alpha[top : top + height, left : left + width] = alpha
        output_rgba.append(torch.cat((padded_rgb, padded_alpha.unsqueeze(-1)), -1))
        output_masks.append(padded_alpha)
    return torch.stack(output_rgba), torch.stack(output_masks)


def _stage_face_foregrounds(
    background_removal_model,
    face_detection_model,
    foreground_images,
    background_options,
    face_options,
    placement_data=None,
):
    staged, full_alpha_by_socket = _stage_layered_foregrounds(
        background_removal_model,
        foreground_images,
        background_options["mask_threshold"],
        background_options["border_cleanup_width"],
        background_options["artifact_cleanup_radius"],
        background_options["gap_fill_radius"],
        background_options["mask_resize_method"],
        placement_data,
        retain_full_alpha=True,
        mask_processing_resolution=background_options[
            "mask_processing_resolution"
        ],
    )
    ordinary = {layer["socket"]: layer for layer in staged["layers"]}
    result, warning_count = [], 0
    ring = _ordered_ring(face_detection_model.connection_sets["face_oval"])
    foregrounds = _ordered_single_foregrounds(foreground_images)
    detection_inputs = []
    detection_shapes = {}
    processing_resolution = int(
        staged.get("mask_processing_resolution", 0) or 0
    )
    for key, original in foregrounds:
        rgb = original[..., :3]
        detection_rgb = bounded_editor_preview(rgb, processing_resolution)
        detection_shapes[key] = detection_rgb.shape[1:3]
        detection_inputs.append(
            (
                key,
                detection_rgb.mul(255).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()[0],
            )
        )
    detected_by_socket, failed_sockets = detect_many_or_warn(
        face_detection_model,
        detection_inputs,
        face_options["maximum_faces"],
        face_options["detection_threshold"],
    )
    warning_count = len(failed_sockets)

    for key, original in foregrounds:
        result.append(ordinary[key])
        rgb = original[..., :3]
        if key in failed_sockets:
            continue
        alpha_info = full_alpha_by_socket[key]
        full_alpha = alpha_info["mask"]
        detection_height, detection_width = detection_shapes[key]
        scale_x = rgb.shape[2] / detection_width
        scale_y = rgb.shape[1] / detection_height
        faces = detected_by_socket[key]
        for face_index, face in enumerate(faces):
            scaled_box = np.asarray(face["bbox_xyxy"], dtype=np.float32) * np.asarray(
                [scale_x, scale_y, scale_x, scale_y], dtype=np.float32
            )
            x1, y1, x2, y2 = _expanded_box(
                scaled_box,
                face_options["bbox_expansion"],
                rgb.shape[2],
                rgb.shape[1],
            )
            landmarks = face["landmarks_xy"] * np.asarray(
                [scale_x, scale_y], dtype=np.float32
            )
            points = landmarks[ring] - np.asarray([x1, y1], dtype=np.float32)
            crop_mask = _polygon_mask(y2 - y1, x2 - x1, points, rgb.device, rgb.dtype)
            alpha_height, alpha_width = full_alpha.shape
            alpha_x1 = max(0, min(alpha_width - 1, x1 * alpha_width // rgb.shape[2]))
            alpha_y1 = max(0, min(alpha_height - 1, y1 * alpha_height // rgb.shape[1]))
            alpha_x2 = max(
                alpha_x1 + 1,
                min(alpha_width, (x2 * alpha_width + rgb.shape[2] - 1) // rgb.shape[2]),
            )
            alpha_y2 = max(
                alpha_y1 + 1,
                min(alpha_height, (y2 * alpha_height + rgb.shape[1] - 1) // rgb.shape[1]),
            )
            alpha_crop = full_alpha[None, alpha_y1:alpha_y2, alpha_x1:alpha_x2]
            alpha_crop = _resize_composite_mask(
                alpha_crop,
                x2 - x1,
                y2 - y1,
                background_options["mask_resize_method"],
            )[0]
            crop_mask = crop_mask * alpha_crop.to(crop_mask)
            crop_mask = _expand_mask(crop_mask, face_options["mask_expansion"]).clamp(
                0, 1
            )[None]
            if not torch.any(crop_mask > 0):
                continue
            socket = f"{key}_face_{face_index}"
            result.append(
                {
                    "socket": socket,
                    "image": rgb[:, y1:y2, x1:x2],
                    "mask": crop_mask,
                    "is_face": True,
                    "parent_socket": key,
                    "uses_embedded_alpha": True,
                    "default_scale": face_options["initial_face_scale"],
                    "feather_radius": face_options["face_feather_radius"],
                }
            )
    staged["layers"] = result
    staged["face_warning_count"] = warning_count
    return staged
