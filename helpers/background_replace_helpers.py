import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def _ordered_ring(edges):
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    start = next(iter(adjacency))
    ring = [start]
    previous = None
    current = start
    while True:
        next_index = next(
            (index for index in adjacency[current] if index != previous), None
        )
        if next_index is None or next_index == start:
            break
        ring.append(next_index)
        previous, current = current, next_index
    return ring


def _polygon_mask(height, width, points, device, dtype):
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in points], fill=255)
    return (
        torch.from_numpy(np.asarray(image).copy())
        .to(device=device, dtype=dtype)
        .div_(255.0)
    )


def _expanded_box(box, padding, width, height):
    x1 = max(0, math.floor(float(box[0])) - padding)
    y1 = max(0, math.floor(float(box[1])) - padding)
    x2 = min(width, math.ceil(float(box[2])) + padding)
    y2 = min(height, math.ceil(float(box[3])) + padding)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detected face has an empty bounding box.")
    return x1, y1, x2, y2


def _largest_face(faces, name):
    if not faces:
        raise ValueError(f"No face was detected in the {name} image.")
    return max(
        faces,
        key=lambda face: (
            max(0.0, float(face["bbox_xyxy"][2] - face["bbox_xyxy"][0]))
            * max(0.0, float(face["bbox_xyxy"][3] - face["bbox_xyxy"][1]))
        ),
    )


def _similarity_transform(source_points, target_points):
    source_points = np.asarray(source_points, dtype=np.float32)
    target_points = np.asarray(target_points, dtype=np.float32)
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    centered_source = source_points - source_center
    centered_target = target_points - target_center
    covariance = centered_source.T @ centered_target
    left, singular_values, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        singular_values[-1] *= -1
        rotation = right.T @ left.T
    scale = float(
        singular_values.sum()
        / max(float((centered_source * centered_source).sum()), 1e-6)
    )
    translation = target_center - scale * (source_center @ rotation.T)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def _transform_source(
    source, oval, foreground, output_height, output_width, scale, rotation, translation
):
    device = source.device
    yy, xx = torch.meshgrid(
        torch.arange(output_height, device=device, dtype=torch.float32),
        torch.arange(output_width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    output_points = torch.stack((xx, yy), dim=-1)
    rotation = torch.as_tensor(rotation, device=device, dtype=torch.float32)
    translation = torch.as_tensor(translation, device=device, dtype=torch.float32)
    source_points = ((output_points - translation) @ rotation) / max(float(scale), 1e-6)
    grid_x = (source_points[..., 0] + 0.5) * (2.0 / source.shape[1]) - 1.0
    grid_y = (source_points[..., 1] + 0.5) * (2.0 / source.shape[0]) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    layers = torch.cat(
        (source.movedim(-1, 0), oval.unsqueeze(0), foreground.unsqueeze(0)), dim=0
    ).unsqueeze(0)
    transformed = F.grid_sample(
        layers, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0]
    return transformed[:3].movedim(0, -1), transformed[3], transformed[4]


def _smoothstep(value):
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _add_control(controls, values, seen, point, value):
    key = (round(float(point[0]), 2), round(float(point[1]), 2))
    if key not in seen:
        seen.add(key)
        controls.append(point)
        values.append(value)


def _warp_target(
    target, source_oval_points, target_oval_points, strength, decay_radius
):
    if strength <= 0:
        return target
    device = target.device
    height, width = target.shape[:2]
    source_points = np.asarray(source_oval_points, dtype=np.float32)
    target_points = np.asarray(target_oval_points, dtype=np.float32)
    center = source_points.mean(axis=0)
    controls = []
    values = []
    seen = set()
    for source_point, target_point in zip(source_points, target_points):
        _add_control(
            controls,
            values,
            seen,
            source_point,
            (target_point - source_point) * float(strength),
        )
    for source_point in source_points:
        direction = source_point - center
        length = np.linalg.norm(direction)
        if length > 0:
            fixed = source_point + direction * (float(decay_radius) / length)
            fixed[0] = np.clip(fixed[0], 0, width - 1)
            fixed[1] = np.clip(fixed[1], 0, height - 1)
            _add_control(controls, values, seen, fixed, np.zeros(2, dtype=np.float32))
    border_step = max(8, min(32, int(decay_radius) // 2))
    for x in range(0, width, border_step):
        _add_control(
            controls,
            values,
            seen,
            np.array([x, 0], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
        _add_control(
            controls,
            values,
            seen,
            np.array([x, height - 1], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
    for y in range(0, height, border_step):
        _add_control(
            controls,
            values,
            seen,
            np.array([0, y], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
        _add_control(
            controls,
            values,
            seen,
            np.array([width - 1, y], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
    for point in ((width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        _add_control(
            controls,
            values,
            seen,
            np.array(point, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )

    controls = torch.as_tensor(np.asarray(controls), device=device, dtype=torch.float32)
    values = torch.as_tensor(np.asarray(values), device=device, dtype=torch.float32)
    controls[:, 0] = (controls[:, 0] + 0.5) * (2.0 / width) - 1.0
    controls[:, 1] = (controls[:, 1] + 0.5) * (2.0 / height) - 1.0
    values[:, 0] *= 2.0 / width
    values[:, 1] *= 2.0 / height
    difference = controls[:, None] - controls[None]
    distance_squared = (difference * difference).sum(dim=-1)
    kernel = distance_squared * torch.log(distance_squared + 1e-6)
    kernel.diagonal().add_(1e-4)
    affine = torch.cat(
        (torch.ones((controls.shape[0], 1), device=device), controls), dim=1
    )
    system = torch.cat(
        (
            torch.cat((kernel, affine), dim=1),
            torch.cat((affine.T, torch.zeros((3, 3), device=device)), dim=1),
        ),
        dim=0,
    )
    coefficients = torch.linalg.solve(
        system, torch.cat((values, torch.zeros((3, 2), device=device)), dim=0)
    )

    grid_rows = []
    x_coordinates = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (
        2.0 / width
    ) - 1.0
    for start in range(0, height, 64):
        end = min(start + 64, height)
        y_coordinates = (
            torch.arange(start, end, device=device, dtype=torch.float32) + 0.5
        ) * (2.0 / height) - 1.0
        yy, xx = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        points = torch.stack((xx, yy), dim=-1)
        difference = points.unsqueeze(-2) - controls
        distance_squared = (difference * difference).sum(dim=-1)
        basis = distance_squared * torch.log(distance_squared + 1e-6)
        point_affine = torch.cat(
            (torch.ones((*points.shape[:-1], 1), device=device), points), dim=-1
        )
        displacement = basis @ coefficients[:-3] + point_affine @ coefficients[-3:]
        pixel_x = torch.arange(width, device=device, dtype=torch.float32)
        pixel_y = torch.arange(start, end, device=device, dtype=torch.float32)
        edge_x = torch.minimum(pixel_x, width - 1 - pixel_x).unsqueeze(0)
        edge_y = torch.minimum(pixel_y, height - 1 - pixel_y).unsqueeze(1)
        displacement *= _smoothstep(torch.minimum(edge_x, edge_y) / 2.0).unsqueeze(-1)
        grid_rows.append(points + displacement)
    grid = torch.cat(grid_rows, dim=0).unsqueeze(0)
    warped = F.grid_sample(
        target.movedim(-1, 0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return warped.squeeze(0).movedim(0, -1)
