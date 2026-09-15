import math

import torch

from ..color_name_data import XKCD_COLOR_NAMES


_HISTOGRAM_LEVELS = 32
_HISTOGRAM_BINS = _HISTOGRAM_LEVELS**3
_PIXEL_CHUNK_SIZE = 1_048_576
_KMEANS_ITERATIONS = 12
_PALETTE_CELL_SIZE = 128
_NAMED_COLOR_NAMES = tuple(XKCD_COLOR_NAMES)
_NAMED_COLOR_RGB = torch.tensor(
    tuple(XKCD_COLOR_NAMES.values()), dtype=torch.float32
) / 255.0
_NAMED_COLOR_LAB = None


def _weighted_rgb_histogram(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = image[..., :3].reshape(-1, 3)
    device = pixels.device
    counts = torch.zeros(_HISTOGRAM_BINS, dtype=torch.int64, device=device)
    rgb_sums = torch.zeros(
        (_HISTOGRAM_BINS, 3), dtype=torch.float32, device=device
    )

    for start in range(0, pixels.shape[0], _PIXEL_CHUNK_SIZE):
        chunk = pixels[start : start + _PIXEL_CHUNK_SIZE].to(torch.float32)
        chunk = torch.nan_to_num(chunk, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        quantized = torch.round(chunk * (_HISTOGRAM_LEVELS - 1)).to(torch.int64)
        indices = (
            quantized[:, 0] * (_HISTOGRAM_LEVELS**2)
            + quantized[:, 1] * _HISTOGRAM_LEVELS
            + quantized[:, 2]
        )
        counts += torch.bincount(indices, minlength=_HISTOGRAM_BINS)
        rgb_sums.index_add_(0, indices, chunk)

    occupied = counts > 0
    occupied_counts = counts[occupied].to(torch.float32)
    occupied_rgb = rgb_sums[occupied] / occupied_counts[:, None]
    return occupied_rgb.cpu(), occupied_counts.cpu()


def _srgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
    )
    transform = rgb.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ transform.T
    xyz = xyz / rgb.new_tensor([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    converted = torch.where(
        xyz > delta**3,
        torch.pow(xyz.clamp_min(0.0), 1.0 / 3.0),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lightness = 116.0 * converted[:, 1] - 16.0
    green_red = 500.0 * (converted[:, 0] - converted[:, 1])
    blue_yellow = 200.0 * (converted[:, 1] - converted[:, 2])
    return torch.stack((lightness, green_red, blue_yellow), dim=1)


def _initial_cluster_centers(
    lab: torch.Tensor, weights: torch.Tensor, cluster_count: int
) -> torch.Tensor:
    selected = [int(torch.argmax(weights).item())]
    nearest_distance = torch.full((lab.shape[0],), torch.inf, dtype=lab.dtype)
    normalized_weight = torch.sqrt(weights / weights.max().clamp_min(1.0))

    while len(selected) < cluster_count:
        latest = lab[selected[-1]]
        distance = torch.sum((lab - latest) ** 2, dim=1)
        nearest_distance = torch.minimum(nearest_distance, distance)
        score = nearest_distance * normalized_weight
        score[selected] = -1.0
        selected.append(int(torch.argmax(score).item()))

    return lab[selected].clone()


def _cluster_histogram(
    rgb: torch.Tensor,
    lab: torch.Tensor,
    weights: torch.Tensor,
    requested_colors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cluster_count = min(
        lab.shape[0],
        max(16, requested_colors * 4),
        256,
    )
    centers = _initial_cluster_centers(lab, weights, cluster_count)

    for _ in range(_KMEANS_ITERATIONS):
        labels = torch.cdist(lab, centers).argmin(dim=1)
        cluster_weights = torch.zeros(cluster_count, dtype=weights.dtype)
        cluster_weights.index_add_(0, labels, weights)
        weighted_lab = torch.zeros_like(centers)
        weighted_lab.index_add_(0, labels, lab * weights[:, None])
        nonempty = cluster_weights > 0
        updated = centers.clone()
        updated[nonempty] = (
            weighted_lab[nonempty] / cluster_weights[nonempty, None]
        )
        if torch.allclose(updated, centers, rtol=0.0, atol=1e-3):
            centers = updated
            break
        centers = updated

    labels = torch.cdist(lab, centers).argmin(dim=1)
    cluster_weights = torch.zeros(cluster_count, dtype=weights.dtype)
    cluster_weights.index_add_(0, labels, weights)
    weighted_rgb = torch.zeros((cluster_count, 3), dtype=rgb.dtype)
    weighted_rgb.index_add_(0, labels, rgb * weights[:, None])
    nonempty = cluster_weights > 0
    return (
        weighted_rgb[nonempty] / cluster_weights[nonempty, None],
        cluster_weights[nonempty],
    )


def _merge_to_requested_count(
    rgb: torch.Tensor,
    weights: torch.Tensor,
    requested_colors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_count = min(requested_colors, rgb.shape[0])
    lab = _srgb_to_lab(rgb)

    while rgb.shape[0] > target_count:
        distances = torch.cdist(lab, lab)
        distances.fill_diagonal_(torch.inf)
        closest = int(torch.argmin(distances).item())
        first = closest // distances.shape[1]
        second = closest % distances.shape[1]
        if second < first:
            first, second = second, first

        combined_weight = weights[first] + weights[second]
        combined_rgb = (
            rgb[first] * weights[first] + rgb[second] * weights[second]
        ) / combined_weight
        keep = torch.ones(rgb.shape[0], dtype=torch.bool)
        keep[second] = False
        rgb = rgb[keep]
        weights = weights[keep]
        lab = lab[keep]
        rgb[first] = combined_rgb
        weights[first] = combined_weight
        lab[first] = _srgb_to_lab(combined_rgb[None])[0]

    output_order = torch.argsort(weights, descending=True)
    return rgb[output_order], weights[output_order]


def _extract_prevalent_rgb_colors(
    image: torch.Tensor,
    color_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("Image must have [batch, height, width, channels] shape.")
    if image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] == 0:
        raise ValueError("Image batch and spatial dimensions must be non-empty.")
    if color_count < 1:
        raise ValueError("Color count must be at least 1.")

    rgb, weights = _weighted_rgb_histogram(image)
    lab = _srgb_to_lab(rgb)
    clustered_rgb, cluster_weights = _cluster_histogram(
        rgb, lab, weights, color_count
    )
    selected_rgb, selected_weights = _merge_to_requested_count(
        clustered_rgb, cluster_weights, color_count
    )
    return selected_rgb.clamp(0.0, 1.0), selected_weights


def _format_hex_palette(colors: torch.Tensor, prefix_hash: bool) -> str:
    prefix = "#" if prefix_hash else ""
    color_values = torch.round(colors * 255.0).to(torch.uint8)
    return ", ".join(
        f"{prefix}{red:02X}{green:02X}{blue:02X}"
        for red, green, blue in color_values.tolist()
    )


def _render_palette_grid(colors: torch.Tensor) -> torch.Tensor:
    color_count = colors.shape[0]
    rows = max(1, math.isqrt(color_count))
    columns = math.ceil(color_count / rows)
    grid = torch.zeros(
        (
            1,
            rows * _PALETTE_CELL_SIZE,
            columns * _PALETTE_CELL_SIZE,
            3,
        ),
        dtype=torch.float32,
    )
    for index, color in enumerate(colors):
        row = index // columns
        column = index % columns
        top = row * _PALETTE_CELL_SIZE
        left = column * _PALETTE_CELL_SIZE
        grid[
            0,
            top : top + _PALETTE_CELL_SIZE,
            left : left + _PALETTE_CELL_SIZE,
        ] = color
    return grid


def _nearest_color_names(colors: torch.Tensor) -> str:
    global _NAMED_COLOR_LAB
    if _NAMED_COLOR_LAB is None:
        _NAMED_COLOR_LAB = _srgb_to_lab(_NAMED_COLOR_RGB)
    palette_lab = _srgb_to_lab(colors)
    nearest = torch.cdist(palette_lab, _NAMED_COLOR_LAB).argmin(dim=1)
    return ", ".join(_NAMED_COLOR_NAMES[index] for index in nearest.tolist())


def _describe_palette(colors: torch.Tensor, weights: torch.Tensor) -> str:
    normalized_weights = weights / weights.sum().clamp_min(1.0)
    lab = _srgb_to_lab(colors)
    lightness = lab[:, 0]
    chroma = torch.linalg.vector_norm(lab[:, 1:], dim=1)
    average_lightness = float(torch.sum(lightness * normalized_weights))
    average_chroma = float(torch.sum(chroma * normalized_weights))

    if average_lightness < 25.0:
        lightness_description = "very dark"
    elif average_lightness < 40.0:
        lightness_description = "dark"
    elif average_lightness < 60.0:
        lightness_description = "midtone"
    elif average_lightness < 78.0:
        lightness_description = "light"
    else:
        lightness_description = "very light"

    if average_chroma < 10.0:
        chroma_description = "neutral"
    elif average_chroma < 25.0:
        chroma_description = "muted"
    elif average_chroma < 50.0:
        chroma_description = "moderately saturated"
    else:
        chroma_description = "vivid"

    chromatic = chroma >= 10.0
    chromatic_weight = normalized_weights[chromatic].sum()
    if chromatic_weight < 0.2:
        temperature_description = "neutral-temperature"
    else:
        hue = torch.rad2deg(
            torch.atan2(lab[chromatic, 2], lab[chromatic, 1])
        ) % 360.0
        chromatic_weights = normalized_weights[chromatic] / chromatic_weight
        warm = (hue < 110.0) | (hue >= 345.0)
        warm_share = float(chromatic_weights[warm].sum())
        if warm_share >= 0.65:
            temperature_description = "warm"
        elif warm_share <= 0.35:
            temperature_description = "cool"
        else:
            temperature_description = "mixed-temperature"

    if colors.shape[0] == 1:
        contrast_description = "uniform color"
    else:
        maximum_distance = float(torch.pdist(lab).max())
        if maximum_distance < 25.0:
            contrast_description = "low color contrast"
        elif maximum_distance < 55.0:
            contrast_description = "moderate color contrast"
        else:
            contrast_description = "high color contrast"

    neutral_share = float(normalized_weights[chroma < 12.0].sum())
    neutral_suffix = (
        " and a strong neutral presence" if neutral_share >= 0.4 else ""
    )
    return (
        f"{temperature_description}, {chroma_description}, "
        f"{lightness_description} palette with {contrast_description}"
        f"{neutral_suffix}"
    )


def extract_prevalent_color_outputs(
    image: torch.Tensor,
    color_count: int,
    prefix_hash: bool,
) -> tuple[str, torch.Tensor, str, str]:
    rgb_palette, palette_weights = _extract_prevalent_rgb_colors(
        image, color_count
    )
    return (
        _format_hex_palette(rgb_palette, prefix_hash),
        _render_palette_grid(rgb_palette),
        _nearest_color_names(rgb_palette),
        _describe_palette(rgb_palette, palette_weights),
    )


def extract_prevalent_hex_colors(
    image: torch.Tensor,
    color_count: int,
    prefix_hash: bool,
) -> str:
    palettes, _, _, _ = extract_prevalent_color_outputs(
        image, color_count, prefix_hash
    )
    return palettes
