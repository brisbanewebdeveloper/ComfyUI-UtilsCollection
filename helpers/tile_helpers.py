import math

import torch
import torch.nn.functional as F


TILE_LAYOUT_FORMAT_VERSION = 1
TILE_MODES = ("tile_size", "grid")
TILE_MASK_PROFILES = ("cosine", "linear")
TILE_DIFFERENTIAL_DIFFUSION_MODES = ("off", "core", "advanced")


def prepare_depth_structure_map(
    depth_map,
    target_height,
    target_width,
    device,
    dtype,
):
    if depth_map is None:
        return None
    if not torch.is_tensor(depth_map) or depth_map.ndim != 4:
        raise ValueError("Depth map must be one BHWC IMAGE tensor.")
    if depth_map.shape[0] != 1:
        raise ValueError("Depth map must contain exactly one image, not a batch.")
    channels = depth_map.shape[-1]
    if channels not in (1, 3, 4):
        raise ValueError("Depth map must have one, three, or four channels.")

    depth_map = depth_map.to(device=device, dtype=dtype)
    if channels == 1:
        depth = depth_map[..., 0]
    else:
        rgb = depth_map[..., :3]
        weights = torch.tensor(
            (0.2126, 0.7152, 0.0722),
            device=device,
            dtype=dtype,
        )
        depth = torch.sum(rgb * weights, dim=-1)
    if not torch.isfinite(depth).all():
        raise ValueError("Depth map contains non-finite values.")

    depth = depth.clamp(0.0, 1.0)
    if depth.shape[1:] != (target_height, target_width):
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        depth = depth.clamp(0.0, 1.0)
    return depth


def apply_depth_structure_mask(mask, depth, influence):
    influence = float(influence)
    if not -1.0 <= influence <= 1.0:
        raise ValueError("Depth influence must be between -1.0 and 1.0.")
    if depth is None:
        return mask
    if depth.shape != mask.shape:
        raise ValueError("Depth tile does not match its denoise mask.")
    selected_depth = depth if influence >= 0.0 else 1.0 - depth
    depth_factor = (1.0 - abs(influence) * selected_depth).clamp(0.0, 1.0)
    return mask * depth_factor


def apply_tile_differential_diffusion(
    model,
    mode,
    value,
):
    if mode not in TILE_DIFFERENTIAL_DIFFUSION_MODES:
        raise ValueError(f"Unsupported tile Differential Diffusion mode: {mode}.")
    if mode == "off":
        return model
    if model is None:
        raise ValueError(
            "Connect a model when tile Differential Diffusion is enabled."
        )
    if mode == "core":
        strength = float(value)
        if not 0.0 <= strength <= 1.0:
            raise ValueError(
                "Core Differential Diffusion value must be between 0.0 and 1.0."
            )
        from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion

        return DifferentialDiffusion.execute(model, strength).args[0]

    multiplier = float(value)
    if multiplier == 0.0:
        raise ValueError(
            "Advanced Differential Diffusion multiplier cannot be zero."
        )

    patched_model = model.clone()

    def advanced_mask(sigma, denoise_mask, extra_options):
        sampling = extra_options["model"].inner_model.model_sampling
        step_sigmas = extra_options["sigmas"]
        sigma_to = sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]
        timestep_from = sampling.timestep(sigma_from)
        timestep_to = sampling.timestep(sigma_to)
        current_timestep = sampling.timestep(sigma[0])
        threshold = (
            (current_timestep - timestep_to)
            / (timestep_from - timestep_to)
            / multiplier
        )
        return (denoise_mask >= threshold).to(denoise_mask.dtype)

    patched_model.set_model_denoise_mask_function(advanced_mask)
    return patched_model


def _validate_common_tiling_inputs(
    image,
    tile_mode,
    tile_width,
    tile_height,
    rows,
    columns,
    overlap,
    mask_profile,
    feather_width,
    mask_strength,
):
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("Tiled sampling requires one BHWC IMAGE tensor.")
    if image.shape[0] != 1:
        raise ValueError("Tiled sampling accepts exactly one image, not an image batch.")
    if image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError("Tiled sampling requires positive image dimensions.")
    if tile_mode not in TILE_MODES:
        raise ValueError(f"Unsupported tile mode: {tile_mode}.")
    if mask_profile not in TILE_MASK_PROFILES:
        raise ValueError(f"Unsupported tile mask profile: {mask_profile}.")
    if overlap < 0:
        raise ValueError("Tile overlap cannot be negative.")
    if not 0.0 <= feather_width <= 1.0:
        raise ValueError("Mask feather width must be between 0.0 and 1.0.")
    if not 0.0 <= mask_strength <= 1.0:
        raise ValueError("Mask strength must be between 0.0 and 1.0.")
    if tile_mode == "tile_size":
        if tile_width < 1 or tile_height < 1:
            raise ValueError("Tile width and height must be positive.")
        if overlap >= tile_width or overlap >= tile_height:
            raise ValueError("Tile overlap must be smaller than tile width and height.")
    else:
        if rows < 1 or columns < 1:
            raise ValueError("Tile rows and columns must be positive.")


def _fixed_axis_ranges(length, tile_size, overlap):
    if tile_size >= length:
        return [(0, length)]
    stride = tile_size - overlap
    ranges = []
    start = 0
    while start < length:
        end = min(start + tile_size, length)
        ranges.append((start, end))
        if end == length:
            break
        start += stride
    return ranges


def _grid_axis_ranges(length, count, overlap):
    edges = [round(index * length / count) for index in range(count + 1)]
    left_halo = overlap // 2
    right_halo = overlap - left_halo
    ranges = []
    for index in range(count):
        start = edges[index]
        end = edges[index + 1]
        if index > 0:
            start = max(0, start - left_halo)
        if index < count - 1:
            end = min(length, end + right_halo)
        if end <= start:
            raise ValueError(
                "Tile grid is too dense for this image; reduce rows or columns."
            )
        ranges.append((start, end))
    return ranges


def _axis_neighbor_overlaps(ranges, index):
    start, end = ranges[index]
    left = max(0, ranges[index - 1][1] - start) if index > 0 else 0
    right = max(0, end - ranges[index + 1][0]) if index + 1 < len(ranges) else 0
    return left, right


def build_tile_records(
    height,
    width,
    tile_mode,
    tile_width,
    tile_height,
    rows,
    columns,
    overlap,
):
    if tile_mode == "tile_size":
        x_ranges = _fixed_axis_ranges(width, tile_width, overlap)
        y_ranges = _fixed_axis_ranges(height, tile_height, overlap)
    else:
        if rows > height or columns > width:
            raise ValueError(
                "Tile rows and columns cannot exceed the source pixel dimensions."
            )
        min_cell_width = min(
            round((index + 1) * width / columns) - round(index * width / columns)
            for index in range(columns)
        )
        min_cell_height = min(
            round((index + 1) * height / rows) - round(index * height / rows)
            for index in range(rows)
        )
        if (columns > 1 and overlap >= min_cell_width) or (
            rows > 1 and overlap >= min_cell_height
        ):
            raise ValueError(
                "Tile overlap must be smaller than every grid cell dimension."
            )
        x_ranges = _grid_axis_ranges(width, columns, overlap)
        y_ranges = _grid_axis_ranges(height, rows, overlap)

    records = []
    for row, (y0, y1) in enumerate(y_ranges):
        top_overlap, bottom_overlap = _axis_neighbor_overlaps(y_ranges, row)
        for column, (x0, x1) in enumerate(x_ranges):
            left_overlap, right_overlap = _axis_neighbor_overlaps(x_ranges, column)
            records.append(
                {
                    "index": len(records),
                    "row": row,
                    "column": column,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "left_overlap": left_overlap,
                    "right_overlap": right_overlap,
                    "top_overlap": top_overlap,
                    "bottom_overlap": bottom_overlap,
                }
            )
    return records, len(y_ranges), len(x_ranges)


def _profile_ramp(length, low, high, profile, device, dtype):
    if length <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    unit = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
    if profile == "cosine":
        unit = 0.5 - 0.5 * torch.cos(unit * math.pi)
    return low + (high - low) * unit


def tile_weight_mask(record, profile, feather_width, strength, device, dtype):
    height = int(record["height"])
    width = int(record["width"])
    floor = 1.0 - float(strength)
    y_weight = torch.ones(height, device=device, dtype=dtype)
    x_weight = torch.ones(width, device=device, dtype=dtype)

    side_specs = (
        (x_weight, int(record["left_overlap"]), True),
        (x_weight, int(record["right_overlap"]), False),
        (y_weight, int(record["top_overlap"]), True),
        (y_weight, int(record["bottom_overlap"]), False),
    )
    for axis_weight, actual_overlap, rising in side_specs:
        if actual_overlap <= 0 or strength <= 0.0:
            continue
        feather = min(
            actual_overlap,
            max(1, round(actual_overlap * float(feather_width))),
        )
        ramp = _profile_ramp(
            feather,
            floor if rising else 1.0,
            1.0 if rising else floor,
            profile,
            device,
            dtype,
        )
        if rising:
            axis_weight[:feather] *= ramp
        else:
            axis_weight[-feather:] *= ramp

    return y_weight[:, None] * x_weight[None, :]


def _vae_compression(vae):
    compression = vae.spacial_compression_encode()
    if isinstance(compression, bool):
        raise ValueError("The connected VAE returned an invalid spatial compression.")
    try:
        compression = int(compression)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The connected VAE does not expose a valid spatial compression."
        ) from exc
    if compression < 1:
        raise ValueError("The connected VAE returned an invalid spatial compression.")
    return compression


def _pad_tile_for_vae(tile, compression):
    height, width = tile.shape[1:3]
    padded_height = math.ceil(height / compression) * compression
    padded_width = math.ceil(width / compression) * compression
    pad_bottom = padded_height - height
    pad_right = padded_width - width
    if pad_bottom == 0 and pad_right == 0:
        return tile, pad_bottom, pad_right
    padded = F.pad(
        tile.movedim(-1, 1),
        (0, pad_right, 0, pad_bottom),
        mode="replicate",
    ).movedim(1, -1)
    return padded, pad_bottom, pad_right


def split_and_encode_tiles(
    image,
    vae,
    tile_mode,
    tile_width,
    tile_height,
    rows,
    columns,
    overlap,
    mask_profile,
    feather_width,
    mask_strength,
    depth_map=None,
    depth_influence=1.0,
):
    _validate_common_tiling_inputs(
        image,
        tile_mode,
        tile_width,
        tile_height,
        rows,
        columns,
        overlap,
        mask_profile,
        feather_width,
        mask_strength,
    )
    if vae is None:
        raise ValueError("Connect a VAE to encode the image tiles.")

    height, width = image.shape[1:3]
    prepared_depth = prepare_depth_structure_map(
        depth_map,
        height,
        width,
        image.device,
        image.dtype,
    )
    records, actual_rows, actual_columns = build_tile_records(
        height,
        width,
        tile_mode,
        tile_width,
        tile_height,
        rows,
        columns,
        overlap,
    )
    compression = _vae_compression(vae)
    image_tiles = []
    latent_tiles = []

    for record in records:
        tile = image[
            :,
            record["y0"]:record["y1"],
            record["x0"]:record["x1"],
            :,
        ]
        padded_tile, pad_bottom, pad_right = _pad_tile_for_vae(tile, compression)
        samples = vae.encode(padded_tile)
        if (
            not torch.is_tensor(samples)
            or samples.ndim not in (4, 5)
            or samples.shape[0] != 1
            or (samples.ndim == 5 and samples.shape[2] != 1)
        ):
            raise ValueError(
                "The VAE must return one spatial latent per image tile."
            )

        mask = tile_weight_mask(
            record,
            mask_profile,
            feather_width,
            mask_strength,
            device=padded_tile.device,
            dtype=padded_tile.dtype,
        )
        if prepared_depth is not None:
            depth_tile = prepared_depth[
                :,
                record["y0"]:record["y1"],
                record["x0"]:record["x1"],
            ][0]
            mask = apply_depth_structure_mask(
                mask,
                depth_tile,
                depth_influence,
            )
        if pad_bottom or pad_right:
            mask = F.pad(mask, (0, pad_right, 0, pad_bottom), value=0.0)
        record["pad_bottom"] = pad_bottom
        record["pad_right"] = pad_right
        record["encoded_width"] = padded_tile.shape[2]
        record["encoded_height"] = padded_tile.shape[1]
        image_tiles.append(padded_tile)
        latent_tiles.append(
            {
                "samples": samples,
                "noise_mask": mask.unsqueeze(0).unsqueeze(0),
            }
        )

    layout = {
        "format": "UC_HIGH_RES_TILE_LAYOUT",
        "version": TILE_LAYOUT_FORMAT_VERSION,
        "original_height": height,
        "original_width": width,
        "channels": image.shape[-1],
        "tile_mode": tile_mode,
        "rows": actual_rows,
        "columns": actual_columns,
        "overlap": overlap,
        "mask_profile": mask_profile,
        "feather_width": float(feather_width),
        "mask_strength": float(mask_strength),
        "depth_structure": prepared_depth is not None,
        "depth_influence": float(depth_influence),
        "vae_compression": compression,
        "tiles": records,
    }
    return image_tiles, latent_tiles, layout


def _unwrap_layout(layout_values):
    if isinstance(layout_values, list):
        non_null = [value for value in layout_values if value is not None]
        if not non_null:
            raise ValueError("Tile layout metadata is missing.")
        layout = non_null[0]
        if any(value != layout for value in non_null[1:]):
            raise ValueError("Tile accumulator received conflicting layout metadata.")
        return layout
    return layout_values


def validate_tile_layout(layout):
    if not isinstance(layout, dict):
        raise ValueError("Tile layout metadata is invalid.")
    if layout.get("format") != "UC_HIGH_RES_TILE_LAYOUT":
        raise ValueError("Tile layout metadata has an unsupported type.")
    if layout.get("version") != TILE_LAYOUT_FORMAT_VERSION:
        raise ValueError("Tile layout metadata has an unsupported version.")
    records = layout.get("tiles")
    if not isinstance(records, list) or not records:
        raise ValueError("Tile layout metadata contains no tiles.")
    return records


def accumulate_tile_images(image_values, layout_values):
    layout = _unwrap_layout(layout_values)
    records = validate_tile_layout(layout)
    images = [image for image in image_values if image is not None]
    if len(images) != len(records):
        raise ValueError(
            f"Tile accumulator expected {len(records)} images but received {len(images)}."
        )

    first = images[0]
    if not torch.is_tensor(first) or first.ndim != 4 or first.shape[0] != 1:
        raise ValueError("Every accumulated tile must be one BHWC IMAGE tensor.")
    device = first.device
    dtype = first.dtype
    channels = first.shape[-1]
    canvas = torch.zeros(
        (1, layout["original_height"], layout["original_width"], channels),
        device=device,
        dtype=dtype,
    )
    weights = torch.zeros(
        (1, layout["original_height"], layout["original_width"], 1),
        device=device,
        dtype=dtype,
    )

    for image, record in zip(images, records):
        if (
            not torch.is_tensor(image)
            or image.ndim != 4
            or image.shape[0] != 1
        ):
            raise ValueError("Every accumulated tile must be one BHWC IMAGE tensor.")
        if image.device != device or image.dtype != dtype:
            raise ValueError("All accumulated tiles must share a device and dtype.")
        if image.shape[-1] != channels:
            raise ValueError("All accumulated tiles must have matching channels.")
        real_height = int(record["height"])
        real_width = int(record["width"])
        if image.shape[1] < real_height or image.shape[2] < real_width:
            raise ValueError(
                f"Decoded tile {record['index']} is smaller than its recorded region."
            )

        cropped = image[:, :real_height, :real_width, :]
        mask = tile_weight_mask(
            record,
            layout["mask_profile"],
            layout["feather_width"],
            layout["mask_strength"],
            device=device,
            dtype=dtype,
        ).unsqueeze(0).unsqueeze(-1)
        y0, y1 = record["y0"], record["y1"]
        x0, x1 = record["x0"], record["x1"]
        canvas[:, y0:y1, x0:x1, :] += cropped * mask
        weights[:, y0:y1, x0:x1, :] += mask

    if torch.any(weights <= 0):
        raise ValueError("Tile layout left uncovered output pixels.")
    return canvas / weights
