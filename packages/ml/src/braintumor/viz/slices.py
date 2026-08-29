"""Slice exporters for the 2D viewer.

Converts (D, H, W) arrays into PNG slices at axial/coronal/sagittal planes.
Used by the backend to serve preview images and by the report generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    """Robust percentile-based normalization to uint8."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def export_slice_pngs(
    volume: np.ndarray,
    mask: np.ndarray | None,
    output_dir: Path,
    *,
    indices: list[int] | None = None,
    modality_index: int = 0,
) -> list[Path]:
    """Export axial slice PNGs (one per requested index) with optional mask overlay.

    The overlay is a semi-transparent color per BraTS region.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if volume.ndim != 4:
        raise ValueError(f"Expected (C, D, H, W); got {volume.shape}")

    if indices is None:
        # Default: 8 evenly spaced slices through the depth.
        depth = volume.shape[1]
        indices = np.linspace(0, depth - 1, num=min(8, depth), dtype=int).tolist()

    out_paths: list[Path] = []
    base = volume[modality_index]  # (D, H, W)
    base_norm = normalize_uint8(base)

    rgb = np.stack([base_norm, base_norm, base_norm], axis=-1)  # (D, H, W, 3)

    if mask is not None:
        # Apply per-region color overlays.
        overlays = {
            1: np.array([255, 165, 60]),    # WT  - orange
            2: np.array([220, 30, 30]),     # TC  - dark red
            4: np.array([140, 0, 140]),     # ET  - purple
        }
        for label, color in overlays.items():
            region = mask == label
            if region.any():
                rgb[region] = (0.5 * rgb[region] + 0.5 * color).astype(np.uint8)

    for i in indices:
        slice_rgb = rgb[i]
        img = Image.fromarray(slice_rgb, mode="RGB")
        path = output_dir / f"slice_axial_{i:04d}.png"
        img.save(path)
        out_paths.append(path)

    return out_paths
