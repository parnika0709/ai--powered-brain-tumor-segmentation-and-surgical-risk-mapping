"""Computational feature extraction from segmentation masks.

Computes per-BraTS-region features:
  - Volume (mm³)
  - Voxel count
  - Centroid (mm, in physical space)
  - Axis-aligned bounding box (mm)
  - Surface area (mm²) via marching cubes
  - Sphericity (shape descriptor)
  - Principal axes (mm, PCA)
  - Intensity statistics over the corresponding voxels in T1ce

No clinical interpretation is performed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure

from braintumor.models.attention_unet_2d import (
    BRATS_LABEL_ET,
    BRATS_LABEL_TC,
    BRATS_LABEL_WT,
)
from braintumor.schemas.contracts import (
    BoundingBox,
    Centroid,
    IntensityStats,
    ShapeFeatures,
    TumorFeatures,
    TumorLabel,
    VolumeInput,
    VolumeModality,
)

LABEL_TO_ENUM = {
    BRATS_LABEL_WT: TumorLabel.WT,
    BRATS_LABEL_TC: TumorLabel.TC,
    BRATS_LABEL_ET: TumorLabel.ET,
}


def _volume_mm3(voxel_count: int, spacing: tuple[float, float, float]) -> float:
    return float(voxel_count * spacing[0] * spacing[1] * spacing[2])


def _centroid_mm(mask: np.ndarray, spacing: tuple[float, float, float]) -> Centroid | None:
    if not mask.any():
        return None
    coords = np.argwhere(mask).astype(np.float64)
    weighted = coords.mean(axis=0)
    return Centroid(
        d=float(weighted[0] * spacing[0]),
        h=float(weighted[1] * spacing[1]),
        w=float(weighted[2] * spacing[2]),
    )


def _bbox_mm(mask: np.ndarray, spacing: tuple[float, float, float]) -> BoundingBox | None:
    if not mask.any():
        return None
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return BoundingBox(
        min_d=float(mins[0] * spacing[0]),
        max_d=float(maxs[0] * spacing[0]),
        min_h=float(mins[1] * spacing[1]),
        max_h=float(maxs[1] * spacing[1]),
        min_w=float(mins[2] * spacing[2]),
        max_w=float(maxs[2] * spacing[2]),
    )


def _principal_axes_mm(
    mask: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Approximate principal axis lengths via PCA on voxel coordinates."""
    if mask.sum() < 3:
        return (0.0, 0.0, 0.0)
    coords = np.argwhere(mask).astype(np.float64) * np.asarray(spacing)
    coords -= coords.mean(axis=0)
    cov = np.cov(coords, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    # Length ≈ 4 * sqrt(eigenvalue) for uniform ellipsoid.
    return tuple(float(4.0 * np.sqrt(max(ev, 0.0))) for ev in sorted(eigenvalues, reverse=True))


def _surface_area_mm2(
    mask: np.ndarray, spacing: tuple[float, float, float]
) -> float:
    """Surface area via marching cubes."""
    if mask.sum() < 6:
        return 0.0
    try:
        verts, faces, _normals, _values = measure.marching_cubes(
            mask.astype(np.float32), level=0.5, spacing=spacing
        )
        if len(faces) == 0:
            return 0.0
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        return float(area.sum())
    except (ValueError, RuntimeError):
        return 0.0


def _sphericity(volume_mm3: float, surface_area_mm2: float) -> float:
    """Sphericity = π^(1/3) * (6V)^(2/3) / A. 1.0 = perfect sphere."""
    if volume_mm3 <= 0 or surface_area_mm2 <= 0:
        return 0.0
    return float((np.pi ** (1.0 / 3.0)) * ((6.0 * volume_mm3) ** (2.0 / 3.0)) / surface_area_mm2)


def _intensity_stats(
    mask: np.ndarray, modality: np.ndarray | None
) -> IntensityStats | None:
    if modality is None or not mask.any():
        return None
    vals = modality[mask]
    if vals.size == 0:
        return None
    return IntensityStats(
        mean=float(vals.mean()),
        std=float(vals.std()),
        median=float(np.median(vals)),
        min=float(vals.min()),
        max=float(vals.max()),
    )


def _select_modality(volume: VolumeInput, target: VolumeModality) -> np.ndarray | None:
    if target not in volume.modalities:
        return None
    idx = volume.modalities.index(target)
    arr = np.asarray(volume.array)
    if arr.ndim != 4:
        return None
    return arr[idx]


def extract_features(
    mask: np.ndarray,
    volume: VolumeInput,
) -> "TumorFeaturesBundle":
    """Compute the full feature bundle for a prediction."""
    from braintumor.schemas.contracts import TumorFeaturesBundle

    spacing = volume.spacing
    t1ce = _select_modality(volume, VolumeModality.T1CE)

    features: list[TumorFeatures] = []
    for label_int, tumor_enum in LABEL_TO_ENUM.items():
        binary = mask == label_int
        voxel_count = int(binary.sum())
        if voxel_count == 0:
            continue

        volume_mm3 = _volume_mm3(voxel_count, spacing)
        bbox = _bbox_mm(binary, spacing)
        centroid = _centroid_mm(binary, spacing)
        axes = _principal_axes_mm(binary, spacing)
        surface = _surface_area_mm2(binary, spacing)
        sphericity = _sphericity(volume_mm3, surface)
        intensity = _intensity_stats(binary, t1ce)

        features.append(
            TumorFeatures(
                label=tumor_enum,
                volume_mm3=volume_mm3,
                voxel_count=voxel_count,
                centroid=centroid,
                bbox=bbox,
                shape=ShapeFeatures(
                    surface_area_mm2=surface,
                    sphericity=sphericity,
                    principal_axes_mm=axes,  # type: ignore[arg-type]
                ),
                intensity=intensity,
            )
        )

    return TumorFeaturesBundle(features=features)
