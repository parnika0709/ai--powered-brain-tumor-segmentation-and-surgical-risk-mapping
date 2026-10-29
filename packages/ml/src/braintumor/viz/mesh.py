"""3D mesh generation using PyVista/VTK.

Produces GLB-format meshes for the frontend ``@react-three/fiber`` viewer.

Outputs:
  - Tumor mesh (per BraTS region or combined)
  - Brain surface mesh (from FLAIR or T1)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

from braintumor.models.attention_unet_2d import (
    BRATS_LABEL_ET,
    BRATS_LABEL_TC,
    BRATS_LABEL_WT,
)

# Colors per region (RGB, 0–255).
REGION_COLORS = {
    BRATS_LABEL_WT: (0.93, 0.49, 0.20),    # orange
    BRATS_LABEL_TC: (0.85, 0.10, 0.10),    # dark red
    BRATS_LABEL_ET: (0.55, 0.00, 0.55),    # purple
}


@dataclass
class MeshResult:
    """Mesh file paths produced by ``generate_meshes``."""

    brain_path: Path
    tumor_paths: dict[int, Path]  # region → file path


def _smooth(mesh: pv.PolyData, iterations: int = 50) -> pv.PolyData:
    """Light smoothing — keeps detail without obliterating features."""
    try:
        return mesh.smooth(n_iter=iterations, relaxation_factor=0.1)
    except Exception:
        return mesh


def _marching_mesh(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    level: float = 0.5,
    smoothing: bool = True,
) -> pv.PolyData | None:
    """Run marching cubes on a binary volume."""
    if not volume.any():
        return None
    try:
        # PyVista's contour expects the array already scaled by spacing.
        grid = pv.wrap(volume.astype(np.float32))
        # Apply spacing via the grid's spacing attribute.
        grid.spacing = spacing
        mesh = grid.contour(isosurfaces=[level], scalars=mesh_or_array(grid))
        if mesh.n_points == 0:
            return None
        return _smooth(mesh) if smoothing else mesh
    except Exception:
        return None


def mesh_or_array(grid: pv.DataSet) -> np.ndarray:
    """PyVista's marching_cubes signatures differ between versions; this
    helper preserves compatibility by always passing an explicit scalars
    array name."""
    # Newer PyVista uses the active scalars by default. We just need a
    # valid scalar array — name it 'values' and assign if needed.
    if "values" not in grid.array_names:
        grid["values"] = np.asarray(grid).flatten()
    return grid["values"]


def generate_tumor_meshes(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    output_dir: Path,
    regions: tuple[int, ...] = (BRATS_LABEL_WT, BRATS_LABEL_TC, BRATS_LABEL_ET),
) -> dict[int, Path]:
    """Generate one GLB mesh per requested BraTS region.

    Returns a mapping ``{region_label: glb_path}``. Regions with zero voxels
    are skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, Path] = {}

    for region in regions:
        binary = (mask == region).astype(np.uint8)
        mesh = _marching_mesh(binary, spacing=spacing, level=0.5)
        if mesh is None:
            continue

        # Translate so the mesh is centered (visual nicety, not physical).
        mesh.translate(-np.asarray(mesh.center), inplace=True)

        color = REGION_COLORS.get(region, (0.5, 0.5, 0.5))
        path = output_dir / f"tumor_region_{region}.glb"
        # PyVista's GLB export expects RGB per point/face. Assign color array.
        mesh.point_data["color"] = np.tile(np.asarray(color), (mesh.n_points, 1))
        try:
            mesh.save(str(path))
            out[region] = path
        except Exception:
            # Fall back to PLY if GLB writer is unavailable.
            fallback = path.with_suffix(".ply")
            mesh.save(str(fallback))
            out[region] = fallback

    return out


def generate_brain_mesh(
    modality: np.ndarray,
    spacing: tuple[float, float, float],
    output_dir: Path,
    intensity_percentile: float = 30.0,
) -> Path | None:
    """Generate a brain surface mesh from a structural modality.

    The intensity percentile acts as a rough threshold for "brain tissue"
    vs. background/air. Robust because we never assume specific units.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Robust threshold via percentile.
    threshold = float(np.percentile(modality[modality > 0], intensity_percentile))
    binary = (modality > threshold).astype(np.uint8)

    # Fill small holes.
    from scipy import ndimage

    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)

    mesh = _marching_mesh(binary, spacing=spacing, level=0.5)
    if mesh is None:
        return None

    mesh.translate(-np.asarray(mesh.center), inplace=True)
    # Subtle gray brain color.
    mesh.point_data["color"] = np.tile(np.array([0.85, 0.85, 0.92]), (mesh.n_points, 1))

    path = output_dir / "brain_surface.glb"
    try:
        mesh.save(str(path))
        return path
    except Exception:
        fallback = path.with_suffix(".ply")
        mesh.save(str(fallback))
        return fallback


def generate_meshes(
    mask: np.ndarray,
    modality: np.ndarray,
    spacing: tuple[float, float, float],
    output_dir: Path,
) -> MeshResult:
    """Generate the full set of meshes for a case."""
    brain_path = generate_brain_mesh(modality, spacing=spacing, output_dir=output_dir)
    tumor_paths = generate_tumor_meshes(mask, spacing=spacing, output_dir=output_dir)

    return MeshResult(
        brain_path=brain_path if brain_path else Path(""),
        tumor_paths=tumor_paths,
    )
