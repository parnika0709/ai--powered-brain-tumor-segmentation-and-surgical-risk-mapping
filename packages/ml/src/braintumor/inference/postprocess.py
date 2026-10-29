"""Post-processing of segmentation predictions.

Standard BraTS-style cleanup:
  - Largest connected component per class
  - Removal of tiny isolated components (< min_voxels)
  - Region merging: TC ⊆ WT, ET ⊆ TC
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from braintumor.models.attention_unet_2d import (
    BRATS_LABEL_BACKGROUND,
    BRATS_LABEL_ET,
    BRATS_LABEL_TC,
    BRATS_LABEL_WT,
)


def largest_connected_component(mask: np.ndarray, label: int) -> np.ndarray:
    """Keep only the largest connected component for ``label``."""
    binary = mask == label
    if not binary.any():
        return mask
    labeled, n = ndimage.label(binary)
    if n <= 1:
        return mask
    component_sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    largest_label = int(np.argmax(component_sizes)) + 1
    out = mask.copy()
    out[(labeled != 0) & (labeled != largest_label) & binary] = BRATS_LABEL_BACKGROUND
    return out


def remove_small_components(mask: np.ndarray, label: int, min_voxels: int = 10) -> np.ndarray:
    """Remove components of ``label`` smaller than ``min_voxels``."""
    binary = mask == label
    if not binary.any():
        return mask
    labeled, n = ndimage.label(binary)
    if n <= 1:
        return mask
    component_sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    out = mask.copy()
    for idx, size in enumerate(component_sizes, start=1):
        if size < min_voxels:
            out[(labeled == idx) & binary] = BRATS_LABEL_BACKGROUND
    return out


def enforce_brats_hierarchy(mask: np.ndarray) -> np.ndarray:
    """Ensure ET ⊆ TC ⊆ WT (BraTS native nesting).

    After argmax, ET voxels that are outside TC are demoted. Same for TC.
    """
    wt = (mask == BRATS_LABEL_WT) | (mask == BRATS_LABEL_TC) | (mask == BRATS_LABEL_ET)
    tc = (mask == BRATS_LABEL_TC) | (mask == BRATS_LABEL_ET)
    et = mask == BRATS_LABEL_ET

    out = np.zeros_like(mask)
    out[wt] = BRATS_LABEL_WT
    out[tc] = BRATS_LABEL_TC
    out[et] = BRATS_LABEL_ET
    return out


def postprocess(mask: np.ndarray, min_voxels: int = 10) -> np.ndarray:
    """Apply the full post-processing pipeline.

    Order:
      1. Remove tiny components per class.
      2. Keep largest connected component per class.
      3. Enforce ET ⊆ TC ⊆ WT nesting.
    """
    out = mask.copy()
    for label in (BRATS_LABEL_ET, BRATS_LABEL_TC, BRATS_LABEL_WT):
        out = remove_small_components(out, label, min_voxels)
    for label in (BRATS_LABEL_ET, BRATS_LABEL_TC, BRATS_LABEL_WT):
        out = largest_connected_component(out, label)
    out = enforce_brats_hierarchy(out)
    return out
