"""Preprocessing transforms for BraTS volumes.

Two transform chains:
  - ``train_transforms``: includes augmentation. Used only in training DataLoader.
  - ``val_transforms``: deterministic. Used in validation, test, and inference.

Augmentation MUST NOT be applied to validation/test sets.
Preprocessing MUST NOT leak information across patients.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)

from braintumor.schemas.contracts import VolumeModality

KEYS = ["image", "label"]
IMAGE_KEYS = ["image"]

# BraTS-native labels for 4-class segmentation:
#   0 = background
#   1 = WT (necrosis + non-enhancing + enhancing)
#   2 = TC (necrosis + non-enhancing)
#   4 = ET (enhancing only)


def _select_keys(modalities: list[VolumeModality]) -> list[str]:
    """Map VolumeModality → string keys used by MONAI transforms."""
    return [m.value for m in modalities]


def train_transforms(
    modalities: list[VolumeModality],
    keys: list[str] = KEYS,
) -> Callable:
    """Augmented training pipeline.

    NOTE: Augmentation is applied **only** in this chain.
    Validation/test/inference must use ``val_transforms()``.
    """
    img_keys = _select_keys(modalities)
    return Compose(
        [
            LoadImaged(keys=img_keys + (["label"] if "label" in keys else [])),
            EnsureChannelFirstd(keys=img_keys + (["label"] if "label" in keys else [])),
            Orientationd(keys=img_keys + (["label"] if "label" in keys else []), axcodes="RAS"),
            ScaleIntensityRangePercentilesd(
                keys=img_keys, lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True
            ),
            RandFlipd(keys=img_keys + (["label"] if "label" in keys else []), prob=0.5, spatial_axis=0),
            RandRotate90d(keys=img_keys + (["label"] if "label" in keys else []), prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=img_keys, prob=0.15, mean=0.0, std=0.1),
            RandShiftIntensityd(keys=img_keys, offsets=0.1, prob=0.5),
            RandScaleIntensityd(keys=img_keys, factors=0.1, prob=0.5),
            RandAdjustContrastd(keys=img_keys, prob=0.3, gamma=(0.7, 1.5)),
            NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True),
        ]
    )


def val_transforms(
    modalities: list[VolumeModality],
    keys: list[str] = KEYS,
) -> Callable:
    """Deterministic validation/test/inference pipeline.

    No randomness. No augmentation. Same output for the same input.
    """
    img_keys = _select_keys(modalities)
    return Compose(
        [
            LoadImaged(keys=img_keys + (["label"] if "label" in keys else [])),
            EnsureChannelFirstd(keys=img_keys + (["label"] if "label" in keys else [])),
            Orientationd(keys=img_keys + (["label"] if "label" in keys else []), axcodes="RAS"),
            Spacingd(
                keys=img_keys + (["label"] if "label" in keys else []),
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear",) * len(img_keys) + ("nearest",),
            ),
            ScaleIntensityRangePercentilesd(
                keys=img_keys, lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True
            ),
            NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True),
        ]
    )


def brats_label_to_4class(label: np.ndarray) -> np.ndarray:
    """Convert BraTS native labels to 4-class training labels.

    BraTS uses {0, 1, 2, 4}. We remap to {0, 1, 2, 3} for contiguous classes.
    """
    out = np.zeros_like(label, dtype=np.uint8)
    out[label == 1] = 1  # NCR/NET → class 1
    out[label == 2] = 1  # ED     → class 1 (WT)
    out[label == 4] = 3  # ET     → class 3
    return out


def fourclass_to_brats(label: np.ndarray) -> np.ndarray:
    """Inverse of ``brats_label_to_4class``.

    Output uses BraTS-native labels {0, 1, 2, 4}. Used at inference time so
    downstream consumers (viewers, reports) see the standard BraTS encoding.
    """
    out = np.zeros_like(label, dtype=np.uint8)
    out[label == 1] = 1
    out[label == 3] = 4
    return out
