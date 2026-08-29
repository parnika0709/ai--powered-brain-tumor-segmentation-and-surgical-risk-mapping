"""Dataset wrapper to provide 2D slices from 3D volumes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any

from braintumor.data.dataset import BratsDataset
from braintumor.data.preprocessing import brats_label_to_4class

class SlicingDataset(Dataset):
    """Wraps a BratsDataset to yield 2D slices for 2D models.

    A 3D volume of shape (C, D, H, W) provides D slices of shape (C, H, W).
    """

    def __init__(
        self,
        base_dataset: BratsDataset,
        slice_idx: int | None = None, # If None, random slice per getitem
    ) -> None:
        self.base_dataset = base_dataset
        self.slice_idx = slice_idx

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.base_dataset[index]
        img_keys = [k for k in item.keys() if k != "label"]

        img_slices = []
        for k in img_keys:
            t = torch.as_tensor(item[k]).float()
            if t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            img_slices.append(t)

        img = torch.stack(img_slices, dim=0)

        lbl = torch.as_tensor(item["label"]).long()
        if lbl.ndim == 4 and lbl.shape[0] == 1:
            lbl = lbl.squeeze(0)

        if img.ndim != 4:
             raise ValueError(f"Expected img shape (C, D, H, W), got {img.shape}")

        depth = img.shape[1]
        z = self.slice_idx if self.slice_idx is not None else np.random.randint(0, depth)
        z = min(z, depth - 1)

        slice_img = img[:, z, :, :] # (C, H, W)
        slice_lbl = lbl[z, :, :]    # (H, W)

        slice_lbl_np = slice_lbl.numpy()
        mapped_lbl = brats_label_to_4class(slice_lbl_np)

        return {
            "image": slice_img,
            "label": torch.from_numpy(mapped_lbl).long()
        }
