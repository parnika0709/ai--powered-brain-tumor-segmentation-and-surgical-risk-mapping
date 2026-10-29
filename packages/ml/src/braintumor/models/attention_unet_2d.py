"""2D Attention U-Net for BraTS-style multi-modal MRI segmentation.

This is the MVP model. It is intentionally simple:
  - 3-channel input (T1ce, T2, FLAIR) — T1 dropped for redundancy.
  - 4-class output (background, WT, TC, ET).
  - MONAI's ``AttentionUnet`` building block for code economy.

Swap-in path: any subclass of ``Predictor`` (e.g., ``NNUnetPredictor``,
``SwinUNETRPredictor``) can register itself with the same name pattern.
The backend never depends on this class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet

from braintumor.schemas.contracts import (
    PredictionMeta,
    PredictionOutput,
    VolumeInput,
    VolumeModality,
)
from braintumor.registry.predictor import Predictor

# BraTS native label encoding:
#   0 = background
#   1 = WT (Whole Tumor)
#   2 = TC (Tumor Core)
#   4 = ET (Enhancing Tumor)
BRATS_LABEL_BACKGROUND = 0
BRATS_LABEL_WT = 1
BRATS_LABEL_TC = 2
BRATS_LABEL_ET = 4

# MVP input channels: T1ce, T2, FLAIR
MVP_MODALITIES: tuple[VolumeModality, ...] = (
    VolumeModality.T1CE,
    VolumeModality.T2,
    VolumeModality.FLAIR,
)


class AttentionUnet2D(nn.Module):
    """MONAI Attention U-Net with config-driven depth."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # MONAI's AttentionUnet requires equal-length channels/strides.
        if len(channels) - 1 != len(strides):
            raise ValueError("len(channels) must equal len(strides) + 1")

        self.model = AttentionUnet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AttentionUnet2DPredictor(Predictor):
    """Inference wrapper implementing the Predictor contract.

    Notes
    -----
    * The model is loaded once and held in memory.
    * ``predict()`` is CPU/GPU-agnostic; PyTorch handles device.
    * For MC-Dropout uncertainty, call ``enable_dropout()`` before
      repeated forward passes (see ``explainability``).
    """

    name = "attn_unet_2d"
    version = "0.1.0"
    description = "2D Attention U-Net (T1ce+T2+FLAIR → 4-class BraTS labels)"

    def __init__(
        self,
        weights_path: Path | None = None,
        device: str | None = None,
        eval_dropout_passes: int = 1,
        chunk_size: int = 16,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_dropout_passes = max(1, eval_dropout_passes)
        self.chunk_size = chunk_size

        self.model = AttentionUnet2D(in_channels=3, out_channels=4).to(self.device)
        self.model.eval()

        if weights_path is not None:
            self.load_weights(weights_path)

    # ------------------------------------------------------------------
    # Predictor interface
    # ------------------------------------------------------------------

    def load_weights(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)

    @torch.no_grad()
    def predict(self, volume: VolumeInput) -> PredictionOutput:
        import time

        start = time.perf_counter()

        x = self._to_tensor(volume)
        logits = self._forward(x)
        probs = torch.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1).to(torch.int8)

        elapsed = time.perf_counter() - start

        return PredictionOutput(
            mask=mask.cpu().numpy(),
            probability=probs.cpu().numpy().astype(np.float16),
            uncertainty=None,
            saliency=None,
            meta=PredictionMeta(
                model_name=self.name,
                model_version=self.version,
                inference_seconds=float(elapsed),
                notes=None,
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, volume: VolumeInput) -> torch.Tensor:
        if volume.array is None:
            raise ValueError("VolumeInput.array is None")

        arr = np.asarray(volume.array)
        if arr.ndim != 4:
            raise ValueError(
                f"Expected (C, D, H, W); got shape {arr.shape}"
            )

        # Select MVP channels by modality.
        name_to_idx = {m.value: i for i, m in enumerate(volume.modalities)}
        missing = [m.value for m in MVP_MODALITIES if m.value not in name_to_idx]
        if missing:
            raise ValueError(
                f"Volume missing required modalities: {missing}. "
                f"Provided: {[m.value for m in volume.modalities]}"
            )
        channel_idx = [name_to_idx[m.value] for m in MVP_MODALITIES]
        arr = arr[channel_idx, ...]  # (3, D, H, W)

        # Per-channel z-score (computed on non-zero voxels).
        for c in range(arr.shape[0]):
            channel = arr[c]
            mask = channel > 0
            if mask.any():
                mean = channel[mask].mean()
                std = channel[mask].std() + 1e-8
                arr[c] = (channel - mean) / std

        tensor = torch.from_numpy(arr.astype(np.float32))
        return tensor.to(self.device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model slice-by-slice, in chunks.

        x is (C, D, H, W). We permute to (D, C, H, W) and slide.
        """
        # (D, C, H, W)
        slices = x.permute(1, 0, 2, 3)
        depth = slices.shape[0]
        outputs: list[torch.Tensor] = []

        for start in range(0, depth, self.chunk_size):
            end = min(start + self.chunk_size, depth)
            batch = slices[start:end]
            out = self.model(batch)
            outputs.append(out)

        return torch.cat(outputs, dim=0).permute(1, 0, 2, 3)  # back to (C, D, H, W)


def build_attn_unet_2d() -> AttentionUnet2DPredictor:
    """Factory for registry use."""
    return AttentionUnet2DPredictor()
