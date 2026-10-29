"""Evaluation script for the Brain Tumor segmentation model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from braintumor.config import config
from braintumor.data.brats_loader import discover_cases
from braintumor.data.dataset import BratsDataset
from braintumor.data.slicing import SlicingDataset
from braintumor.data.preprocessing import val_transforms
from braintumor.data.splits import discover_and_split
from braintumor.models.attention_unet_2d import AttentionUnet2D
from braintumor.eval.metrics import evaluate_case, aggregate_metrics
from braintumor.schemas.contracts import VolumeModality

def main():
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_root", type=str, default=str(config.data_root))
    parser.add_argument("--split_file", type=str, default=str(config.split_file))
    args = parser.parse_args()

    # 1. Setup
    root = Path(args.data_root)
    train, val, test = discover_and_split(root, split_file=Path(args.split_file))

    from braintumor.models.attention_unet_2d import MVP_MODALITIES
    modalities = list(MVP_MODALITIES)

    test_ds = BratsDataset(test, modalities, transform=val_transforms(modalities))
    test_loader = DataLoader(
        SlicingDataset(test_ds),
        batch_size=1,
        shuffle=False
    )

    # 2. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from braintumor.models.attention_unet_2d import build_attn_unet_2d
    from braintumor.registry import registry
    registry.register("attn_unet_2d", build_attn_unet_2d)

    model = registry.get("attn_unet_2d").model.to(device)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3. Evaluate
    print(f"Evaluating on {len(test)} cases...")
    all_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            target = y.cpu().numpy()[0]

            # Note: la_metrics expect BraTS regions.
            # Our model predicts 4-class {0, 1, 2, 3}.
            # We must convert it back to BraTS native {0, 1, 2, 4} using fourclass_to_brats.
            from braintumor.data.preprocessing import fourclass_to_brats
            pred_brats = fourclass_to_brats(pred)
            target_brats = fourclass_to_brats(target) # Assume target is 4-class

            # Mock spacing for 2D slice
            spacing = (1.0, 1.0, 1.0)

            # evaluate_case expects 3D. For 2D slices, we can wrap them in a dummy dimension.
            res = evaluate_case(
                pred_brats[np.newaxis, ...],
                target_brats[np.newaxis, ...],
                spacing
            )
            all_metrics.append(res)

    aggregated = aggregate_metrics(all_metrics)
    print("\nAggregate Metrics (Mean ± Std):")
    for region, m in aggregated.items():
        print(f"\nRegion {region}:")
        for metric, val in m.items():
            print(f"  {metric}: {val['mean']:.4f} ± {val['std']:.4f} (n={val['n']})")

if __name__ == "__main__":
    main()
