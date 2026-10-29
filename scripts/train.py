"""Main training script for the Brain Tumor segmentation model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from braintumor.config import config, MLConfig
from braintumor.data.brats_loader import discover_cases
from braintumor.data.dataset import BratsDataset
from braintumor.data.slicing import SlicingDataset
from braintumor.data.preprocessing import train_transforms, val_transforms
from braintumor.data.splits import discover_and_split
from braintumor.data.synthetic import setup_synthetic_dataset
from braintumor.models.attention_unet_2d import AttentionUnet2D
from braintumor.training.train import fit, TrainConfig

def main():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Segmentation Model")
    parser.add_argument("--data_root", type=str, default=str(config.data_root))
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for verification")
    args = parser.parse_args()

    # 1. Data Setup
    root = Path(args.data_root)
    if args.synthetic:
        print("Using synthetic dataset for verification...")
        setup_synthetic_dataset(root, n_cases=20)

    try:
        train_cases, val_cases, _ = discover_and_split(root)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    modalities = [config.model_channels[0] == 3 and (
        # In our case, we use T1ce, T2, FLAIR as per AttentionUnet2D.MVP_MODALITIES
        # This should be imported from the model or defined centrally.
        # For now, we'll use the ones defined in AttentionUnet2D.
        # Let's import them.
        None
    )]
    # Correction: use the modalities required by the model.
    from braintumor.models.attention_unet_2d import MVP_MODALITIES
    modalities = list(MVP_MODALITIES)

    # 2. Dataset & Loaders
    train_ds = BratsDataset(train_cases, modalities, transform=train_transforms(modalities))
    val_ds = BratsDataset(val_cases, modalities, transform=val_transforms(modalities))

    train_loader = DataLoader(
        SlicingDataset(train_ds),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        SlicingDataset(val_ds),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # 3. Model
    model = AttentionUnet2D(
        in_channels=config.in_channels,
        out_channels=config.out_channels
    )

    # 4. Train
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=config.checkpoint_dir
    )

    print(f"Starting training for {args.epochs} epochs...")
    history = fit(model, train_loader, val_loader, train_cfg)
    print("Training complete.")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()
