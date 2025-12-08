"""
training.py

Script to train the TransformerUNet model using the dataset built in test/load_data.py
and the training utilities from train_eval.py.
"""

import os
from pathlib import Path
import torch
from torch.optim import Adam

from model.st_unet.st_unet import UNet3D
from transformer_cnn_model.model_architecture import TransformerUNet
from transformer_cnn_model.train_eval_functions.train_eval import training_unet, validation_unet
from transformer_cnn_model.preprocessing.load_data import build_dataloaders
from transformer_cnn_model.config import data_cfg, model_cfg, train_cfg


def main():
    # -----------------------
    # 1. Device configuration
    # -----------------------
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        pin_memory = True
    else:
        device = torch.device("cpu")
        print("Using CPU")
        pin_memory = False

    # -----------------------
    # 2. Build DataLoaders
    # -----------------------
    print("Building dataloaders (this may take a while the first time)...")

    data_cfg.cache_dir.mkdir(exist_ok=True)
    train_cfg.ckpt_dir.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=pin_memory,
        year_target=data_cfg.year_target,
        dir_folders=data_cfg.dir_folders,
        device="cpu",
        use_cache=data_cfg.use_cache,
        cache_dir=data_cfg.cache_dir,
    )

    # Peek at one batch to infer T
    x_sample, y_sample = next(iter(train_loader))
    B, T, H, W = x_sample.shape
    print(f"Sample batch shape: x={x_sample.shape}, y={y_sample.shape}")
    print(f"Detected T (time steps) = {T}")

    # -----------------------
    # 3. Instantiate the model
    # -----------------------
    if model_cfg.architecture == "transunet":
        model = TransformerUNet(
            n_channels=T,
            n_classes=model_cfg.n_classes,
            use_temporal_transformer=model_cfg.use_temporal_transformer,
            init_hid_dim=model_cfg.init_hid_dim,
            kernel_size=model_cfg.kernel_size,
            pooling=model_cfg.pooling,
            bilinear=model_cfg.bilinear,
            drop_channels=model_cfg.drop_channels,
            p_drop=model_cfg.p_drop,
            d_model=model_cfg.d_model,
            nhead=model_cfg.nhead,
            num_layers=model_cfg.num_layers,
            dim_feedforward=model_cfg.dim_feedforward,
            dropout=model_cfg.dropout,
        )
    elif model_cfg.architecture == "unet3d":
        model = UNet3D(
            n_channels=T,
            n_classes=model_cfg.n_classes,
            init_hid_dim=model_cfg.init_hid_dim,
            kernel_size=model_cfg.kernel_size,
            pooling=model_cfg.pooling,
            bilinear=model_cfg.bilinear,
            drop_channels=model_cfg.drop_channels,
            p_drop=model_cfg.p_drop,
        )
    else:
        raise ValueError(f"Unknown architecture: {model_cfg.architecture}")

    model.to(device)

    # -----------------------
    # 4. Optimizer
    # -----------------------
    optimizer = Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # -----------------------
    # 5. Training loop
    # -----------------------
    for epoch in range(1, train_cfg.num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{train_cfg.num_epochs} =====")
        ckpt_path = train_cfg.ckpt_dir / f"{model_cfg.architecture}_epoch{epoch:03d}.pt"

        train_losses = training_unet(
            model,
            train_loader,
            optimizer,
            nonwater=train_cfg.nonwater_label,
            water=train_cfg.water_label,
            pixel_size=train_cfg.pixel_size,
            water_threshold=train_cfg.water_threshold,
            device=str(device),
            loss_f=train_cfg.loss_f,
            physics=train_cfg.physics,
            verbose=True,
        )

        mean_train_loss = float(torch.tensor(train_losses).mean())
        print(f"Train loss: {mean_train_loss:.6f}")

        # ---- Validation ----
        val_losses, acc, prec, rec, f1, csi = validation_unet(
            model,
            val_loader,
            nonwater=0,
            water=1,
            device=str(device),
            loss_f="BCE",
            water_threshold=0.5,
        )

        mean_val_loss = float(torch.tensor(val_losses).mean())
        print(f"Val loss: {mean_val_loss:.6f}")
        print(
            f"Val metrics: "
            f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, csi={csi:.4f}"
        )

        # ---- Save checkpoint ----
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    # -----------------------
    # 6. (Optional) Test evaluation
    # -----------------------
    print("\nEvaluating on test set...")
    test_losses, acc, prec, rec, f1, csi = validation_unet(
        model,
        test_loader,
        nonwater=0,
        water=1,
        device=str(device),
        loss_f="BCE",
        water_threshold=0.5,
    )

    mean_test_loss = float(torch.tensor(test_losses).mean())
    print(f"Test loss: {mean_test_loss:.6f}")
    print(
        f"Test metrics: "
        f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, csi={csi:.4f}"
    )


if __name__ == "__main__":
    main()
