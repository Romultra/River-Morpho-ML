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

    ## ------ All adjustable parameters for loading, training, validation are in this block ------
    # Loading parameters
    batch_size = 16          # batch size
    num_workers = 12        # number of DataLoader workers
    year_target = 5         # target year for prediction
    use_cache = True        # whether to use cached data

    # Data directories
    dir_folders = "data/satellite/dataset_month3"         # March dataset
    cache_dir = Path("transformer_cnn_model/cache")       # folder for cached tensors
    cache_dir.mkdir(exist_ok=True)                        # safe even if it already exists

    # Directory to save checkpoints
    ckpt_dir = Path("transformer_cnn_model/checkpoints")     # checkpoint directory (trained model epochs saved here)
    ckpt_dir.mkdir(exist_ok=True)                            # safe even if it already exists
    # checkpoint path of each epoch saved in 5. Training loop. To modify filename layout, change in that section.

    # Training and validation parameters
    lr = 1e-4   # learning rate
    weight_decay = 1e-5      # weight decay for optimizer
    num_epochs = 100         # number of training epochs
    nonwater = 0             # label values
    water = 1                # label values
    pixel_size = 60          # size of one pixel in meters
    water_threshold = 0.5    # threshold for water classification from model output probabilities
    loss_f = "BCE"           # loss function: 'BCE', 'Focal', etc.
    physics = False          # whether to use physics-based loss additions
    verbose = True           # whether to print batch-wise training progress
    ## -------------------------------------------------------------------------------------------

    # -----------------------
    # 2. Build DataLoaders
    # -----------------------
    # NOTE: data is always loaded on CPU; we move batches to GPU in the training loop
    print("Building dataloaders (this may take a while the first time)...")

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=batch_size,           
        num_workers=num_workers,
        pin_memory=pin_memory,
        year_target=year_target,          # still 4-in / 1-out
        dir_folders=dir_folders,
        device="cpu",           # datasets stay on CPU; batches moved to GPU in train_eval
        use_cache=use_cache,         # <--- enable caching
        cache_dir=cache_dir,    # <--- tell it where to store/load .pt files
    )

    # Peek at one batch to infer T (number of time steps)
    x_sample, y_sample = next(iter(train_loader))
    B, T, H, W = x_sample.shape
    print(f"Sample batch shape: x={x_sample.shape}, y={y_sample.shape}")
    print(f"Detected T (time steps) = {T}")

    # -----------------------
    # 3. Instantiate the model
    # -----------------------
    # # Using TransformerUNet with temporal transformer
    # model = TransformerUNet(
    #     n_channels=T,   # number of temporal input frames (years)
    #     n_classes=1,    # binary water prediction
    #     use_temporal_transformer=True,  # set to True to enable temporal transformer
    #     # other hyperparameters use defaults from architecture.py
    # )
    # model.to(device) 

    # Alternatively, use UNet3D without transformer:
    model = UNet3D(
        n_channels=T,
        n_classes=1,
        init_hid_dim=8,      
        kernel_size=3,
        pooling='max',
        bilinear=False,
        drop_channels=False,
        p_drop=None,
    )
    model.to(device)

    # print(model)

    # -----------------------
    # 4. Optimizer
    # -----------------------
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # -----------------------
    # 5. Training loop
    # -----------------------
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        ckpt_path = ckpt_dir / f"unet3d_epoch{epoch:03d}.pt"     # checkpoint path of each epoch

        # ---- Training ----
        train_losses = training_unet(
            model,
            train_loader,
            optimizer,
            nonwater=0,
            water=1,
            pixel_size=60,
            water_threshold=0.5,
            device=str(device),   # training_unet expects a string like 'cuda:0' or 'cpu'
            loss_f="BCE",         # you can try 'Focal' etc if desired
            physics=False,        # set True if you want physics-based loss additions
            verbose=True,         # print batch-wise training progress
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
