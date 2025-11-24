"""
load_data.py

Utility functions to build PyTorch Datasets and DataLoaders for the JamUNet /
UNet3D models from the preprocessed satellite images.

It relies on the original dataset_generation.py functions and keeps the same
spatial split as in Magherini (2024):                      
- training: 28 upstream reaches
- validation: 1 intermediate reach
- testing: 1 most downstream reach

Outputs:
    - train_loader, val_loader, test_loader: ready to be passed to architecture.UNet3D
"""

from pathlib import Path
from typing import Optional, Union
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformer_cnn_model.preprocessing.dataset_generation import create_full_dataset


def build_dataset(
    split: str,
    year_target: int = 5,
    nonwater_threshold: int = 480_000,
    nodata_value: int = -1,
    nonwater_value: int = 0,
    dir_folders: str = r"data/satellite/dataset_month3",
    collection: str = r"JRC_GSW1_4_MonthlyHistory",
    scaled_classes: bool = True,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """
    Build a TensorDataset for a given split ("training", "validation", "testing")
    using the same logic as in the thesis and dataset_generation.py.

    The resulting dataset has:
        inputs:  shape (N, T, H, W)
        targets: shape (N, H, W)

    These are ready to be fed into UNet3D, which expects x of shape (B, T, H, W).

    Parameters
    ----------
    split : {"training", "validation", "testing"}
        Which dataset split to load.
    year_target : int
        Number of consecutive years in each sequence (4 inputs + 1 target if 5).
    nonwater_threshold : int
        Minimum allowed number of non-water pixels to keep an input-target pair.
    nodata_value : int
        Value used for "no data" pixels in the scaled classes.
    nonwater_value : int
        Value used for "non-water" pixels in the scaled classes.
    dir_folders : str
        Root folder where the reach subfolders are stored.
    collection : str
        Name of the satellite image collection.
    scaled_classes : bool
        Whether pixel values are in [-1, 0, 1] or original [0, 1, 2].
    dtype : torch.dtype
        Data type of the returned tensors (float32 recommended for CNN/Transformer).
    device : str
        Device where tensors are created ("cpu" recommended; move to GPU in training loop).

    Returns
    -------
    torch.utils.data.TensorDataset
        Dataset with (inputs, targets).
    """
    dataset = create_full_dataset(
        train_val_test=split,
        year_target=year_target,
        nonwater_threshold=nonwater_threshold,
        nodata_value=nodata_value,
        nonwater_value=nonwater_value,
        dir_folders=dir_folders,
        collection=collection,
        scaled_classes=scaled_classes,
        device=device,
        dtype=dtype,
    )
    return dataset


def get_or_create_dataset(
    split: str,
    dir_folders: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.int64,
    cache_dir: Optional[Union[str, Path]] = None,
    cache_prefix: str = "month3",
    use_cache: bool = False,
):

    """
    Returns a TensorDataset for the given split, optionally using on-disk caching. 
    Arguments of build_dataloaders() does not affect caching. Caching only removes the need to rebuild from raw data, 
    i.e. create_full_dataset(). Thus .tif and .csv files are only read and tensors are only built once.

    split: 'training', 'validation', or 'testing'
    dir_folders: root folder for the satellite data (e.g. 'data/satellite/dataset_month3')
    device: device for the initial tensor creation (should be 'cpu' for caching)
    dtype: tensor dtype (e.g. torch.int64 or torch.float32)
    cache_dir: directory where cached tensors are stored
    cache_prefix: prefix for cache filenames (e.g. 'month3')
    use_cache: if True, try to load from cache; if missing, build and then save
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f"{cache_prefix}_{split}.pt"
    else:
        cache_path = None

    # Try to load from cache
    if use_cache and cache_path is not None and cache_path.exists():
        print(f"[CACHE] Loading {split} dataset from {cache_path}")
        inputs, targets = torch.load(cache_path, map_location="cpu")
        dataset = TensorDataset(inputs.to(device), targets.to(device))
        return dataset

    # Otherwise, build from raw data
    print(f"[CACHE] Building {split} dataset from raw data (this may take a while)...")
    dataset = create_full_dataset(
        train_val_test=split,
        dir_folders=dir_folders,
        device=device,
        dtype=dtype,
        # year_target, nonwater_threshold, etc. use their defaults here
    )

    # Save to cache (on CPU)
    if use_cache and cache_path is not None:
        print(f"[CACHE] Saving {split} dataset to {cache_path}")
        inputs, targets = dataset.tensors
        # Ensure CPU before saving
        inputs = inputs.to("cpu")
        targets = targets.to("cpu")
        torch.save((inputs, targets), cache_path)

    return dataset


def build_dataloaders(
    batch_size: int = 2,
    num_workers: int = 0,
    pin_memory: bool = False,
    year_target: int = 5,
    dir_folders: str = r"data/satellite/dataset_month3",
    collection: str = r"JRC_GSW1_4_MonthlyHistory",
    device: str = "cpu",
    use_cache: bool = False,
    cache_dir: Union[str, Path] = "cache",
):


    """
    Build train/val/test DataLoaders for UNet3D using the original spatial split.

    Parameters
    ----------
    batch_size : int
        Batch size for all three loaders.
    num_workers : int
        Number of workers for the DataLoaders (set >0 if your system supports it).
    pin_memory : bool
        Whether to pin memory for faster GPU transfers.
    year_target : int
        Number of consecutive years (4 inputs + 1 target if 5).
    dir_folders : str
        Root folder with dataset subdirectories.
    collection : str
        Satellite image collection name.
    device : str
        Device for the underlying tensors ("cpu" recommended).

    Returns
    -------
    (train_loader, val_loader, test_loader)
        Three DataLoaders ready for training / evaluation.
    """

    # Ensure cache directory exists
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    # Use folder name as cache prefix (e.g. 'dataset_month3')
    cache_prefix = Path(dir_folders).name

    # Build TensorDatasets (N, T, H, W) and (N, H, W), using caching if enabled
    train_dataset = get_or_create_dataset(
        split="training",
        dir_folders=dir_folders,
        device="cpu",               # build/cache on CPU
        dtype=torch.float32,        # good for CNN/Transformer
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
        use_cache=use_cache,
    )

    val_dataset = get_or_create_dataset(
        split="validation",
        dir_folders=dir_folders,
        device="cpu",
        dtype=torch.float32,
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
        use_cache=use_cache,
    )

    test_dataset = get_or_create_dataset(
        split="testing",
        dir_folders=dir_folders,
        device="cpu",
        dtype=torch.float32,
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
        use_cache=use_cache,
    )

    # Wrap them into DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check: try to build loaders and print shapes of one batch
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        device="cpu",     # create tensors on CPU; move to GPU in the training loop
        use_cache=True,  # True if you want to test caching here too
        cache_dir="transformer_cnn_model/cache",
    )


    batch = next(iter(train_loader))
    x, y = batch  # x: (B, T, H, W), y: (B, H, W)
    print("Train batch input shape :", x.shape)
    print("Train batch target shape:", y.shape)
