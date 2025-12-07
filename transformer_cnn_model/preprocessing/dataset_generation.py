# This module contains the functions used for the dataset generation
# needed as input and target for the deep-learning model

import os
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from osgeo import gdal
gdal.UseExceptions()

from torch.utils.data import TensorDataset

from preprocessing.satellite_analysis_pre import count_pixels
from preprocessing.satellite_analysis_pre import load_avg


def load_image_array(path, scaled_classes=True):
    """
    Load a single image using GDAL library. Convert and return it into a numpy array
    with dtype = np.float32.
    It is implemented and tested to work with JRC collection exported in grayscale
    (i.e., one channel with pixel values 0, 1, and 2).

    It can also scale the original pixel values by setting the new classes as follows:
            - no-data:   -1
            - non-water: 0
            - water:     1

    Inputs:
        path : str
            Full path of the image to be loaded.
        scaled_classes : bool
            Whether pixel classes are scaled to [-1, 1] (recommended) or kept
            in the original [0, 2] range.

    Output:
        img_array : np.ndarray
            2D array representing the loaded image.
    """
    img = gdal.Open(path)
    img_array = img.ReadAsArray().astype(np.float32)

    if scaled_classes:
        img_array = img_array.astype(int)
        img_array[img_array == 0] = -1
        img_array[img_array == 1] = 0
        img_array[img_array == 2] = 1

    return img_array


def create_dir_list(
    train_val_test,
    dir_folders="data/satellite/dataset",   
    collection="JRC_GSW1_4_MonthlyHistory",
):
    """
    Get list of paths of training, validation and testing datasets.

    Inputs:
        train_val_test : {"training", "validation", "testing"}
        dir_folders    : str
            Directory where folders are stored. Default: "data/satellite/dataset".
        collection     : str
            Satellite image collection. Default: "JRC_GSW1_4_MonthlyHistory".

    Output:
        list_dir : list[str]
            Paths of training, validation and testing dataset folders.
    """
    list_dir = []

    for item in os.listdir(dir_folders):
        full_path = os.path.join(dir_folders, item)
        if os.path.isdir(full_path):
            if (train_val_test in item) and (collection in item):
                list_dir.append(full_path)

    # sort by reach id
    list_dir.sort(key=lambda x: int(x.split(f"_{train_val_test}_r")[-1]))
    return list_dir


def create_list_images(
    train_val_test,
    reach,
    dir_folders="data/satellite/dataset",  
    collection="JRC_GSW1_4_MonthlyHistory",
):
    """
    Return the paths of the satellite images present within a folder
    given use and reach.

    Inputs:
        train_val_test : {"training", "validation", "testing"}
        reach          : int
            Reach number.
        dir_folders    : str
        collection     : str

    Output:
        list_dir_images : list[str]
            Path for each image of the dataset.
    """
    folder = os.path.join(str(dir_folders), f"{collection}_{train_val_test}_r{reach}")
    list_dir_images = []

    for image in os.listdir(folder):
        if image.endswith(".tif"):
            path_image = os.path.join(folder, image)
            list_dir_images.append(path_image)

    # Ensure deterministic ordering across OS/filesystems
    list_dir_images.sort()
    return list_dir_images


def create_datasets(
    train_val_test,
    reach,
    year_target=5,
    nodata_value=-1,
    dir_folders="data/satellite/dataset",
    collection="JRC_GSW1_4_MonthlyHistory",
    scaled_classes=True,
    max_year_with_avg=2021,   # <--- new parameter (default 2021)
):
    """
    Create the input and target dataset for each specific use and reach.
    Return two lists of lists.

    Inputs:
        train_val_test : {"training", "validation", "testing"}
        reach          : int
        year_target    : int
        nodata_value   : int
        dir_folders    : str
        collection     : str
        scaled_classes : bool
        max_year_with_avg : int
            Last year for which average CSVs exist. Images beyond this year
            are ignored to avoid FileNotFoundError.

    Outputs:
        input_dataset, target_dataset : list[list[np.ndarray]]
    """
    # list of image paths
    list_dir_images = create_list_images(train_val_test, reach, dir_folders, collection)

    # load all images as arrays
    images_array = [
        load_image_array(list_dir_images[i], scaled_classes=scaled_classes)
        for i in range(len(list_dir_images))
    ]

    # ---- NEW: cap the years to those that actually have averages ----
    # Assume first image is 1988, then 1989, ..., as in the original code
    start_year = 1988
    all_years = list(range(start_year, start_year + len(images_array)))

    # Keep only years <= max_year_with_avg, and truncate images_array accordingly
    valid_indices = [i for i, y in enumerate(all_years) if y <= max_year_with_avg]

    if not valid_indices:
        # No usable years with averages -> return empty datasets
        return [], []

    max_idx = max(valid_indices) + 1  # slice is exclusive
    images_array = images_array[:max_idx]
    years = all_years[:max_idx]
    # -----------------------------------------------------------------

    # load season averages only for the valid years
    avg_imgs = [
        load_avg(
            train_val_test,
            reach,
            year,
            dir_averages="data/satellite/averages",
        )
        for year in years
    ]

    # replace missing data - images are now binary
    good_images_array = [
        np.where(image == nodata_value, avg_imgs[i], image)
        for i, image in enumerate(images_array)
    ]

    input_dataset = []
    target_dataset = []

    # build n-to-1 sequences
    for i in range(len(good_images_array) - year_target):
        input_dataset.append(good_images_array[i : i + year_target - 1])
        target_dataset.append([good_images_array[i + year_target - 1]])

    return input_dataset, target_dataset



def combine_datasets(
    train_val_test,
    reach,
    year_target=5,
    nonwater_threshold=480000,
    nodata_value=-1,
    nonwater_value=0,
    dir_folders="data/satellite/dataset",
    collection="JRC_GSW1_4_MonthlyHistory",
    scaled_classes=True,
):
    """
    Filter image combinations based on `non-water` threshold.
    """
    input_dataset, target_dataset = create_datasets(
        train_val_test,
        reach,
        year_target,
        nodata_value,
        dir_folders,
        collection,
        scaled_classes,
    )

    filtered_input_dataset, filtered_target_dataset = [], []

    for input_images, target_image in zip(input_dataset, target_dataset):
        input_combs = []
        for img in input_images:
            nonwater_ok = count_pixels(img, nonwater_value) < nonwater_threshold
            input_combs.append(nonwater_ok)

        if all(input_combs):
            target_nonwater_ok = (
                count_pixels(target_image[0], nonwater_value) < nonwater_threshold
            )
            if target_nonwater_ok:
                input_tensor = [img for img in input_images]
                target_tensor = target_image[0]

                filtered_input_dataset.append(input_tensor)
                filtered_target_dataset.append(target_tensor)

    return filtered_input_dataset, filtered_target_dataset


def create_full_dataset(
    train_val_test,
    year_target=5,
    nonwater_threshold=480000,
    nodata_value=-1,
    nonwater_value=0,
    dir_folders="data/satellite/dataset",
    collection="JRC_GSW1_4_MonthlyHistory",
    scaled_classes=True,
    device="cpu",                 # <-- safer default on Ubuntu
    dtype=torch.float32,          # <-- better for CNN/Transformer
):
    """
    Generate the full dataset for the given split, combining all reaches.
    """
    stacked_dict = {"input": [], "target": []}

    for folder in os.listdir(dir_folders):
        if train_val_test in folder:
            reach_id = folder.split("_r", 1)[1]
            inputs, target = combine_datasets(
                train_val_test,
                int(reach_id),
                year_target,
                nonwater_threshold,
                nodata_value,
                nonwater_value,
                dir_folders,
                collection,
                scaled_classes,
            )
            stacked_dict["input"].extend(inputs)
            stacked_dict["target"].extend(target)

    if dtype is None:
        input_tensor = torch.tensor(stacked_dict["input"], device=device)
        target_tensor = torch.tensor(stacked_dict["target"], device=device)
    else:
        input_tensor = torch.tensor(stacked_dict["input"], dtype=dtype, device=device)
        target_tensor = torch.tensor(stacked_dict["target"], dtype=dtype, device=device)

    dataset = TensorDataset(input_tensor, target_tensor)
    return dataset


# ----------------------------------------- #
# TEMPORAL SPLIT                           #
# ----------------------------------------- #


def split_list(
    train_val_test,
    reach,
    month,
    year_end_train=2009,
    year_end_val=2015,
    dir_folders="data/satellite",             
    collection="JRC_GSW1_4_MonthlyHistory",
):
    """
    Split the image list into training, validation and testing sub-lists.
    """
    if month not in {1, 2, 3, 4}:
        raise Exception(
            f"The specified month is {month}, which is not allowed. "
            "It can be either 1, 2, 3 or 4."
        )

    dir_dataset = os.path.join(dir_folders, f"dataset_month{month}")
    all_paths = create_list_images(
        train_val_test, reach, dir_folders=dir_dataset, collection=collection
    )

    train_list, val_list, test_list = [], [], []

    for path in all_paths:
        # Ubuntu-friendly year extraction (no '\\')
        filename = os.path.basename(path)
        year = int(filename.split("_")[0])

        if year <= year_end_train:
            train_list.append(path)
        elif year_end_train < year <= year_end_val:
            val_list.append(path)
        elif year > year_end_val:
            test_list.append(path)

    return train_list, val_list, test_list


def create_split_datasets(
    train_val_test,
    reach,
    month,
    use_dataset,
    year_end_train=2009,
    year_end_val=2015,
    year_target=5,
    nodata_value=-1,
    dir_folders="data/satellite",
    collection="JRC_GSW1_4_MonthlyHistory",
    scaled_classes=True,
):
    """
    Creates the input and target datasets for temporal splitting.
    """
    train_list, val_list, test_list = split_list(
        train_val_test,
        reach,
        month,
        year_end_train,
        year_end_val,
        dir_folders,
        collection,
    )

    if use_dataset == "training":
        images_train = [load_image_array(p, scaled_classes=scaled_classes)
                        for p in train_list]
        years_train = [int(os.path.basename(p).split("_")[0]) for p in train_list]
        avg_train = [
            load_avg(train_val_test, reach, year, dir_averages="data/satellite/averages")
            for year in years_train
        ]
        good_images_train = [
            np.where(img == nodata_value, avg_train[i], img)
            for i, img in enumerate(images_train)
        ]
        input_train, target_train = [], []
        for i in range(len(good_images_train) - year_target):
            input_train.append(good_images_train[i : i + year_target - 1])
            target_train.append([good_images_train[i + year_target - 1]])
        return [input_train, target_train]

    elif use_dataset == "validation":
        images_val = [load_image_array(p, scaled_classes=scaled_classes)
                      for p in val_list]
        years_val = [int(os.path.basename(p).split("_")[0]) for p in val_list]
        avg_val = [
            load_avg(train_val_test, reach, year, dir_averages="data/satellite/averages")
            for year in years_val
        ]
        good_images_val = [
            np.where(img == nodata_value, avg_val[i], img)
            for i, img in enumerate(images_val)
        ]
        input_val, target_val = [], []
        for i in range(len(good_images_val) - year_target):
            input_val.append(good_images_val[i : i + year_target - 1])
            target_val.append([good_images_val[i + year_target - 1]])
        return [input_val, target_val]

    elif use_dataset == "testing":
        images_test = [load_image_array(p, scaled_classes=scaled_classes)
                       for p in test_list]
        years_test = [int(os.path.basename(p).split("_")[0]) for p in test_list]
        avg_test = [
            load_avg(train_val_test, reach, year, dir_averages="data/satellite/averages")
            for year in years_test
        ]
        good_images_test = [
            np.where(img == nodata_value, avg_test[i], img)
            for i, img in enumerate(images_test)
        ]
        input_test, target_test = [], []
        for i in range(len(good_images_test) - year_target):
            input_test.append(good_images_test[i : i + year_target - 1])
            target_test.append([good_images_test[i + year_target - 1]])
        return [input_test, target_test]

    else:
        raise Exception(
            f"The given use_dataset is {use_dataset} but is wrong. "
            "The possible choices are `training`, `validation`, `testing`."
        )


def combine_split_datasets(
    train_val_test,
    reach,
    month,
    use_dataset,
    year_end_train=2009,
    year_end_val=2015,
    year_target=5,
    nonwater_threshold=480000,
    nodata_value=-1,
    nonwater_value=0,
    dir_folders="data/satellite",
    collection="JRC_GSW1_4_MonthlyHistory",
    scaled_classes=True,
):
    """
    Filter temporal split dataset based on non-water threshold.
    """
    dataset = create_split_datasets(
        train_val_test,
        reach,
        month,
        use_dataset,
        year_end_train,
        year_end_val,
        year_target,
        nodata_value,
        dir_folders,
        collection,
        scaled_classes,
    )

    input_dataset, target_dataset = dataset[0], dataset[1]
    filtered_inputs, filtered_targets = [], []

    for input_images, target_image in zip(input_dataset, target_dataset):
        input_combs = []
        for img in input_images:
            nonwater_ok = count_pixels(img, nonwater_value) < nonwater_threshold
            input_combs.append(nonwater_ok)

        if all(input_combs):
            target_nonwater_ok = (
                count_pixels(target_image[0], nonwater_value) < nonwater_threshold
            )
            if target_nonwater_ok:
                input_tensor = [img for img in input_images]
                target_tensor = target_image[0]
                filtered_inputs.append(input_tensor)
                filtered_targets.append(target_tensor)

    return [filtered_inputs, filtered_targets]


def create_split_dataset(
    month,
    use_dataset,
    year_target=5,
    year_end_train=2009,
    year_end_val=2015,
    nonwater_threshold=480000,
    nodata_value=-1,
    nonwater_value=0,
    dir_folders="data/satellite",
    collection="JRC_GSW1_4_MonthlyHistory",
    scaled_classes=True,
    device="cpu",                 # <-- safer default on Ubuntu
    dtype=torch.float32,          # <-- better for CNN/Transformer
):
    """
    Generate the full temporal-split dataset for the given use, combining all reaches.
    """
    train_val_test = ["training", "validation", "testing"]
    dir_dataset = os.path.join(dir_folders, f"dataset_month{month}")
    stacked_dataset = {"input": [], "target": []}

    for folder in os.listdir(dir_dataset):
        for use in train_val_test:
            if use in folder:
                reach_id = folder.split("_r", 1)[1]
                filtered_dataset = combine_split_datasets(
                    use,
                    reach_id,
                    month,
                    use_dataset,
                    year_end_train,
                    year_end_val,
                    year_target,
                    nonwater_threshold,
                    nodata_value,
                    nonwater_value,
                    dir_folders,
                    collection,
                    scaled_classes,
                )

                stacked_dataset["input"].extend(filtered_dataset[0])
                stacked_dataset["target"].extend(filtered_dataset[1])

    inputs = torch.tensor(stacked_dataset["input"], dtype=dtype, device=device)
    targets = torch.tensor(stacked_dataset["target"], dtype=dtype, device=device)
    dataset = TensorDataset(inputs, targets)
    return dataset
