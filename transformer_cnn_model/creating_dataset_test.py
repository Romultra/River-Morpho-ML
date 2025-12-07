from transformer_cnn_model.preprocessing.dataset_generation import split_list, create_split_dataset
import torch
import os

# 1. Check the years in each split for one reach
train_list, val_list, test_list = split_list("training", reach=5, month=3)
get_year = lambda p: int(os.path.basename(p).split("_")[0])

print("TRAIN years:", [get_year(p) for p in train_list[:10]])
print("VAL years:",   [get_year(p) for p in val_list])
print("TEST years:",  [get_year(p) for p in test_list])

# 2. Inspect one sample from your training dataset
train_ds = create_split_dataset(month=3, use_dataset="training")
x, y = train_ds[0]
print("input shape:", x.shape, "target shape:", y.shape)
print("input unique:", torch.unique(x))
print("target unique:", torch.unique(y))
