import os
from os.path import join

for dataset in ["100_5_3", "300_vary"]:
    dir_path = join("output_dirs", dataset)
    reg_dir = join(dir_path, "train_17")
    ab1_dir = join(dir_path, "train_17_011")
    ab2_dir = join(dir_path, "train_17_101")
    ab3_dir = join(dir_path, "train_17_110")
    for trail in os.listdir(reg_dir):
        reg_file = join(reg_dir, trail, "polynomials.txt")
        ab1_file = join(ab1_dir, trail, "polynomials.txt")
        ab2_file = join(ab2_dir, trail, "polynomials.txt")
        ab3_file = join(ab3_dir, trail, "polynomials.txt")
        with open(reg_file, "r") as f:
            reg_lines = float(f.readlines()[6].split("Loss = ")[1].strip()) < 1
        with open(ab1_file, "r") as f:
            ab1_lines = float(f.readlines()[6].split("Loss = ")[1].strip()) < 1
        with open(ab2_file, "r") as f:
            ab2_lines = float(f.readlines()[6].split("Loss = ")[1].strip()) < 1
        with open(ab3_file, "r") as f:
            ab3_lines = float(f.readlines()[6].split("Loss = ")[1].strip()) < 1
        if not reg_lines:
            if ab1_lines:
                print(f"In dataset: {dataset}, Thread: {trail}, the full method failed but the version without guessing coefficients succeeded.")
            if ab2_lines:
                print(f"In dataset: {dataset}, Thread: {trail}, the full method failed but the version without regularization succeeded.")
            if ab3_lines:
                print(f"In dataset: {dataset}, Thread: {trail}, the full method failed but the version without rounding coefficients succeeded.")
