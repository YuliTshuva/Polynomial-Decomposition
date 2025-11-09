"""
Yuli Tshuva
Write a pipeline that run the algorithms on the entire dataset.
"""

import os.path
from os.path import join
import pandas as pd
import subprocess
from functions import *

# Constants
DATASET_PATH = join("data", "dataset_100_5_3.csv")


# Load the dataset
df = pd.read_csv(DATASET_PATH)

# Set train file
train = "train_17"  # 18

# Iterate through the dataset
for i, (p, q) in enumerate(zip(df['P(x)'], df['Q(x)']), start=1):
    for j in range(3):
        if os.path.exists(join("output_dirs", f"{train}_{j_to_bool(0, j)}{j_to_bool(1, j)}{j_to_bool(2, j)}", f"thread_{i}")):
            print(f"[{train}_{train}_{j_to_bool(0, j)}{j_to_bool(1, j)}{j_to_bool(2, j)}] Thread {i} already exists, skipping...")
        else:
            subprocess.run(
                args=["python3", f"{train}.py", p, q, str(i), j_to_bool(0, j), j_to_bool(1, j), j_to_bool(2, j)], )
