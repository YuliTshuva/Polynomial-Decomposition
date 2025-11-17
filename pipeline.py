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
DATASET_PATH = join("data", "dataset_hybrid_200_deg15.csv")


# Load the dataset
df = pd.read_csv(DATASET_PATH)

# Set train file
train = "train_17"  # 18
mode = "hybrid"

# Create the working directory if it doesn't exist
WORKING_DIR = join("output_dirs", "hybrid_200", train)
os.makedirs(WORKING_DIR, exist_ok=True)

if "ablation" == mode:
    # Iterate through the dataset
    for i, (p, q) in enumerate(zip(df['P(x)'], df['Q(x)']), start=1):
        for j in range(3):
            if os.path.exists(join("output_dirs", f"{train}_{j_to_bool(0, j)}{j_to_bool(1, j)}{j_to_bool(2, j)}", f"thread_{i}")):
                print(f"[{train}_{j_to_bool(0, j)}{j_to_bool(1, j)}{j_to_bool(2, j)}] Thread {i} already exists, skipping...")
            else:
                subprocess.run(
                    args=["python", f"{train}.py", p, q, str(i), j_to_bool(0, j), j_to_bool(1, j), j_to_bool(2, j)], )

if mode == "hybrid":
    df["Decomposable"] = df["Decomposable"].astype(int)
    # Iterate through the dataset
    for i in range(1, len(df) + 1):
        if os.path.exists(join(WORKING_DIR, f"thread_{i}")):
            print(f"[{train}_hybrid] Thread {i} already exists, skipping...")
        else:
            if df.loc[i - 1, "Decomposable"] == 1:
                p = df.loc[i - 1, "P(x)"]
                q = df.loc[i - 1, "Q(x)"]
                args = ["python3", f"{train}.py", p, q, str(i), "1", "1", "1"]
                args += ["10", "0.001", "300", "1", "1", "1000", "4000"]
                args += ["output_dirs/hybrid_200/train_17"]
                args += ["None"]
            else:
                args = ["python3", f"{train}.py", "5", "3", str(i), "1", "1", "1"]
                args += ["10", "0.001", "300", "1", "1", "1000", "4000"]
                args += ["output_dirs/hybrid_200/train_17"]
                args += [df.loc[i - 1, "R(x)"]]

            subprocess.run(args=args)