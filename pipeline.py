"""
Yuli Tshuva
Write a pipeline that run the algorithms on the entire dataset.
"""
import os.path
# Imports
from os.path import join
import pandas as pd
import subprocess

# Constants
DATASET_PATH = join("data", "dataset_300_vary.csv")

# Load the dataset
df = pd.read_csv(DATASET_PATH)

# Set train file
train = "train_17"  # 18

# Iterate through the dataset
for i, (p, q) in enumerate(zip(df['P(x)'], df['Q(x)']), start=1):
    if os.path.exists(join("output_dirs", f"{train}", f"thread_{i}")):
        print(f"[{train}] Thread {i} already exists, skipping...")
        continue
    subprocess.run(args=["python3", f"{train}.py", p, q, str(i), "0", "1", "1"])
