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
DATASET_PATH = join("data", "dataset.csv")

# Load the dataset
df = pd.read_csv(DATASET_PATH)

# Iterate through the dataset
for i, (p, q) in enumerate(zip(df['P(x)'], df['Q(x)']), start=1):
    for train in ['train_17', 'train_18']:
        if os.path.exists(join("output_dirs", f"{train}", f"thread_{i}")):
            print(f"[{train}] Thread {i} already exists, skipping...")
            continue
        subprocess.run(args=["python", f"{train}.py", p, q, str(i)])