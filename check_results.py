"""
Yuli Tshuva
Check the results of the algorithm.
"""

# Imports
import matplotlib.pyplot as plt
from os.path import join
import os
from matplotlib import rcParams
import numpy as np

# Constants
WORKING_DIR = "output_dirs"
rcParams['font.family'] = 'Times New Roman'

# Sum the successes
successes1, successes2, total_successes = 0, 0, 0
scale_successes1, scale_successes2 = {}, {}
for thread in os.listdir(join(WORKING_DIR, "train_18")):
    if int(thread.split("_")[1]) > 60:
        continue
    thread_dir1 = join(WORKING_DIR, "train_17", thread)
    thread_dir2 = join(WORKING_DIR, "train_18", thread)
    with open(join(thread_dir1, "polynomials.txt"), "r") as f:
        loss1 = float(f.readlines()[6].split("Loss = ")[1].strip())
    with open(join(thread_dir2, "polynomials.txt"), "r") as f:
        loss2 = float(f.readlines()[6].split("Loss = ")[1].strip())

    # Calculate the scale
    scale = ((int(thread.split("_")[1]) - 1) // 3 + 1) * 10
    if scale not in scale_successes1:
        scale_successes1[scale] = 0
    if scale not in scale_successes2:
        scale_successes2[scale] = 0

    # Check if the model succeeded
    if loss1 < 1e-5:
        successes1 += 1
        scale_successes1[scale] += 1
    if loss2 < 1e-5:
        successes2 += 1
        scale_successes2[scale] += 1
    if loss1 < 1e-5 or loss2 < 1e-5:
        total_successes += 1

plt.figure(figsize=(8, 5))
plt.title(f"Successes per scale (total of {total_successes} ({successes1}|{successes2}))", fontsize=20)
xs = np.array(list(scale_successes1.keys()))
ys1, ys2 = list(scale_successes1.values()), list(scale_successes2.values())
plt.bar(xs-1.5, ys1, color="royalblue", width=3, edgecolor="black", label="Train 17")
plt.bar(xs+1.5, ys2, color="hotpink", width=3, edgecolor="black", label="Train 18")
plt.xticks(list(scale_successes1.keys()), rotation=45)
plt.yticks([0, 1, 2, 3])
plt.xlabel("Scale", fontsize=15)
plt.ylabel("Number of successes", fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig(join("plots", "successes_per_scale.png"))
plt.show()
