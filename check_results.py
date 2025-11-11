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
rcParams['font.family'] = 'Times New Roman'


def analyze_for_100_5_3():
    """
    combinations = [[3, 5]]
    scales = range(10, 201, 10)
    repetitions = 5
    """
    # Sum the successes
    successes1, successes2, total_successes = 0, 0, 0
    scale_successes1, scale_successes2, scale_successes = {}, {}, {}
    repetitions = 5

    # The working directory
    WORKING_DIR = join("output_dirs", "100_5_3")

    for thread in os.listdir(join(WORKING_DIR, "train_18")):
        thread_dir1 = join(WORKING_DIR, "train_17", thread)
        thread_dir2 = join(WORKING_DIR, "train_18", thread)
        with open(join(thread_dir1, "polynomials.txt"), "r") as f:
            loss1 = float(f.readlines()[6].split("Loss = ")[1].strip())
        with open(join(thread_dir2, "polynomials.txt"), "r") as f:
            loss2 = float(f.readlines()[6].split("Loss = ")[1].strip())

        # Calculate the scale
        scale = ((int(thread.split("_")[1]) - 1) // repetitions + 1) * 10
        if scale not in scale_successes1:
            scale_successes1[scale] = 0
        if scale not in scale_successes2:
            scale_successes2[scale] = 0
        if scale not in scale_successes:
            scale_successes[scale] = 0

        # Check if the model succeeded
        if loss1 < 1e-5:
            successes1 += 1
            scale_successes1[scale] += 1
        if loss2 < 1e-5:
            successes2 += 1
            scale_successes2[scale] += 1
        if loss1 < 1e-5 or loss2 < 1e-5:
            total_successes += 1
            scale_successes[scale] += 1

    plt.figure(figsize=(8, 5))
    plt.title(f"Successes per scale (total of {total_successes} ({successes1}|{successes2}))", fontsize=20)
    xs = np.array(list(scale_successes1.keys()))
    ys1, ys2 = list(scale_successes1.values()), list(scale_successes2.values())
    width, space = 2, 2
    plt.bar(xs, ys1, color="royalblue", width=width, edgecolor="black", label="Integer case")
    plt.bar(xs - space, ys2, color="hotpink", width=width, edgecolor="black", label="Normalized case")
    plt.bar(xs + space, list(scale_successes.values()), color="turquoise", width=width, edgecolor="black",
            label="Total")
    plt.xticks(list(scale_successes1.keys()), rotation=45)
    plt.yticks(range(repetitions + 1))
    plt.xlabel("Scale", fontsize=15)
    plt.ylabel("Number of successes", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join("plots", "successes_per_scale_100_5_3.png"))
    plt.show()


def analyze_for_300_vary():
    """
    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 8]]
    scales = range(10, 201, 10)
    repetitions = 3
    """
    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 8]]

    # Modify the working directory
    global WORKING_DIR
    WORKING_DIR = join(WORKING_DIR, "300_vary")

    for i in range(len(combinations)):
        # Sum the successes
        successes1, successes2, total_successes = 0, 0, 0
        scale_successes1, scale_successes2, scale_successes = {}, {}, {}
        repetitions = 3
        for thread in os.listdir(join(WORKING_DIR, "train_18")):
            if int(thread.split("_")[1]) > 60 * (i + 1) or int(thread.split("_")[1]) <= 60 * i:
                continue
            thread_dir1 = join(WORKING_DIR, "train_17", thread)
            thread_dir2 = join(WORKING_DIR, "train_18", thread)
            with open(join(thread_dir1, "polynomials.txt"), "r") as f:
                loss1 = float(f.readlines()[6].split("Loss = ")[1].strip())
            with open(join(thread_dir2, "polynomials.txt"), "r") as f:
                loss2 = float(f.readlines()[6].split("Loss = ")[1].strip())

            # Calculate the scale
            scale = ((int(thread.split("_")[1]) - 60 * i - 1) // repetitions + 1) * 10
            if scale not in scale_successes1:
                scale_successes1[scale] = 0
            if scale not in scale_successes2:
                scale_successes2[scale] = 0
            if scale not in scale_successes:
                scale_successes[scale] = 0

            # Check if the model succeeded
            if loss1 < 1e-5:
                successes1 += 1
                scale_successes1[scale] += 1
            if loss2 < 1e-5:
                successes2 += 1
                scale_successes2[scale] += 1
            if loss1 < 1e-5 or loss2 < 1e-5:
                total_successes += 1
                scale_successes[scale] += 1

        plt.figure(figsize=(8, 5))
        plt.title(
            f"Successes per scale for {combinations[i][1]}_{combinations[i][0]} (total of {total_successes} ({successes1}|{successes2}))",
            fontsize=20)
        xs = np.array(list(scale_successes1.keys()))
        ys1, ys2 = list(scale_successes1.values()), list(scale_successes2.values())
        width, space = 2, 2
        plt.bar(xs, ys1, color="royalblue", width=width, edgecolor="black", label="Integer case")
        plt.bar(xs - space, ys2, color="hotpink", width=width, edgecolor="black", label="Normalized case")
        plt.bar(xs + space, list(scale_successes.values()), color="turquoise", width=width, edgecolor="black",
                label="Total")
        plt.xticks(list(scale_successes1.keys()), rotation=45)
        plt.yticks(range(repetitions + 1))
        plt.xlabel("Scale", fontsize=15)
        plt.ylabel("Number of successes", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(join("plots", f"successes_per_scale_300_vary_{combinations[i][1]}_{combinations[i][0]}.png"))
        plt.close()


def analyze_for_100_5_3_ablation():
    """
    combinations = [[3, 5]]
    scales = range(10, 201, 10)
    repetitions = 5
    """
    # Sum the successes
    successes1, successes2, successes3, successes4 = 0, 0, 0, 0
    scale_successes1, scale_successes2, scale_successes3, scale_successes4 = {}, {}, {}, {}
    repetitions = 5

    # The working directory
    WORKING_DIR = join("output_dirs", "100_5_3")

    for thread in os.listdir(join(WORKING_DIR, "train_18")):
        thread_dir1 = join(WORKING_DIR, "train_17_011", thread)
        thread_dir2 = join(WORKING_DIR, "train_17_101", thread)
        thread_dir3 = join(WORKING_DIR, "train_17_110", thread)
        thread_dir4 = join(WORKING_DIR, "train_17", thread)
        with open(join(thread_dir1, "polynomials.txt"), "r") as f:
            loss1 = float(f.readlines()[6].split("Loss = ")[1].strip())
        with open(join(thread_dir2, "polynomials.txt"), "r") as f:
            loss2 = float(f.readlines()[6].split("Loss = ")[1].strip())
        with open(join(thread_dir3, "polynomials.txt"), "r") as f:
            loss3 = float(f.readlines()[6].split("Loss = ")[1].strip())
        with open(join(thread_dir4, "polynomials.txt"), "r") as f:
            loss4 = float(f.readlines()[6].split("Loss = ")[1].strip())

        # Calculate the scale
        scale = ((int(thread.split("_")[1]) - 1) // repetitions + 1) * 10
        if scale not in scale_successes1:
            scale_successes1[scale] = 0
        if scale not in scale_successes2:
            scale_successes2[scale] = 0
        if scale not in scale_successes3:
            scale_successes3[scale] = 0
        if scale not in scale_successes4:
            scale_successes4[scale] = 0

        # Check if the model succeeded
        if loss1 < 1e-5:
            successes1 += 1
            scale_successes1[scale] += 1
        if loss2 < 1e-5:
            successes2 += 1
            scale_successes2[scale] += 1
        if loss3 < 1e-5:
            successes3 += 1
            scale_successes3[scale] += 1
        if loss4 < 1e-5:
            successes4 += 1
            scale_successes4[scale] += 1

    plt.figure(figsize=(8, 5))
    plt.title(f"Ablation results ({successes3}|{successes1}|{successes2}|{successes4})", fontsize=20)
    xs = np.array(list(scale_successes1.keys()))
    ys1, ys2, ys3, ys4 = (list(scale_successes1.values()), list(scale_successes2.values()),
                          list(scale_successes3.values()), list(scale_successes4.values()))
    width, space = 1.5, 2
    plt.bar(xs - 1.5 * space, ys3, color="red", width=width, edgecolor="black", label="Without rounding coefficients")
    plt.bar(xs - 0.5 * space, ys1, color="turquoise", width=width, edgecolor="black",
            label="Without guessing coefficients")
    plt.bar(xs + 0.5 * space, ys2, color="royalblue", width=width, edgecolor="black",
            label="Without using regularization")
    plt.bar(xs + 1.5 * space, ys4, color="hotpink", width=width, edgecolor="black", label="Full model")
    plt.xticks(list(scale_successes1.keys()), rotation=45)
    plt.yticks(range(repetitions + 1))
    plt.xlabel("Scale", fontsize=15)
    plt.ylabel("Number of successes", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join("plots", "ablation_results_100_5_3.png"))
    plt.show()


def analyze_for_300_vary_ablation():
    """
    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 8]]
    scales = range(10, 201, 10)
    repetitions = 3
    """
    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 8]]

    # Modify the working directory
    WORKING_DIR = join("output_dirs", "300_vary")

    for i in range(len(combinations)):
        # Sum the successes
        successes1, successes2, successes3, successes4 = 0, 0, 0, 0
        scale_successes1, scale_successes2, scale_successes3, scale_successes4 = {}, {}, {}, {}
        repetitions = 3
        for thread in os.listdir(join(WORKING_DIR, "train_18")):
            if int(thread.split("_")[1]) > 60 * (i + 1) or int(thread.split("_")[1]) <= 60 * i:
                continue
            thread_dir1 = join(WORKING_DIR, "train_17_011", thread)
            thread_dir2 = join(WORKING_DIR, "train_17_101", thread)
            thread_dir3 = join(WORKING_DIR, "train_17_110", thread)
            thread_dir4 = join(WORKING_DIR, "train_17", thread)
            with open(join(thread_dir1, "polynomials.txt"), "r") as f:
                loss1 = float(f.readlines()[6].split("Loss = ")[1].strip())
            with open(join(thread_dir2, "polynomials.txt"), "r") as f:
                loss2 = float(f.readlines()[6].split("Loss = ")[1].strip())
            with open(join(thread_dir3, "polynomials.txt"), "r") as f:
                loss3 = float(f.readlines()[6].split("Loss = ")[1].strip())
            with open(join(thread_dir4, "polynomials.txt"), "r") as f:
                loss4 = float(f.readlines()[6].split("Loss = ")[1].strip())

            # Calculate the scale
            scale = ((int(thread.split("_")[1]) - 60 * i - 1) // repetitions + 1) * 10
            if scale not in scale_successes1:
                scale_successes1[scale] = 0
            if scale not in scale_successes2:
                scale_successes2[scale] = 0
            if scale not in scale_successes3:
                scale_successes3[scale] = 0
            if scale not in scale_successes4:
                scale_successes4[scale] = 0

            # Check if the model succeeded
            if loss1 < 1e-5:
                successes1 += 1
                scale_successes1[scale] += 1
            if loss2 < 1e-5:
                successes2 += 1
                scale_successes2[scale] += 1
            if loss3 < 1e-5:
                successes3 += 1
                scale_successes3[scale] += 1
            if loss4 < 1e-5:
                successes4 += 1
                scale_successes4[scale] += 1

        plt.figure(figsize=(8, 5))
        plt.title(
            f"Ablation successes per scale for {combinations[i][1]}_{combinations[i][0]} ({successes3}|{successes1}|{successes2}|{successes4})",
            fontsize=20)
        xs = np.array(list(scale_successes1.keys()))
        ys1, ys2, ys3, ys4 = (list(scale_successes1.values()), list(scale_successes2.values()),
                              list(scale_successes3.values()), list(scale_successes4.values()))
        width, space = 2, 2
        plt.bar(xs - 1.5 * space, ys3, color="red", width=width, edgecolor="black",
                label="Without rounding coefficients")
        plt.bar(xs - 0.5 * space, ys1, color="turquoise", width=width, edgecolor="black",
                label="Without guessing coefficients")
        plt.bar(xs + 0.5 * space, ys2, color="royalblue", width=width, edgecolor="black",
                label="Without using regularization")
        plt.bar(xs + 1.5 * space, ys4, color="hotpink", width=width, edgecolor="black", label="Full model")
        plt.xticks(list(scale_successes1.keys()), rotation=45)
        plt.yticks(range(repetitions + 1))
        plt.xlabel("Scale", fontsize=15)
        plt.ylabel("Number of successes", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            join("plots", f"ablation_successes_per_scale_300_vary_{combinations[i][1]}_{combinations[i][0]}.png"))
        plt.close()


analyze_for_100_5_3_ablation()
