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
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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
    plt.title(f"Successes per scale", fontsize=20)
    xs = np.array(list(scale_successes1.keys()))
    ys1, ys2 = list(scale_successes1.values()), list(scale_successes2.values())
    width, space = 2, 2
    plt.bar(xs, ys1, color="royalblue", width=width, edgecolor="black", label=f"Integer version ({successes1})")
    plt.bar(xs - space, ys2, color="hotpink", width=width, edgecolor="black", label=f"Real version ({successes2})")
    plt.bar(xs + space, list(scale_successes.values()), color="turquoise", width=width, edgecolor="black",
            label=f"Total ({total_successes})")
    plt.xticks(list(scale_successes1.keys()), rotation=45)
    plt.yticks(range(repetitions + 1))
    plt.xlabel("Scale", fontsize=15)
    plt.ylabel("Number of successes", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join("plots", "default_hp_results", "successes_per_scale_100_5_3.png"))
    plt.show()


def analyze_for_hybrid_200():
    """
    combinations = [[3, 5]]
    scales = range(10, 201, 10)
    repetitions = 5
    """
    # Sum the successes
    successes = {}

    # The working directory
    WORKING_DIR = join("output_dirs", "hybrid_200", "train_17")
    n_samples = len(os.listdir(WORKING_DIR))

    for thread in os.listdir(WORKING_DIR):
        thread_dir = join(WORKING_DIR, thread)
        with open(join(thread_dir, "polynomials.txt"), "r") as f:
            loss = float(f.readlines()[6].split("Loss = ")[1].strip())

        if loss < 1e-5:
            successes[int(thread.split("_")[-1])] = 1
        else:
            successes[int(thread.split("_")[-1])] = 0

    plt.figure(figsize=(8, 5))
    plt.title(f"Success of decomposition", fontsize=20)
    xs = np.array(sorted(list(successes.keys())))
    ys = np.array([successes[x] for x in xs])
    plt.bar(xs[:n_samples // 2], ys[:n_samples // 2], color="royalblue",
            label=f"Decomposable case ({np.sum(ys[:n_samples // 2])} / {n_samples // 2})")
    plt.bar(xs[n_samples // 2: n_samples], ys[n_samples // 2: n_samples], color="hotpink",
            label=f"Non-Decomposable case ({np.sum(ys[n_samples // 2:n_samples])} / {n_samples // 2})")
    plt.xticks(xs[::n_samples // 10] - 1, rotation=0)
    plt.yticks([0, 1])
    plt.xlabel("Experiment", fontsize=15)
    plt.ylabel("Success", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join("plots", "hybrid_200_results.png"))
    plt.close()


def analyze_for_300_vary():
    """
    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 7]]
    scales = range(10, 201, 10)
    repetitions = 3
    """
    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 7]]

    # Modify the working directory
    WORKING_DIR = join("output_dirs", "300_vary")

    plt.subplots(2, 2, figsize=(18, 12))

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

        if i == 4:
            continue

        plt.subplot(2, 2, i + 1)
        plt.title(
            f"Successes per scale for {combinations[i][1]}_{combinations[i][0]}",
            fontsize=20)
        xs = np.array(list(scale_successes1.keys()))
        ys1, ys2 = list(scale_successes1.values()), list(scale_successes2.values())
        width, space = 2, 2
        plt.bar(xs, ys1, color="royalblue", width=width, edgecolor="black", label=f"Integer version ({successes1})")
        plt.bar(xs - space, ys2, color="hotpink", width=width, edgecolor="black",
                label=f"Real version ({successes2})")
        plt.bar(xs + space, list(scale_successes.values()), color="turquoise", width=width, edgecolor="black",
                label=f"Total ({total_successes})")
        plt.xticks(list(scale_successes1.keys()), rotation=45)
        plt.yticks(range(repetitions + 1))
        if i == 2 or i == 3:
            plt.xlabel("Scale", fontsize=15)
        if i == 0 or i == 2:
            plt.ylabel("Number of successes", fontsize=15)
        plt.legend()

    plt.suptitle("Successes per scale for 300_vary combinations", fontsize=30)
    plt.tight_layout(pad=1.3)
    plt.savefig(join("plots", "default_hp_results", f"successes_per_scale_300_vary.png"))
    plt.show()


def analyze_both_100_and_300():
    """
    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 7]]
    scales = range(10, 201, 10)
    repetitions = 3
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

    success_100 = scale_successes1
    successes_100 = successes1

    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 7]]
    WORKING_DIR = join("output_dirs", "300_vary")

    success_dicts = []
    success_1_counts = []
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

        success_dicts.append(scale_successes1)
        success_1_counts.append(successes1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 5), gridspec_kw={'width_ratios': [1, 2]})

    width, space = 2.3, 2.3

    ax1.set_title(f"Success rate over Dataset 100_5_3", fontsize=20)
    xs = np.array(sorted(list(success_100.keys())))
    ys = np.array([success_100[x] for x in xs])
    ax1.bar(xs, ys, color="royalblue", edgecolor="black", width=2 * width,
            label=f"$(deg\_h, deg\_g) = (3, 5)$ ({successes_100} / 100)")
    ax1.set_xticks(list(success_100.keys()))
    ax1.set_xticklabels(list(success_100.keys()), rotation=45)
    ax1.set_yticks(range(5 + 1))
    ax1.set_xlabel("Coefficients Scale", fontsize=15)
    ax1.set_ylabel("Success Rate", fontsize=15)
    ax1.legend()

    ax2.set_title(f"Success rate over Dataset 300_vary", fontsize=20)
    colors = ["violet", "dodgerblue", "hotpink", "turquoise", "salmon"]
    labels = [f"$(deg\_h, deg\_g) = ({combinations[i][0]}, {combinations[i][1]})$ ({success_1_counts[i]}/60)" for i in
              range(len(combinations))]
    for i, dct in enumerate(success_dicts):
        xs = np.array(sorted(list(dct.keys())))
        ys = np.array([dct[x] for x in xs])
        ax2.bar(1.5 * xs + (i - 2) * space, ys, width=width, color=colors[i], label=labels[i], edgecolor="black")

    ax2.set_xticks(1.5 * xs)
    ax2.set_xticklabels(sorted(list(success_100.keys())), rotation=45)
    ax2.set_yticks(range(3 + 1))
    ax2.set_xlabel("Coefficients Scale", fontsize=15)
    ax2.legend()

    plt.suptitle("Successes rate per scale over Datasets 100_5_3 and 300_vary", fontsize=25)

    plt.tight_layout()
    plt.savefig(join("plots", "default_hp_results", f"successes_per_scale_100_and_300.png"))
    plt.show()


def analyze_ablation():
    plt.subplots(2, 3, figsize=(20, 10))
    plt.suptitle("Ablation study", fontsize=40, y=0.99)
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

    plt.subplot(2, 3, 1)
    plt.title(f"Success per scale - 100", fontsize=27)
    xs = np.array(list(scale_successes1.keys()))
    ys1, ys2, ys3, ys4 = (list(scale_successes1.values()), list(scale_successes2.values()),
                          list(scale_successes3.values()), list(scale_successes4.values()))
    width, space = 1.5, 2
    plt.bar(xs - 1.5 * space, ys3, color="red", width=width, edgecolor="black",
            label=f"Without rounding coefficients ({successes3})")
    plt.bar(xs - 0.5 * space, ys1, color="turquoise", width=width, edgecolor="black",
            label=f"Without guessing coefficients ({successes1})")
    plt.bar(xs + 0.5 * space, ys2, color="royalblue", width=width, edgecolor="black",
            label=f"Without using regularization ({successes2})")
    plt.bar(xs + 1.5 * space, ys4, color="hotpink", width=width, edgecolor="black",
            label=f"Full model ({successes4})")
    plt.xticks(list(scale_successes1.keys()), rotation=45, fontsize=15)
    plt.yticks(range(repetitions + 1), fontsize=15)
    # plt.xlabel("Scale", fontsize=15)
    plt.ylabel("Amount of successes", fontsize=20)
    plt.legend()

    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 7]]

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

        plt.subplot(2, 3, i + 2)
        plt.title(f"Successes per scale for {combinations[i][1]}_{combinations[i][0]}", fontsize=27)
        xs = np.array(list(scale_successes1.keys()))
        ys1, ys2, ys3, ys4 = (list(scale_successes1.values()), list(scale_successes2.values()),
                              list(scale_successes3.values()), list(scale_successes4.values()))
        width, space = 2, 2
        plt.bar(xs - 1.5 * space, ys3, color="red", width=width, edgecolor="black",
                label=f"Without rounding coefficients ({successes3})")
        plt.bar(xs - 0.5 * space, ys1, color="turquoise", width=width, edgecolor="black",
                label=f"Without guessing coefficients ({successes1})")
        plt.bar(xs + 0.5 * space, ys2, color="royalblue", width=width, edgecolor="black",
                label=f"Without using regularization ({successes2})")
        plt.bar(xs + 1.5 * space, ys4, color="hotpink", width=width, edgecolor="black",
                label=f"Full model ({successes4})")
        plt.xticks(list(scale_successes1.keys()), rotation=45, fontsize=15)
        plt.yticks(range(repetitions + 1), fontsize=15)
        if i > 2:
            plt.xlabel("Scale", fontsize=20)
        if i in [1, 3]:
            plt.ylabel("Amount of successes", fontsize=20)
        plt.legend()

    plt.tight_layout(pad=1.3)
    plt.savefig(join("plots", "ablation", f"ablation_study.png"))
    plt.show()


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
    plt.title(f"Ablation results", fontsize=20)
    xs = np.array(list(scale_successes1.keys()))
    ys1, ys2, ys3, ys4 = (list(scale_successes1.values()), list(scale_successes2.values()),
                          list(scale_successes3.values()), list(scale_successes4.values()))
    width, space = 1.5, 2
    plt.bar(xs - 1.5 * space, ys3, color="red", width=width, edgecolor="black",
            label=f"Without rounding coefficients ({successes3})")
    plt.bar(xs - 0.5 * space, ys1, color="turquoise", width=width, edgecolor="black",
            label=f"Without guessing coefficients ({successes1})")
    plt.bar(xs + 0.5 * space, ys2, color="royalblue", width=width, edgecolor="black",
            label=f"Without using regularization ({successes2})")
    plt.bar(xs + 1.5 * space, ys4, color="hotpink", width=width, edgecolor="black",
            label=f"Full model ({successes4})")
    plt.xticks(list(scale_successes1.keys()), rotation=45)
    plt.yticks(range(repetitions + 1))
    plt.xlabel("Scale", fontsize=15)
    plt.ylabel("Number of successes", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join("plots", "ablation_results_100_5_3.png"))
    plt.close()


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
        plt.title(f"Ablation successes per scale for {combinations[i][1]}_{combinations[i][0]}", fontsize=20)
        xs = np.array(list(scale_successes1.keys()))
        ys1, ys2, ys3, ys4 = (list(scale_successes1.values()), list(scale_successes2.values()),
                              list(scale_successes3.values()), list(scale_successes4.values()))
        width, space = 2, 2
        plt.bar(xs - 1.5 * space, ys3, color="red", width=width, edgecolor="black",
                label=f"Without rounding coefficients ({successes3})")
        plt.bar(xs - 0.5 * space, ys1, color="turquoise", width=width, edgecolor="black",
                label=f"Without guessing coefficients ({successes1})")
        plt.bar(xs + 0.5 * space, ys2, color="royalblue", width=width, edgecolor="black",
                label=f"Without using regularization ({successes2})")
        plt.bar(xs + 1.5 * space, ys4, color="hotpink", width=width, edgecolor="black",
                label=f"Full model ({successes4})")
        plt.xticks(list(scale_successes1.keys()), rotation=45)
        plt.yticks(range(repetitions + 1))
        plt.xlabel("Scale", fontsize=15)
        plt.ylabel("Number of successes", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            join("plots", f"ablation_successes_per_scale_300_vary_{combinations[i][1]}_{combinations[i][0]}.png"))
        plt.close()


def find_mean_epoch():
    """
    Find the mean epoch for convergence.
    """
    WORKING_DIR = join("output_best_hp", "dataset_300_vary", "train_17")
    epochs, success = [], []

    for thread in os.listdir(WORKING_DIR):
        thread_dir = join(WORKING_DIR, thread)
        with open(join(thread_dir, "polynomials.txt"), "r") as f:
            lines = f.readlines()
            loss = float(lines[6].split("Loss = ")[1].strip())
            epoch = int(lines[6].split("Epoch")[1].split(":")[0].strip())
            if loss < 1e-5:
                success.append(1)
            else:
                success.append(0)
            epochs.append(epoch)

    epochs = np.array(epochs)
    success = np.array(success)
    successful_epochs = epochs[success == 1]
    failed_epochs = epochs[success == 0]

    mean_epochs, mean_successful, mean_fail = np.mean(epochs), np.mean(successful_epochs), np.mean(failed_epochs)
    print(f"Mean epochs: {mean_epochs}, Mean successful epochs: {mean_successful}, Mean failed epochs: {mean_fail}")

    # Plot the histogram where successful and failed epochs are in different colors
    plt.figure(figsize=(8, 5))
    plt.title(f"Epochs until convergence (mean: {mean_epochs:.2f})", fontsize=25)
    plt.hist(successful_epochs, bins=60, color="turquoise", label=f"Successful trail"
                                                                  f" (mean: {mean_successful:.2f})")
    plt.hist(failed_epochs, bins=100, color="salmon", label=f"Failed trail (mean: {mean_fail:.2f})")
    plt.xlabel("Epochs until convergence", fontsize=19)
    plt.ylabel("Number of experiments", fontsize=19)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Increase font size
    plt.legend(fontsize=17)
    plt.tight_layout()
    plt.savefig(join("plots", "mean_epochs_until_convergence_300_vary.png"))
    plt.show()


def plot_loss_distribution_of_fails():
    """
    Find the mean epoch for convergence.
    """
    WORKING_DIR = join("output_best_hp", "dataset_hybrid_1000_deg15", "train_17")
    epochs, success, decomposable, losses = [], [], [], []

    for i in range(1, 1001):
        thread_dir = join(WORKING_DIR, "thread_" + str(i))
        with open(join(thread_dir, "polynomials.txt"), "r") as f:
            lines = f.readlines()
            loss = float(lines[6].split("Loss = ")[1].strip())
            losses.append(loss)
            epoch = int(lines[6].split("Epoch")[1].split(":")[0].strip())
            if loss < 1e-5:
                success.append(1)
            else:
                success.append(0)
            epochs.append(epoch)
            if i <= 500:
                decomposable.append(1)
            else:
                decomposable.append(0)

    epochs = np.array(epochs)
    success = np.array(success)
    decomposable = np.array(decomposable)
    losses = np.array(losses)

    dec_losses, non_dec_losses = losses[decomposable + success == 1], losses[decomposable == 0]
    dec_epochs, non_dec_epochs = epochs[decomposable + success == 1], epochs[decomposable == 0]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    alpha = 0.7
    plt.subplot(1, 3, 1)
    plt.title(f"Final loss", fontsize=20)
    plt.hist(non_dec_losses, bins=100, color="salmon", label=f"Non-decomposable polynomials", alpha=alpha)
    plt.hist(dec_losses, bins=1000, color="turquoise", label=f"Decomposable polynomials", alpha=alpha)
    plt.xlabel("Final Loss", fontsize=17)
    plt.xscale("log")
    plt.xlim(np.min(losses) * 0.9, np.max(losses) * 1.1)
    plt.ylabel("Number of experiments", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Increase font size
    plt.legend(fontsize=14)

    plt.subplot(1, 3, 2)
    plt.title(f"Epochs until convergence", fontsize=20)
    plt.hist(non_dec_epochs, bins=100, color="salmon", label=f"Non-decomposable polynomials", alpha=alpha)
    plt.hist(dec_epochs, bins=60, color="turquoise", label=f"Decomposable polynomials", alpha=alpha)
    plt.xlabel("Epochs until convergence", fontsize=17)
    plt.ylabel("Number of experiments", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Increase font size
    plt.legend(fontsize=14)

    plt.subplot(1, 3, 3)
    # Plot a 2d plot of final loss vs epochs until convergence, where decomposable and non-decomposable are in different colors
    # Plot with log scale on y axis
    plt.title(f"Final loss vs Epochs until convergence", fontsize=20)
    plt.scatter(dec_epochs, dec_losses, color="turquoise", label=f"Decomposable polynomials", alpha=alpha)
    plt.scatter(non_dec_epochs, non_dec_losses, color="salmon", label=f"Non-decomposable polynomials", alpha=alpha)
    plt.yscale("log")
    plt.ylim(np.min(losses) * 0.9, np.max(losses) * 1.1)
    plt.xlabel("Epochs until convergence", fontsize=17)
    plt.ylabel("Final Loss", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Increase font size
    plt.legend(fontsize=14, loc="lower right")

    plt.suptitle("Distributions of Decomposable vs Non-Decomposable Polynomials", fontsize=26)

    plt.tight_layout(pad=1.9)
    plt.savefig(join("plots", "rq1_plots", "distribution_of_decomposable_vs_non_decomposable.png"))
    plt.show()


def put_losses_in_figure():
    plt.subplots(3, 3, figsize=(13, 10))
    interesting_runs = [1, 2, 4, 3, 22, 30, 524, 530, 540]
    loss_list = []
    all_losses = []
    for run in interesting_runs:
        loss_path = f"output_best_hp/dataset_hybrid_1000_deg15_sample_for_plot/train_17/thread_{run}/losses.pkl"
        with open(loss_path, "rb") as f:
            losses = pickle.load(f)

        loss_list.append(losses)
        all_losses += losses

    total_min, total_max = np.min(all_losses) / 10, np.max(all_losses)

    for i in range(1, 10):
        losses = loss_list[i - 1]
        if i < 4:
            losses.append(0)
        plt.subplot(3, 3, i)
        plt.plot(range(1, len(losses) + 1), losses, color="red")
        plt.title(f"Run {interesting_runs[i - 1]}", fontsize=18)
        if i > 6:
            plt.xlabel("Epochs", fontsize=15)
        if i % 3 == 1:
            plt.ylabel(f"Class {i // 3 + 1}\nLoss", fontsize=15)
        plt.yscale("log")
        plt.ylim(total_min * 0.9, total_max * 1.1)

    plt.suptitle(
        "Loss curves of decomposable polynomials our method decomposed successfully (class 1)\ndidn't manage to decompose (class 2) and non-decomposable polynomials (class 3)",
        fontsize=25)
    plt.tight_layout(pad=2.0)
    plt.savefig(join("plots", "rq1_plots", "loss_curves_examples.png"))
    plt.show()


def classify_decomposable():
    """
    Find the mean epoch for convergence.
    """
    WORKING_DIR = join("output_best_hp", "dataset_hybrid_1000_deg15", "train_17")
    epochs, success, decomposable, losses = [], [], [], []

    for i in range(1, 1001):
        thread_dir = join(WORKING_DIR, "thread_" + str(i))
        with open(join(thread_dir, "polynomials.txt"), "r") as f:
            lines = f.readlines()
            loss = float(lines[6].split("Loss = ")[1].strip())
            losses.append(loss)
            epoch = int(lines[6].split("Epoch")[1].split(":")[0].strip())
            if loss < 1e-5:
                success.append(1)
            else:
                success.append(0)
            epochs.append(epoch)
            if i <= 500:
                decomposable.append(1)
            else:
                decomposable.append(0)

    epochs = np.array(epochs)
    success = np.array(success)
    decomposable = np.array(decomposable)
    losses = np.array(losses)

    dec_losses, non_dec_losses = losses[decomposable + success == 1], losses[decomposable == 0]
    dec_epochs, non_dec_epochs = epochs[decomposable + success == 1], epochs[decomposable == 0]

    # Built a df to classify decomposable vs non-decomposable based on final loss and epochs until convergence
    data = {
        "final_loss": np.concatenate([dec_losses, non_dec_losses]),
        "epochs": np.concatenate([dec_epochs, non_dec_epochs]),
        "label": np.concatenate([np.ones(len(dec_losses)), np.zeros(len(non_dec_losses))])
    }
    df = pd.DataFrame(data)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df[["final_loss", "epochs"]], df["label"], test_size=0.2, random_state=42
    )

    # Print how many decomposable and non-decomposable in train and test sets
    print("Train set:")
    print(y_train.value_counts())
    print("Test set:")
    print(y_test.value_counts())

    # Train a xgboost model
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # Create a df of test set with predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of classifying decomposable vs non-decomposable: {accuracy * 100:.2f}%")

    # Create a df of the test set with predictions
    df = pd.DataFrame({"labels": y_test, "predictions": y_pred})
    pass


if __name__ == "__main__":
    analyze_both_100_and_300()
