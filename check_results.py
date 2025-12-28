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


def analyze_ablation():
    plt.subplots(3, 2, figsize=(13, 17))
    plt.suptitle("Ablation study results", fontsize=40, y=0.99)
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

    plt.subplot(3, 2, 1)
    plt.title(f"Ablation success per scale - 100", fontsize=27)
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

        plt.subplot(3, 2, i + 2)
        plt.title(f"Ablation successes per scale for {combinations[i][1]}_{combinations[i][0]}", fontsize=27)
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
    plt.savefig(
        join("plots", "ablation", f"ablation_study.png"))
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
    plt.subplot(1, 3, 1)
    plt.title(f"Final loss", fontsize=20)
    plt.hist(dec_losses, bins=400, color="turquoise", label=f"Decomposable polynomials", alpha=0.7)
    plt.hist(non_dec_losses, bins=100, color="salmon", label=f"Non-decomposable polynomials", alpha=0.7)
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
    plt.hist(non_dec_epochs, bins=100, color="salmon", label=f"Non-decomposable polynomials", alpha=0.7)
    plt.hist(dec_epochs, bins=60, color="turquoise", label=f"Decomposable polynomials", alpha=0.7)
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
    plt.scatter(dec_epochs, dec_losses, color="turquoise", label=f"Decomposable polynomials", alpha=0.6)
    plt.scatter(non_dec_epochs, non_dec_losses, color="salmon", label=f"Non-decomposable polynomials", alpha=0.6)
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
    scale = 1.5
    plt.subplots(3, 3, figsize=(10 * scale, 9 * scale))
    interesting_runs = [1, 2, 4, 3, 22, 30, 524, 530, 540]
    paths = [f"output_best_hp/dataset_hybrid_1000_deg15/train_17/thread_{i}/loss.png" for i in interesting_runs]



    plt.suptitle("Loss curves for decomposable vs Non-decomposable experiments", fontsize=20 * scale, y=0.95)
    plt.tight_layout(pad=2.0)
    plt.savefig(join("plots", "rq1_plots", "loss_curves_examples.png"))
    plt.show()


if __name__ == "__main__":
    put_losses_in_figure()
