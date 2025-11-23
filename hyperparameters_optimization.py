"""
Yuli Tshuva
In this peace of code I'll try to find the best hyperparameters for my algorithm.
This will be a heavy code that will take weeks to run.
I'll discalme and say the since the time complexity is heavy, the parameters search space will be limited,
around the used ones, and on a small evaluation set.
What I actually need for this:
0. Understand what hyperparameters I want to optimize.
1. Define a validation set.
    What do I want from the validation set?
    - It needs to be small, so the optimization will be faster.
    - It needs to be representative, so the optimization will be meaningful.
2. Define an objective function.
3. Define the search space for the hyperparameters.
4. Adjust the optimization algorithm.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Imports
import sympy as sp
import pandas as pd
from os.path import join
from create_dataset import random_int
import subprocess
import random
from functions import get_time
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"

# Parameters
VALIDATION_SET_PATH = join("data", "validation_set.csv")
HP_DIR = "hp_optimization"

# Define the hyperparameters search space
HYPERPARAMETERS_SPACE_17 = {
    "LR": [5, 10, 20],
    "MIN_LR": [1e-5, 1e-4, 1e-3, 1e-2],
    "EARLY_STOPPING": [200, 300, 400, 500],
    "LAMBDA1": [0.1, 0.5, 1, 5, 10, 20],
    "LAMBDA2": [0.1, 0.5, 1, 5, 10, 20],
    "LAMBDA3": [500, 1e3, 1e4, 1e5],
    "FORCE_COEFFICIENTS": [3000, 4000, 5000, 6000]
}

HYPERPARAMETERS_SPACE_18 = {
    "LAMBDA2": [0.1, 0.5, 1, 5, 10, 20, 100],
    "LAMBDA4": [0.1, 0.5, 1, 5, 10, 20, 100],
}


def create_validation_set():
    # Constants
    x = sp.symbols("x")

    # Set the df
    df = pd.DataFrame(columns=["P(x)", "Q(x)"])

    # Run up the degrees
    combinations = [[3, 5], [3, 6], [4, 4]]
    scales = range(10, 201, 20)
    repetitions = 3
    for deg_q, deg_p in combinations:
        # Iterate over the scales
        for scale in scales:
            # Repeat {repetitions} times
            for _ in range(repetitions):
                # Generate t
                p = sum([random_int(scale, allow_zero=(i != deg_p)) * x ** i for i in range(deg_p + 1)])
                q = sum([random_int(scale, allow_zero=(i != deg_q)) * x ** i for i in range(deg_q + 1)])
                # Add a row to the df
                df.loc[df.shape[0]] = [p, q]

    # Save the df to a csv file
    df.to_csv(VALIDATION_SET_PATH, index=False, columns=["P(x)", "Q(x)"])


def get_hyperparameters_combination(train="train_17"):
    if train == "train_17":
        hp_combinations = {
            "LR": random.sample(HYPERPARAMETERS_SPACE_17["LR"], 1)[0],
            "MIN_LR": random.sample(HYPERPARAMETERS_SPACE_17["MIN_LR"], 1)[0],
            "EARLY_STOPPING": random.sample(HYPERPARAMETERS_SPACE_17["EARLY_STOPPING"], 1)[0],
            "LAMBDA1": random.sample(HYPERPARAMETERS_SPACE_17["LAMBDA1"], 1)[0],
            "LAMBDA2": random.sample(HYPERPARAMETERS_SPACE_17["LAMBDA2"], 1)[0],
            "LAMBDA3": random.sample(HYPERPARAMETERS_SPACE_17["LAMBDA3"], 1)[0],
            "FORCE_COEFFICIENTS": random.sample(HYPERPARAMETERS_SPACE_17["FORCE_COEFFICIENTS"], 1)[0]
        }
    else:
        hp_combinations = {
            "LAMBDA2": random.sample(HYPERPARAMETERS_SPACE_18["LAMBDA2"], 1)[0],
            "LAMBDA4": random.sample(HYPERPARAMETERS_SPACE_18["LAMBDA4"], 1)[0],
        }
    return hp_combinations


def validate_hp_combination(hp_combination, train="train_17"):
    # Iterate over the hp dir
    working_dir = join(HP_DIR, train)
    for dir in os.listdir(working_dir):
        # Load the hyperparameters
        hp_path = join(working_dir, dir, "hyperparameters.txt")

        # Read the hyperparameters file
        with open(hp_path, "r") as f:
            lines = f.readlines()

        if train == "train_17":
            # Parse the hyperparameters
            existing_hp_combination = {
                "LR": float(lines[0].strip().split(": ")[1]),
                "MIN_LR": float(lines[1].strip().split(": ")[1]),
                "EARLY_STOPPING": int(lines[2].strip().split(": ")[1]),
                "LAMBDA1": float(lines[3].strip().split(": ")[1]),
                "LAMBDA2": float(lines[4].strip().split(": ")[1]),
                "LAMBDA3": float(lines[5].strip().split(": ")[1]),
                "FORCE_COEFFICIENTS": float(lines[6].strip().split(": ")[1])
            }
        else:
            # Parse the hyperparameters
            existing_hp_combination = {
                "LAMBDA2": float(lines[0].strip().split(": ")[1]),
                "LAMBDA4": float(lines[1].strip().split(": ")[1]),
            }

        # Compare the hyperparameters
        if hp_combination == existing_hp_combination:
            return False

    return True


def list_hp_values(hp_combination, train="train_17"):
    if train == "train_17":
        return [str(hp_combination["LR"]), str(hp_combination["MIN_LR"]), str(hp_combination["EARLY_STOPPING"]),
                str(hp_combination["LAMBDA1"]), str(hp_combination["LAMBDA2"]), str(hp_combination["LAMBDA3"]),
                str(hp_combination["FORCE_COEFFICIENTS"])]
    else:
        return [str(hp_combination["LAMBDA2"]), str(hp_combination["LAMBDA4"])]


def analyze_results(train="train_17"):
    working_dir = join(HP_DIR, train)
    trail_dirs = os.listdir(working_dir)
    trail_to_score = {}
    for trail_dir in trail_dirs:
        trail_path = join(working_dir, trail_dir)
        threads_path = join(trail_path, "threads")
        thread_dirs = os.listdir(threads_path)
        total_score = 0
        for thread_dir in thread_dirs:
            thread_path = join(threads_path, thread_dir)
            result_path = join(thread_path, "polynomials.txt")
            with open(result_path, "r") as f:
                result = float(f.readlines()[6].split("Loss = ")[1].strip())
                if result < 1:
                    total_score += 1

        with open(join(trail_path, "score.txt"), "w") as f:
            trail_to_score[int(trail_dir.split("_")[-1])] = total_score
            f.write(f"Total score: {total_score}/{len(thread_dirs)}\n")

    plt.figure()
    # Plot for each trail its score like a histogram
    trails = list(trail_to_score.keys())
    plt.bar(x=trails, height=list(trail_to_score.values()), color="royalblue", edgecolor="black")
    plt.plot([min(trails), max(trails)], [len(thread_dirs)] * 2, color="turquoise")
    plt.yticks(range(0, len(thread_dirs) + 1, 10))
    plt.xticks(trails)
    plt.xlabel("Trail number", fontsize=15)
    plt.ylabel("Score (correct decompositions)", fontsize=15)
    plt.title(f"Hyperparameters Optimization Results ({'integer case' if train == 'train_17' else 'normalized case'})",
              fontsize=20)
    plt.savefig(join("plots", f"{train}_hp_optimization_results.png"))
    plt.show()

    # Find what trail hyperparameters had the best score
    best_trail = max(trail_to_score, key=trail_to_score.get)
    print(f"Best trail is trail {best_trail} with score {trail_to_score[best_trail]}/{len(thread_dirs)}")


def extract_hp_for_dir(hp_path, train):
    # Read the hyperparameters file
    with open(hp_path, "r") as f:
        lines = f.readlines()

    if train == "train_17":
        # Parse the hyperparameters
        hp_combination = {
            "LR": float(lines[0].strip().split(": ")[1]),
            "MIN_LR": float(lines[1].strip().split(": ")[1]),
            "EARLY_STOPPING": int(lines[2].strip().split(": ")[1]),
            "LAMBDA1": float(lines[3].strip().split(": ")[1]),
            "LAMBDA2": float(lines[4].strip().split(": ")[1]),
            "LAMBDA3": float(lines[5].strip().split(": ")[1]),
            "FORCE_COEFFICIENTS": float(lines[6].strip().split(": ")[1])
        }
    else:
        # Parse the hyperparameters
        hp_combination = {
            "LAMBDA2": float(lines[0].strip().split(": ")[1]),
            "LAMBDA4": float(lines[1].strip().split(": ")[1]),
        }

    return hp_combination


def main():
    # Create the validation set if it doesn't exist
    if not os.path.exists(VALIDATION_SET_PATH):
        print(f"[{get_time()}] Creating validation set...")
        create_validation_set()

    # Load the validation set
    df = pd.read_csv(VALIDATION_SET_PATH)

    # Set train file
    train = "train_17"  # 18

    # Run the process of HP optimization
    while True:
        # Find trail number
        working_dir = join(HP_DIR, train)
        trail_num = os.listdir(working_dir)
        new_trail = True
        if len(trail_num) == 0:
            trail_num = 1
        else:
            trail_num = max([int(name.split("_")[-1]) for name in trail_num]) + 1
            if len(os.listdir(join(working_dir, f"trail_{trail_num - 1}", "threads"))) < df.shape[0]:
                trail_num -= 1
                new_trail = False

        # Get a hyperparameters combination
        if new_trail:
            hp_combination = get_hyperparameters_combination(train=train)

            hp_combination = {
                "LR": 10,
                "MIN_LR": 0.001,
                "EARLY_STOPPING": 300,
                "LAMBDA1": 2,
                "LAMBDA2": 1,
                "LAMBDA3": 10000,
                "FORCE_COEFFICIENTS": 4000,
            }
        else:
            hp_path = join(working_dir, f"trail_{trail_num}", "hyperparameters.txt")
            hp_combination = extract_hp_for_dir(hp_path, train=train)

        # Set directories paths
        hp_dir = join(HP_DIR, train, f"trail_{trail_num}")
        threads_dir = join(hp_dir, "threads")
        hp_path = join(hp_dir, "hyperparameters.txt")

        if new_trail:
            # Validate the hyperparameters combination
            if not validate_hp_combination(hp_combination, train=train):
                print(f"[{get_time()}]: Hyperparameters combination already exists, getting a new one...")
                continue

            # Create a new directory for the hyperparameters combination
            os.makedirs(hp_dir, exist_ok=False)

            # Create a new directory for the threads
            os.makedirs(threads_dir, exist_ok=False)

            # Save the hyperparameters combination to a file
            with open(hp_path, "w") as f:
                for key, value in hp_combination.items():
                    f.write(f"{key}: {value}\n")

        print(f"[trail {trail_num}] Evaluating hyperparameters combination: {hp_combination}")

        # Iterate through the dataset
        for i, (p, q) in enumerate(zip(df['P(x)'], df['Q(x)']), start=1):
            if os.path.exists(join(threads_dir, f"thread_{i}")):
                print(
                    f"[{get_time()}] Trail {trail_num} thread {i} already exists, skipping...")
            else:
                argv_list = ["python", f"{train}.py", p, q, str(i)]
                if train == "train_17":
                    argv_list += ["1", "1", "1"]
                argv_list += list_hp_values(hp_combination, train=train) + [threads_dir]
                print(f"[{get_time()}] Starting trail {trail_num} thread {i}.")
                subprocess.run(args=argv_list, )


if __name__ == "__main__":
    main()
