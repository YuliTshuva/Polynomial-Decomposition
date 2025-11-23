"""
Yuli Tshuva
Write a pipeline that run the algorithms on the entire dataset.
"""

import os.path
from os.path import join
import pandas as pd
import subprocess
from functions import *

# Set train file
train = "train_17"  # 18
mode = "standard"  # standard / ablation / hybrid

if mode == "standard":
    # Constants
    ATTEMPTS = 5
    DATASETS = ["dataset_100_5_3.csv", "dataset_300_vary.csv", "dataset_hybrid_1000_deg15.csv"]

    hp_combination = {
        "LR": 5,
        "MIN_LR": 0.001,
        "EARLY_STOPPING": 300,
        "LAMBDA1": 20,
        "LAMBDA2": 0.5,
        "LAMBDA3": 100000,
        "FORCE_COEFFICIENTS": 3000,
    }

    REGULARIZATION_DECAY = 0.85

    # Define the regularization decay as a function of the run number
    run_proces = lambda num: 1 if num <= 2 else REGULARIZATION_DECAY ** (num - 2)

    for _ in range(ATTEMPTS):
        # Iterate through the datasets
        for dataset in DATASETS:
            # Load the dataset
            df = pd.read_csv(join("data", dataset))

            if "Decomposable" in df.columns:
                df["Decomposable"] = df["Decomposable"].astype(int)

            # Create the working directory if it doesn't exist
            WORKING_DIR = join("output", dataset.split(".csv")[0], train)
            os.makedirs(WORKING_DIR, exist_ok=True)

            # Iterate through the dataset
            for i in range(1, len(df) + 1):
                success = False
                run_num = 1
                thread_dir = join(WORKING_DIR, f"thread_{i}")

                # Note the first run
                folder_exists = os.path.exists(thread_dir)
                if not folder_exists:
                    os.makedirs(thread_dir, exist_ok=True)
                    with open(join(thread_dir, "run_1.txt"), "w") as f:
                        f.write("")

                if folder_exists:
                    # Update the run number
                    run_file = [file for file in os.listdir(thread_dir) if "run_" in file][0]
                    run_num = int(run_file.split("_")[1].split(".")[0]) + 1
                    os.remove(join(thread_dir, run_file))
                    with open(join(thread_dir, f"run_{run_num}.txt"), "w") as f:
                        f.write("")

                    # Check the result
                    result_path = join(thread_dir, "polynomials.txt")
                    with open(result_path, "r") as f:
                        result = float(f.readlines()[6].split("Loss = ")[1].strip())
                        if result < 1:
                            success = True

                # If a solution was not found, run again with decayed regularization
                if not success:
                    if dataset != "dataset_hybrid_1000_deg15.csv":
                        p = df.loc[i - 1, "P(x)"]
                        q = df.loc[i - 1, "Q(x)"]
                        args = ["python3", f"{train}.py", p, q, str(i), "1", "1", "1"]
                        args += [str(hp_combination["LR"]), str(hp_combination["MIN_LR"]),
                                 str(hp_combination["EARLY_STOPPING"]),
                                 str(hp_combination["LAMBDA1"] * run_proces(run_num)),
                                 str(hp_combination["LAMBDA2"] * run_proces(run_num)),
                                 str(hp_combination["LAMBDA3"]),
                                 str(hp_combination["FORCE_COEFFICIENTS"])]
                        args += [WORKING_DIR]
                        args += ["None"]

                    else:
                        if df.loc[i - 1, "Decomposable"] == 1:
                            p = df.loc[i - 1, "P(x)"]
                            q = df.loc[i - 1, "Q(x)"]
                            args = ["python3", f"{train}.py", p, q, str(i), "1", "1", "1"]
                            args += [str(hp_combination["LR"]), str(hp_combination["MIN_LR"]),
                                     str(hp_combination["EARLY_STOPPING"]),
                                     str(hp_combination["LAMBDA1"] * run_proces(run_num)),
                                     str(hp_combination["LAMBDA2"] * run_proces(run_num)),
                                     str(hp_combination["LAMBDA3"]),
                                     str(hp_combination["FORCE_COEFFICIENTS"])]
                            args += [WORKING_DIR]
                            args += ["None"]
                        else:
                            args = ["python3", f"{train}.py", "5", "3", str(i), "1", "1", "1"]
                            args += [str(hp_combination["LR"]), str(hp_combination["MIN_LR"]),
                                     str(hp_combination["EARLY_STOPPING"]),
                                     str(hp_combination["LAMBDA1"] * run_proces(run_num)),
                                     str(hp_combination["LAMBDA2"] * run_proces(run_num)),
                                     str(hp_combination["LAMBDA3"]),
                                     str(hp_combination["FORCE_COEFFICIENTS"])]
                            args += [WORKING_DIR]
                            args += [df.loc[i - 1, "R(x)"]]

                    subprocess.run(args=args)

if "ablation" == mode:
    # Iterate through the dataset
    for i, (p, q) in enumerate(zip(df['P(x)'], df['Q(x)']), start=1):
        for j in range(3):
            if os.path.exists(
                    join("output_dirs", f"{train}_{j_to_bool(0, j)}{j_to_bool(1, j)}{j_to_bool(2, j)}", f"thread_{i}")):
                print(
                    f"[{train}_{j_to_bool(0, j)}{j_to_bool(1, j)}{j_to_bool(2, j)}] Thread {i} already exists, skipping...")
            else:
                subprocess.run(
                    args=["python", f"{train}.py", p, q, str(i), j_to_bool(0, j), j_to_bool(1, j), j_to_bool(2, j)], )

if mode == "hybrid":
    # Constants
    DATASET_PATH = join("data", "dataset_hybrid_200_deg15.csv")
    # Load the dataset
    df = pd.read_csv(DATASET_PATH)

    # Create the working directory if it doesn't exist
    WORKING_DIR = join("output_dirs", "hybrid_200", train)
    os.makedirs(WORKING_DIR, exist_ok=True)

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
