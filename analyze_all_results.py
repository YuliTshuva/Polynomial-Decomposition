"""
Yuli Tsvhua
Analyzing all the results of the overall run.
"""

# Imports
import os
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna
import pickle
import numpy as np
from matplotlib import rcParams

# Constants
OUTPUT_DIR = "output_best_hp"
DATASETS = [file for file in os.listdir(OUTPUT_DIR) if os.path.isdir(join(OUTPUT_DIR, file))]
TRAIN = "train_17"
PLOTS_DIR = join("plots", "full_pipeline_best_hp")
CLASSIFIER_DIR = join("classifier")
os.makedirs(CLASSIFIER_DIR, exist_ok=True)
N_JOBS = 10
SEED = 42
K_FOLDS = 4
rcParams["font.family"] = "Times New Roman"


def get_all_results():
    # Define a dictionary to hold the results
    results = {}

    # Iterate through the datasets
    for dataset in DATASETS:
        # Start dataset results
        results[dataset] = {}

        # Define dataset path
        dataset_path = join(OUTPUT_DIR, dataset, TRAIN)

        # Iterate through the threads
        for thread in os.listdir(dataset_path):
            # Define thread path
            thread_path = join(dataset_path, thread)

            # Extract thread number
            thread_num = int(thread.split("_")[1])

            # Check success
            with open(join(thread_path, "polynomials.txt"), "r") as f:
                loss = float(f.readlines()[6].split("Loss = ")[1].strip())
                success = 1 if loss < 1 else 0

            # Check run success
            run_success = int(
                [file.split("_")[1].split(".")[0] for file in os.listdir(thread_path) if "run" in file][0])

            # Register result
            results[dataset][thread_num] = {
                "success": success,
                "run_success": run_success
            }

    return results


def plot_results(results, run):
    # Create a directory for the run plots
    run_plots_dir = join(PLOTS_DIR, f"run_{run}")
    os.makedirs(run_plots_dir, exist_ok=True)

    # Sum the successes
    dataset = "dataset_100_5_3"
    repetitions = 5

    dct = results[dataset]
    successes = {}
    for thread in dct:
        successes[thread] = dct[thread]["success"] if dct[thread]["run_success"] <= run else 0

    scale_successes = {}
    for thread in successes:
        scale = ((thread - 1) // repetitions + 1) * 10
        if scale not in scale_successes:
            scale_successes[scale] = successes[thread]
        else:
            scale_successes[scale] += successes[thread]

    # Total successes
    successes = sum(successes.values())

    plt.figure(figsize=(8, 5))
    plt.title(f"Successes per scale", fontsize=20)
    xs, ys = np.array(list(scale_successes.keys())), list(scale_successes.values())
    width, space = 2, 2
    plt.bar(xs, ys, color="royalblue", width=width, edgecolor="black", label=f"Total Success: ({successes})")
    plt.xticks(xs, rotation=45)
    plt.yticks(range(repetitions + 1))
    plt.xlabel("Scale", fontsize=15)
    plt.ylabel("Number of successes", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(run_plots_dir, "successes_per_scale_100_5_3.png"))
    plt.close()

    # Update the dataset
    dataset = "dataset_300_vary"
    # Set the combinations
    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 7]]

    for i in range(len(combinations)):
        # Sum the successes
        successes, scale_successes = 0, {}
        repetitions = 3
        dct = results[dataset]
        for thread in dct:
            if thread > 60 * (i + 1) or thread <= 60 * i:
                continue
            # Calculate the scale
            scale = ((thread - 60 * i - 1) // repetitions + 1) * 10
            if scale not in scale_successes:
                scale_successes[scale] = 0

            # Check if the model succeeded
            if dct[thread]["success"] and dct[thread]["run_success"] <= run:
                successes += 1
                scale_successes[scale] += 1

        plt.figure(figsize=(8, 5))
        plt.title(f"Successes per scale for {combinations[i][1]}_{combinations[i][0]}", fontsize=20)
        xs, ys = np.array(list(scale_successes.keys())), list(scale_successes.values())
        width, space = 2, 2
        plt.bar(xs, ys, color="royalblue", width=width, edgecolor="black", label=f"Total Success: ({successes})")
        plt.xticks(xs, rotation=45)
        plt.yticks(range(repetitions + 1))
        plt.xlabel("Scale", fontsize=15)
        plt.ylabel("Number of successes", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(join(run_plots_dir, f"successes_per_scale_300_vary_{combinations[i][1]}_{combinations[i][0]}.png"))
        plt.close()

    # Count the success amount for hybrid dataset
    dataset = "dataset_hybrid_1000_deg15"
    dct = results[dataset]
    runs = 5
    run_to_successes = {run: 0 for run in range(1, runs + 1)}
    for run in range(1, runs + 1):
        successes = 0
        for thread in dct:
            success = dct[thread]["success"] if dct[thread]["run_success"] <= run else 0
            successes += success
            if success == 1 and thread > 500:
                raise Exception(f"A non-decomposable polynomial ({thread}) was marked as success.")
        run_to_successes[run] = successes

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.title(f"Successes per attempts for hybrid dataset", fontsize=20)
    xs, ys = np.array(list(run_to_successes.keys())), list(run_to_successes.values())
    width, space = 0.4, 0.2
    plt.bar(xs, ys, color="royalblue", width=width, edgecolor="black", label=f"Total Success: ({ys[-1]})")
    plt.xticks(xs, rotation=45)
    plt.yticks([max(ys)])
    plt.xlabel("Amount of attempts", fontsize=15)
    plt.ylabel("Successes", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(PLOTS_DIR, "successes_per_attempts_hybrid_1000_deg15.png"))
    plt.close()


def classify_results(results):
    # Load dataset_hybrid_1000_deg15.csv
    dataset = "dataset_hybrid_1000_deg15"

    # Define the target variable
    y = pd.Series([results[dataset][i]["success"] for i in range(1, len(results[dataset]) // 2 + 1)])

    # Set features df path
    features_df_path = join(CLASSIFIER_DIR, f"{dataset}_features.csv")

    # If not exists, create it. Else, load it.
    if not os.path.exists(features_df_path):
        data_path = join("data", f"{dataset}.csv")
        df = pd.read_csv(data_path)

        # Keep only the decomposable polynomials
        df = df[df["Decomposable"] == 1]

        # Keep only the input polynomials feature
        df.drop(["Decomposable", "P(x)", "Q(x)"], axis=1, inplace=True)

        # Add column in df for every degree in the polynomials
        max_degree = 15
        for i in range(max_degree + 1):
            df[f"coeff_{i}"] = 0

        # Iterate through the df
        for i in tqdm(range(df.shape[0]), desc="Extracting coefficients to DataFrame", total=df.shape[0]):
            # Extract the coefficients out of the i-th polynomial
            coeffs = sp.Poly(sp.simplify(df.iloc[i]["R(x)"])).all_coeffs()[::-1]
            # Update the coefficients in the df
            for j in range(len(coeffs)):
                df.at[i, f"coeff_{j}"] = float(coeffs[j])
        # Save the features df
        df.to_csv(features_df_path, index=False)
    else:
        df = pd.read_csv(features_df_path)

    # Drop the polynomial column
    df.drop(["R(x)"], axis=1, inplace=True)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # Optuna objective function
    def objective(trial):
        # Search space for RF hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 4, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

        model = RandomForestClassifier(
            **params,
            random_state=SEED,
            n_jobs=N_JOBS
        )

        # 4-fold cross-validation on the training set
        cv_score = cross_val_score(
            model, X_train, y_train,
            cv=K_FOLDS,
            scoring="roc_auc",
            n_jobs=N_JOBS
        ).mean()

        return cv_score

    # Create and run Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )

    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # Train the best model on the full training set
    best_model = RandomForestClassifier(
        **study.best_params,
        random_state=SEED,
        n_jobs=N_JOBS
    )

    best_model.fit(X_train, y_train)

    # Evaluate on the held-out test set
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_pred)

    print("\n\nTest Accuracy of best model:", test_acc)
    print("\n\nTest AUC of best model:", test_auc)

    # Save the model as a pickle file
    model_path = join(CLASSIFIER_DIR, f"{dataset}_rf_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    # Save the dataset
    X_train.to_csv(join(CLASSIFIER_DIR, f"X_train.csv"), index=False)
    X_test.to_csv(join(CLASSIFIER_DIR, f"X_test.csv"), index=False)
    y_train.to_csv(join(CLASSIFIER_DIR, f"y_train.csv"), index=False)
    y_test.to_csv(join(CLASSIFIER_DIR, f"y_test.csv"), index=False)


def view_results():
    # Load the datasets
    X_train = pd.read_csv(join(CLASSIFIER_DIR, f"X_train.csv"))
    X_test = pd.read_csv(join(CLASSIFIER_DIR, f"X_test.csv"))
    y_train = pd.read_csv(join(CLASSIFIER_DIR, f"y_train.csv"))
    y_test = pd.read_csv(join(CLASSIFIER_DIR, f"y_test.csv"))

    # Load the model
    model_path = join(CLASSIFIER_DIR, f"dataset_hybrid_1000_deg15_rf_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Evaluate on the held-out train and test set
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': test_pred})
    df.to_csv(join(CLASSIFIER_DIR, f"test_predictions.csv"), index=False)

    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_pred)

    print("Train Accuracy of loaded model:", train_acc)
    print("Train AUC of loaded model:", train_auc)
    print("Test Accuracy of loaded model:", test_acc)
    print("Test AUC of loaded model:", test_auc)


def main():
    # Get the model's results
    results = get_all_results()

    # Set the amount of runs
    runs = 5
    for run in range(1, runs + 1):
        # Plot the results
        plot_results(results, run)


if __name__ == "__main__":
    main()
