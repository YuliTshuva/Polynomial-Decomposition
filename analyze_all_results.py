"""
Yuli Tsvhua
Analyzing all the results of the overall run.
"""

# Imports
import torch
import os
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
# import optuna
import pickle
import numpy as np
from matplotlib import rcParams
# from fastai.tabular.all import *

# Constants
OUTPUT_DIR = "output_best_hp"
DATASETS = ["dataset_100_5_3", "dataset_300_vary"]
TRAIN = "train_17"
PLOTS_DIR = join("plots", "full_pipeline_best_hp")
CLASSIFIER_DIR = join("classifier")
os.makedirs(CLASSIFIER_DIR, exist_ok=True)
N_JOBS = 10
SEED = 42
K_FOLDS = 4
rcParams["font.family"] = "Times New Roman"
LABEL_SIZE = 35
TITLE_SIZE = 38
SUPTITLE_SIZE = 41
LINESTYLES = ["solid", "dashed", "dotted", "dashdot", (0, (3, 1, 1, 1))]



def get_all_results(output_dir):
    # Define a dictionary to hold the results
    results = {}

    # Iterate through the datasets
    for dataset in DATASETS:
        # Start dataset results
        results[dataset] = {}

        # Define dataset path
        dataset_path = join(output_dir, dataset, TRAIN)

        if not os.path.exists(dataset_path):
            dataset_path = dataset_path.replace("dataset_", "")

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
            try:
                run_success = int(
                    [file.split("_")[1].split(".")[0] for file in os.listdir(thread_path) if "run" in file][0])
            except:
                run_success = 6

            # Register result
            results[dataset][thread_num] = {
                "success": success,
                "run_success": run_success
            }

    return results


def plot_results(results1, results2, results3, run, style="bin"):
    run_plots_dir = PLOTS_DIR
    os.makedirs(run_plots_dir, exist_ok=True)

    fig, ax = plt.subplots(2, 3, figsize=(20, 12))

    dataset = "dataset_100_5_3"
    repetitions = 5

    dct1, dct2, dct3 = results1[dataset], results2[dataset], results3[dataset]
    successes1, successes2, successes3 = {}, {}, {}
    for thread in dct1:
        successes1[thread] = dct1[thread]["success"] if dct1[thread]["run_success"] <= run else 0
        successes2[thread] = dct2[thread]["success"] if dct2[thread]["run_success"] <= run else 0
        successes3[thread] = dct3[thread]["success"] if dct3[thread]["run_success"] <= run else 0

    scale_successes1, scale_successes2, scale_successes3 = {}, {}, {}
    for thread in successes1:
        scale = ((thread - 1) // repetitions + 1) * 10
        if scale not in scale_successes1:
            scale_successes1[scale] = successes1[thread]
        else:
            scale_successes1[scale] += successes1[thread]
        if scale not in scale_successes2:
            scale_successes2[scale] = successes2[thread]
        else:
            scale_successes2[scale] += successes2[thread]
        if scale not in scale_successes3:
            scale_successes3[scale] = successes3[thread]
        else:
            scale_successes3[scale] += successes3[thread]

    successes1, successes2, successes3 = sum(successes1.values()), sum(successes2.values()), sum(successes3.values())

    ax[0, 0].set_title(f"Successes per scale - 100", fontsize=TITLE_SIZE)
    xs = np.array(list(scale_successes1.keys()))
    ys1, ys2, ys3 = list(scale_successes1.values()), list(scale_successes2.values()), list(scale_successes3.values())

    sorted_indices = np.argsort(xs)
    xs = xs[sorted_indices]
    ys1 = np.array(ys1)[sorted_indices]
    ys2 = np.array(ys2)[sorted_indices]
    ys3 = np.array(ys3)[sorted_indices]

    width = 2
    if style == "bin":
        ax[0, 0].bar(xs - width, ys1, color="royalblue", width=width, edgecolor="black", label="Best hp")
        ax[0, 0].bar(xs,        ys2, color="hotpink",   width=width, edgecolor="black", label="Baseline")
        ax[0, 0].bar(xs + width, ys3, color="turquoise", width=width, edgecolor="black", label="Results3")
    if style == "line":
        ax[0, 0].plot(xs, ys1, color="royalblue", label="Best hp", linestyle=LINESTYLES[0])
        ax[0, 0].plot(xs, ys3, color="turquoise", label="Base method", linestyle=LINESTYLES[1])
        ax[0, 0].plot(xs, ys2, color="hotpink",   label="Baseline", linestyle=LINESTYLES[2])
    ax[0, 0].set_xticks(xs, labels=xs, rotation=45)
    ax[0, 0].set_yticks(range(repetitions + 1))
    ax[0, 0].set_xlabel("Coefficients Scale", fontsize=LABEL_SIZE)
    ax[0, 0].set_ylabel("Success Rate", fontsize=LABEL_SIZE)
    # No legend here

    dataset = "dataset_300_vary"
    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 7]]

    for i in range(len(combinations)):
        successes1, successes2, successes3 = 0, 0, 0
        scale_successes1, scale_successes2, scale_successes3 = {}, {}, {}
        repetitions = 3
        dct1, dct2, dct3 = results1[dataset], results2[dataset], results3[dataset]
        for thread in dct1:
            if thread > 60 * (i + 1) or thread <= 60 * i:
                continue
            scale = ((thread - 60 * i - 1) // repetitions + 1) * 10
            if scale not in scale_successes1:
                scale_successes1[scale] = 0
            if scale not in scale_successes2:
                scale_successes2[scale] = 0
            if scale not in scale_successes3:
                scale_successes3[scale] = 0

            if dct1[thread]["success"] and dct1[thread]["run_success"] <= run:
                successes1 += 1
                scale_successes1[scale] += 1
            if dct2[thread]["success"] and dct2[thread]["run_success"] <= run:
                successes2 += 1
                scale_successes2[scale] += 1
            if dct3[thread]["success"] and dct3[thread]["run_success"] <= run:
                successes3 += 1
                scale_successes3[scale] += 1

        row, col = (0 if i < 2 else 1), (i + 1) % 3
        ax[row, col].set_title(
            f"Successes per scale for {combinations[i][1]}_{combinations[i][0]}", fontsize=TITLE_SIZE)
        xs = np.array(list(scale_successes1.keys()))
        ys1, ys2, ys3 = list(scale_successes1.values()), list(scale_successes2.values()), list(scale_successes3.values())

        sorted_indices = np.argsort(xs)
        xs = xs[sorted_indices]
        ys1 = np.array(ys1)[sorted_indices]
        ys2 = np.array(ys2)[sorted_indices]
        ys3 = np.array(ys3)[sorted_indices]

        if style == "bin":
            ax[row, col].bar(xs - width, ys1, color="royalblue", width=width, edgecolor="black", label="Best hp")
            ax[row, col].bar(xs,         ys2, color="hotpink",   width=width, edgecolor="black", label="Baseline")
            ax[row, col].bar(xs + width,  ys3, color="turquoise", width=width, edgecolor="black", label="Results3")
        if style == "line":
            ax[row, col].plot(xs, ys1, color="royalblue", label="Best hp", linestyle=LINESTYLES[0])
            ax[row, col].plot(xs, ys3, color="turquoise", label="Base method", linestyle=LINESTYLES[1])
            ax[row, col].plot(xs, ys2, color="hotpink",   label="Baseline", linestyle=LINESTYLES[2])
        ax[row, col].set_xticks(xs, labels=xs, rotation=45)
        ax[row, col].set_yticks(range(repetitions + 1))
        if i > 2:
            ax[row, col].set_xlabel("Coefficients Scale", fontsize=LABEL_SIZE)
        if i in [0, 3]:
            ax[row, col].set_ylabel("Success Rate", fontsize=LABEL_SIZE)
        # No legend here

    # Single shared legend at the bottom
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=20,
               bbox_to_anchor=(0.5, -0.01), frameon=True)

    if run != 6:
        fig.suptitle(f"Iteration {run}", fontsize=30)
    if run == 6:
        fig.suptitle(f"Ensemble Pipeline", fontsize=SUPTITLE_SIZE)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)  # Make room for the legend
    fig.savefig(join(run_plots_dir, f"{style}_results_run_{run}.pdf"), bbox_inches="tight")
    plt.show()

    return
    # ... (rest of the function unchanged)

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

    # Subsample indices such that the dataset is balanced
    decomposable_indices = y[y == 1].index
    non_decomposable_indices = y[y == 0].index
    sampled_decomposable_indices = np.random.choice(decomposable_indices, size=len(non_decomposable_indices),
                                                    replace=False)
    balanced_indices = np.concatenate([non_decomposable_indices, sampled_decomposable_indices])
    df = df.loc[balanced_indices].reset_index(drop=True)
    y = y.loc[balanced_indices].reset_index(drop=True)

    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # Load existing sets
    X_train = pd.read_csv(join(CLASSIFIER_DIR, f"X_train.csv"))
    X_test = pd.read_csv(join(CLASSIFIER_DIR, f"X_test.csv"))
    y_train = pd.read_csv(join(CLASSIFIER_DIR, f"y_train.csv"))
    y_test = pd.read_csv(join(CLASSIFIER_DIR, f"y_test.csv"))

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
            scoring="accuracy",
            n_jobs=N_JOBS
        ).mean()

        return cv_score

    # Create and run Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )

    # study.optimize(objective, n_trials=100, show_progress_bar=True)

    # # Train the best model on the full training set
    # best_model = RandomForestClassifier(
    #     **study.best_params,
    #     random_state=SEED,
    #     n_jobs=N_JOBS
    # )

    # best_model.fit(X_train, y_train)

    # Evaluate on the held-out test set
    # test_pred = best_model.predict(X_test)
    # test_acc = accuracy_score(y_test, test_pred)
    # test_auc = roc_auc_score(y_test, test_pred)
    #
    # print("\n\nRF Test Accuracy of best model:", test_acc)
    # print("\n\nRF Test AUC of best model:", test_auc)

    # # Save the datasets
    # X_train.to_csv(join(CLASSIFIER_DIR, f"X_train.csv"), index=False)
    # X_test.to_csv(join(CLASSIFIER_DIR, f"X_test.csv"), index=False)
    # y_train.to_csv(join(CLASSIFIER_DIR, f"y_train.csv"), index=False)
    # y_test.to_csv(join(CLASSIFIER_DIR, f"y_test.csv"), index=False)

    # # Save the model as a pickle file
    # model_path = join(CLASSIFIER_DIR, f"{dataset}_rf_model.pkl")
    # with open(model_path, "wb") as f:
    #     pickle.dump(best_model, f)

    # Create a simple MLP model using fastai
    X_train["target"] = y_train
    dls = TabularDataLoaders.from_df(
        X_train,
        procs=[Normalize],
        cont_names=X_train.columns.drop("target").tolist(),
        cat_names=[],
        y_names='target',
        bs=8
    )

    learn = tabular_learner(
        dls,
        layers=[50, 100, 50],  # MLP architecture
        lr=1e-3,
        metrics=accuracy
    )

    learn.fit_one_cycle(100)

    # Predict using the MLP model
    mlp_test_dl = dls.test_dl(X_test)
    mlp_preds, _ = learn.get_preds(dl=mlp_test_dl)
    mlp_test_pred = torch.round(mlp_preds).squeeze()
    mlp_test_acc = accuracy_score(y_test, mlp_test_pred)
    print("MLP Test Accuracy of best model:", mlp_test_acc)

    # Evaluate over the train set
    mlp_train_dl = dls.test_dl(X_train)
    mlp_train_preds, _ = learn.get_preds(dl=mlp_train_dl)
    mlp_train_pred = torch.round(mlp_train_preds).squeeze()
    mlp_train_acc = accuracy_score(y_train, mlp_train_pred)
    print("MLP Train Accuracy of best model:", mlp_train_acc)

    # Also save the MLP model
    mlp_model_path = join(CLASSIFIER_DIR, f"{dataset}_mlp_model.pkl")
    learn.export(mlp_model_path)


def view_results():
    # Load the datasets
    X_train = pd.read_csv(join(CLASSIFIER_DIR, f"X_train.csv"))
    X_test = pd.read_csv(join(CLASSIFIER_DIR, f"X_test.csv"))
    y_train = pd.read_csv(join(CLASSIFIER_DIR, f"y_train.csv"))
    y_test = pd.read_csv(join(CLASSIFIER_DIR, f"y_test.csv"))

    # Load the model
    rf_model_path = join(CLASSIFIER_DIR, f"dataset_hybrid_1000_deg15_rf_model.pkl")
    mlp_model_path = join(CLASSIFIER_DIR, f"dataset_hybrid_1000_deg15_mlp_model.pkl")
    with open(rf_model_path, "rb") as f:
        rf_model = pickle.load(f)

    mlp_model = load_learner(mlp_model_path)

    # Evaluate on the held-out train and test set
    train_pred = rf_model.predict(X_train)
    test_pred = rf_model.predict(X_test)

    df = pd.DataFrame({'Actual': y_test.values.flatten(), 'RF Predicted': test_pred})

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("Train Accuracy of loaded model:", train_acc)
    print("Test Accuracy of loaded model:", test_acc)
    print()

    X_train["target"] = y_train
    X_test["target"] = y_test
    # Do the same for the MLP model
    mlp_train_dl = mlp_model.dls.test_dl(X_train)
    mlp_test_dl = mlp_model.dls.test_dl(X_test)
    mlp_train_preds, _ = mlp_model.get_preds(dl=mlp_train_dl)
    mlp_test_preds, _ = mlp_model.get_preds(dl=mlp_test_dl)
    mlp_train_pred = torch.round(mlp_train_preds).squeeze()
    mlp_test_pred = torch.round(mlp_test_preds).squeeze()
    mlp_train_acc = accuracy_score(y_train, mlp_train_pred)
    mlp_test_acc = accuracy_score(y_test, mlp_test_pred)

    print("MLP Train Accuracy of loaded model:", mlp_train_acc)
    print("MLP Test Accuracy of loaded model:", mlp_test_acc)

    df["MLP Predicted"] = mlp_test_pred.numpy()
    df.to_csv(join(CLASSIFIER_DIR, f"test_predictions.csv"), index=False)


def run_baseline():
    """
    Run sympy's baseline on the datasets and plot the results.
    """
    import pickle
    if os.path.exists("temp.pkl"):
        results = pickle.load(open("temp.pkl", "rb"))
        return results

    # Load the datasets
    df1 = pd.read_csv(join("data", "dataset_100_5_3.csv"))
    df2 = pd.read_csv(join("data", "dataset_300_vary.csv"))

    results = {}

    # Iterate through the datasets and run sympy's factor on each polynomial, and check if the result is decomposable or not
    for j, df in enumerate([df1, df2]):
        results[DATASETS[j]] = {}

        for i in tqdm(range(df.shape[0]), desc="Running sympy's baseline", total=df.shape[0]):
            g = sp.Poly(df.iloc[i]["P(x)"])
            h = sp.Poly(df.iloc[i]["Q(x)"])
            # Compose the polynomials
            f = sp.simplify(g.subs(list(g.free_symbols)[0], h.as_expr()))
            # Factor the polynomial
            factored_f = sp.decompose(f)
            # Check if the factored polynomial is decomposable or not
            results[DATASETS[j]][i+1] = {
                "success": 1 if len(factored_f) > 1 else 0,
                "run_success": 1
            }

    with open("temp.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


def main():
    # Get the model's results
    results1, results2, results3 = get_all_results("output_best_hp"), run_baseline(), get_all_results("output_dirs")

    # classify_results(results1)
    # view_results()

    # Set the amount of runs
    plot_results(results1, results2, results3, 6, style="line")


if __name__ == "__main__":
    main()
