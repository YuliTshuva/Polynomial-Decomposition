"""
Yuli Tshuva
Creating the dataset for testing the models.
The dataset will be in format of a csv of strings out of comfort.
"""

# Imports
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
from os.path import join
from functions import *
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"


def random_int(scale, allow_zero=False):
    num = int(random.random() * 2 * scale - scale)
    if not allow_zero:
        while num == 0:
            num = int(random.random() * 2 * scale - scale)
    return num


# Constants
x = sp.symbols("x")


def baseline_dataset():
    # Set the df
    df = pd.DataFrame(columns=["P(x)", "Q(x)"])

    # Run up the degrees
    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 7]]
    scales = range(10, 201, 10)
    repetitions = 3
    for deg_q, deg_p in combinations:
        # Iterate over the scales
        for scale in scales:
            # Repeat 5 times
            for _ in range(repetitions):
                # Generate t
                p = sum([random_int(scale, allow_zero=(i != deg_p)) * x ** i for i in range(deg_p + 1)])
                q = sum([random_int(scale, allow_zero=(i != deg_q)) * x ** i for i in range(deg_q + 1)])
                # Add a row to the df
                df.loc[df.shape[0]] = [p, q]

    # Save the df to a csv file
    df.to_csv(join("data", "dataset_300_vary.csv"), index=False, columns=["P(x)", "Q(x)"])


def hybrid_dataset():
    """
    Creat a dataset with half the polynomials being decomposable and half not.
    Working with degree = 15.
    """
    # Set the df
    df = pd.DataFrame(columns=["R(x)", "Decomposable", "P(x)", "Q(x)"])

    # Set parameters
    degree = 15
    deg_q = 3
    deg_p = degree // deg_q
    low_scale = 100
    high_scale = 1e12
    n_samples = 1000

    # Set a dict for the distribution of coefficients
    coeffs_distribution = {deg: [] for deg in range(degree + 1)}

    for _ in range(n_samples // 2):
        # Decomposable case
        q = sum([random_int(low_scale, allow_zero=(i != deg_q)) * x ** i for i in range(deg_q + 1)])
        p = sum([random_int(low_scale, allow_zero=(i != deg_p)) * x ** i for i in range(deg_p + 1)])
        # Multiply them
        r = p.subs(x, q).expand().simplify()
        # Get r's coefficients as a list
        r_coeffs = [int(coef) for coef in r.as_poly(x).all_coeffs()]
        # Update the coeffs_distribution dict
        for deg, coef in enumerate(reversed(r_coeffs)):
            coeffs_distribution[deg].append(coef)
        # Add a row to the df
        df.loc[df.shape[0]] = [r, 1, p, q]

    # Calculate the standard deviation of each coefficient's distribution
    coeffs_distribution = {deg: np.std(np.array(coeffs_distribution[deg])) for deg in coeffs_distribution}

    for _ in range(n_samples // 2):
        # Non-decomposable case
        # Generate a random polynomial of degree 15
        r = sum([int(random.gauss(sigma=coeffs_distribution[i]/3)) * x ** i for i in range(degree + 1)])
        # Add a row to the df
        df.loc[df.shape[0]] = [r, 0, None, None]

    # Save the df to a csv file
    df.to_csv(join("data", "dataset_hybrid_1000_deg15.csv"), index=False)


def check_non_decomposable():
    """
    Check that the non-decomposable polynomials are indeed non-decomposable.
    """
    # Load the dataset
    df = pd.read_csv(join("data", "dataset_hybrid_1000_deg15.csv"))

    df["Decomposable"] = df["Decomposable"].astype(int)
    df = df[df["Decomposable"] == 0]

    for i in range(df.shape[0]):
        with open("job2_outs.txt", 'w') as f:
            f.write(f"Checking row {i} / {df.shape[0] - 1}\n")
        row = df.iloc[i]
        r = row["R(x)"]
        r = sp.sympify(r)
        if is_decomposable_kozen_landau(r, x):
            Exception(f"[Row {i}]: Polynomial {r} is decomposable, but should not be!")


def compare_coefficients():
    df = pd.read_csv(join("data", "dataset_hybrid_1000_deg15.csv"))
    df["Decomposable"] = df["Decomposable"].astype(int)
    df1 = df[df["Decomposable"] == 1]["R(x)"]
    df2 = df[df["Decomposable"] == 0]["R(x)"]

    # Plot the distribution of the coefficients of each power coefficient in df1 and df2
    max_degree = 15

    # Create the subplots
    plt.subplots(4, 4, figsize=(20, 20))

    # Analyze each degree
    for deg in range(max_degree + 1):
        print("Analyzing degree", deg)
        coeffs1 = []
        coeffs2 = []
        for r in df1:
            r = sp.sympify(r)
            coeff = r.coeff(x, deg)
            coeffs1.append(int(coeff))
        for r in df2:
            r = sp.sympify(r)
            coeff = r.coeff(x, deg)
            coeffs2.append(int(coeff))
        plt.subplot(4, 4, deg + 1)
        plt.hist(coeffs1, bins=50, alpha=0.5, label='Decomposable', color='turquoise', density=True)
        plt.hist(coeffs2, bins=50, alpha=0.5, label='Non-Decomposable', color='red', density=True)
        plt.title(f'Coefficient Distribution for x^{deg}', fontsize=24)
        plt.xlabel('Coefficient Value', fontsize=18)
        plt.ylabel('Density', fontsize=18)
        plt.legend()

    # Adjust layout and show plot
    plt.suptitle("Coefficient Distributions for Decomposable vs Non-Decomposable Polynomials", fontsize=35, y=0.99)
    plt.tight_layout(pad=1.2)
    plt.savefig(join("plots", 'hybrid_new_coefficient_distributions.png'))
    plt.show()


if __name__ == "__main__":
    compare_coefficients()
