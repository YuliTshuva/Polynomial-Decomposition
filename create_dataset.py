"""
Yuli Tshuva
Creating the dataset for testing the models.
The dataset will be in format of a csv of strings out of comfort.
"""

# Imports
import pandas as pd
from os.path import join
from functions import *


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
    combinations = [[3, 5], [3, 6], [3, 4], [4, 4], [2, 8]]
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
    scale = 100
    n_samples = 200

    for _ in range(n_samples // 2):
        # Decomposable case
        q = sum([random_int(scale, allow_zero=(i != deg_q)) * x ** i for i in range(deg_q + 1)])
        p = sum([random_int(scale, allow_zero=(i != deg_p)) * x ** i for i in range(deg_p + 1)])
        # Multiply them
        r = p.subs(q, x).simplify()
        # Add a row to the df
        df.loc[df.shape[0]] = [r, 1, p, q]

    for _ in range(n_samples // 2):
        # Non-decomposable case
        # Generate a random polynomial of degree 15
        r = sum([random_int(scale, allow_zero=(i != degree)) * x ** i for i in range(degree + 1)])
        # Add a row to the df
        df.loc[df.shape[0]] = [r, 0, None, None]

    # Save the df to a csv file
    df.to_csv(join("data", "dataset_hybrid_200_deg15.csv"), index=False)


def check_non_decomposable():
    """
    Check that the non-decomposable polynomials are indeed non-decomposable.
    """
    # Load the dataset
    df = pd.read_csv(join("data", "dataset_hybrid_200_deg15.csv"))

    df["Decomposable"] = df["Decomposable"].astype(int)
    df = df[df["Decomposable"] == 0]

    for i in range(df.shape[0]):
        with open("job2_outs.txt", 'a') as f:
            f.write(f"Checking row {i} / {df.shape[0]}...")
        row = df.iloc[i]
        r = row["R(x)"]
        r = sp.sympify(r)
        if is_decomposable_kozen_landau(r, x):
            Exception(f"[Row {i}]: Polynomial {r} is decomposable, but should not be!")


if __name__ == "__main__":
    # Create the hybrid dataset
    check_non_decomposable()
