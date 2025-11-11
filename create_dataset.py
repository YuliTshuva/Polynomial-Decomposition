"""
Yuli Tshuva
Creating the dataset for testing the models.
The dataset will be in format of a csv of strings out of comfort.
"""

# Imports
import sympy as sp
import pandas as pd
import random
from os.path import join

# Constants
x = sp.symbols("x")

# Set the df
df = pd.DataFrame(columns=["P(x)", "Q(x)"])


def random_int(scale, allow_zero=False):
    num = int(random.random() * 2 * scale - scale)
    if not allow_zero:
        while num == 0:
            num = int(random.random() * 2 * scale - scale)
    return num


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
# df.to_csv(join("data", "dataset_300_vary.csv"), index=False, columns=["P(x)", "Q(x)"])
