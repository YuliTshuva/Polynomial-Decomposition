"""
Yuli Tshuva
In this file I'll rank all possible combinations rounding Q's coefficients to integers by the loss values.
"""

# Imports
import pandas as pd
import itertools
import numpy as np
from os.path import join
import torch
from sympy import symbols, Poly, sympify
import re
from efficient_model import EfficientPolynomialSearch

# Constants
TARGET_DIR = join("output_dirs", "train_12", "thread_2")
MODEL_PATH = join(TARGET_DIR, "model.pth")
SOLUTION_PATH = join(TARGET_DIR, "solution.pkl")
POLY_FILE = join(TARGET_DIR, "polynomials.txt")
DEGREE, DEG_Q = 15, 3

# Extract R(x) from the polynomials file
with open(POLY_FILE, 'r') as f:
    lines = f.readlines()
    qx = lines[2].split("Q(x): ")[1]
    rx = lines[3].split("R(x): ")[1]

x = symbols('x')
poly_str_fixed = re.sub(r'(\d)(x)', r'\1*\2', rx)
poly_str_fixed = poly_str_fixed.replace('^', '**')
poly_expr = sympify(poly_str_fixed)
poly = Poly(poly_expr, x)
# Get the coefficients of R(x) from the highest degree to the lowest
coeffs = np.array(poly.all_coeffs())

q_poly_str_fixed = re.sub(r'(\d)(x)', r'\1*\2', qx)
q_poly_str_fixed = q_poly_str_fixed.replace('^', '**')
q_poly_expr = sympify(q_poly_str_fixed)
q_poly = Poly(q_poly_expr, x)
# Get the coefficients of R(x) from the lowest degree to the highest
q_coeffs = np.array(q_poly.all_coeffs())[::-1]

# Initialize the model
model = EfficientPolynomialSearch(DEGREE, DEG_Q)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
# Create a predict function from the highest degree to the lowest
predict = lambda model: model().flip(0).detach().numpy()

# Calculate the original loss
original_loss = np.mean(np.abs(predict(model) - coeffs))

# Get P and Q
Q = model.Q.detach().numpy()

# Find all Q combinations
combinations = [list(bits) for bits in itertools.product([0, 1], repeat=len(Q))]

q_to_loss = {}
q_to_loss[str(round(Q[0], 3)) + ''.join(
    [f' + {round(q, 3)}*x**{i}' for i, q in enumerate(Q[1:], start=1)])] = original_loss
q_to_loss[str(round(q_coeffs[0], 3)) + ''.join(
    [f' + {round(q, 3)}*x**{i}' for i, q in enumerate(q_coeffs[1:], start=1)])] = 0
sign = np.sign(Q)
Q = np.abs(Q)
for combination in combinations:
    # Create a new Q by rounding the original Q coefficients
    new_Q = (Q + combination).astype(np.int64) * sign
    model.Q = torch.nn.Parameter(torch.tensor(new_Q, dtype=torch.float64))
    # Predict the new coefficients
    new_coeffs = predict(model)
    # Calculate the new loss
    new_loss = np.mean(np.abs(new_coeffs - coeffs))
    # Covert to expression typed list
    expression = str(new_Q[0]) + ''.join([f' + {q}*x**{i}' for i, q in enumerate(new_Q[1:], start=1)])
    # Add to the dict
    q_to_loss[expression] = new_loss

# Sort the combinations by loss
sorted_combinations = sorted(list(q_to_loss.keys()), key=lambda key: q_to_loss[key])
sorted_losses = [round(q_to_loss[key], 4) for key in sorted_combinations]
# Save the sorted combinations
df = pd.DataFrame([[a, b] for a, b in zip(sorted_combinations, sorted_losses)], columns=['Q', 'loss'])
df.to_csv(join(TARGET_DIR, "sorted_rounds.csv"), index=False)
