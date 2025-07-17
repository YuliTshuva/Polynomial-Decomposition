"""
Yuli Tshuva
Find the closest solution to the one my algorithm outputted.
"""

# Imports
from os.path import join
from sympy import symbols, Poly, sympify
import re
import pickle
from scipy.optimize import minimize
import subprocess


# Constants
# TARGET_DIR = join("output_dirs", "train_14", "thread_0")


def find_closest_solution(TARGET_DIR, DEGREE, DEG_Q):
    MODEL_PATH = join(TARGET_DIR, "model.pth")
    SOLUTION_PATH = join(TARGET_DIR, "solution.pkl")
    POLY_FILE = join(TARGET_DIR, "polynomials.txt")

    # Define the variable
    x = symbols('x')

    def expression_to_polynomial(expression):
        poly_str_fixed = re.sub(r'(\d)(x)', r'\1*\2', expression)
        poly_str_fixed = poly_str_fixed.replace('^', '**')
        poly_expr = sympify(poly_str_fixed)
        poly = Poly(poly_expr, x)
        # Get the coefficients of R(x) from the highest degree to the lowest
        coeffs = list(poly.all_coeffs())
        return coeffs

    # Extract R(x) from the polynomials file
    with open(POLY_FILE, 'r') as f:
        lines = f.readlines()

    # Extract expressions
    found_px = expression_to_polynomial(lines[7].split("P(x): ")[1])
    found_qx = expression_to_polynomial(lines[8].split("Q(x): ")[1])

    # Load the close form solution
    with open(SOLUTION_PATH, "rb") as f:
        solution, ps, qs = pickle.load(f)

    # Covert to list type
    ps, qs = list(ps)[::-1], list(qs)[::-1]

    # Set a list of expected values
    target = found_px + found_qx

    # Get the function
    free_vars = []
    for a in ps + qs:
        if a not in solution:
            solution[a] = a
            free_vars.append(a)
    expression = [solution[a] for a in ps + qs]

    # Calculate the error
    error = 0
    for i in range(len(ps), len(target)):
        error += (expression[i] - target[i]) ** 2

    # Find approximation for the free variables that minimize the error
    def objective_function(free_vars_values):
        specific_error = error
        for i, var in enumerate(free_vars):
            specific_error = specific_error.subs(var, free_vars_values[i])
        return specific_error.evalf()  # Evaluate the expression to a float

    # Initial guess for the free variables
    initial_guess = [1.0] * len(free_vars)
    # Minimize the objective function
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', options={
        'xatol': 1e-10,  # Tighter tolerance on parameter changes
        'fatol': 1e-10,  # Tighter tolerance on function value changes
        'maxiter': 20000,  # Allow more iterations
        'maxfev': 50000,  # Allow more function evaluations
    })
    # Extract the optimized values
    optimized_values = result.x
    # Update the solution
    for key in solution:
        for i, var in enumerate(free_vars):
            solution[key] = solution[key].subs(var, optimized_values[i])

    # Construct the final solution
    for i, var in enumerate(free_vars):
        solution[var] = float(optimized_values[i])

    P = [solution[a] for a in ps]
    Q = [solution[a] for a in qs]

    # Write the solution
    with open(POLY_FILE, "a") as f:
        f.write("\nClosest solution:\n")
        f.write(f"Closest P(x): {P}" + "\n")
        f.write(f"Closest Q(x): {Q}" + "\n")
