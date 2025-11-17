import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sympy as sp
from sympy import expand
import torch
import torch.nn as nn
import time
import math
import numpy as np

rcParams["font.family"] = "Times New Roman"
EFFICIENT_MODEL_PATH = "efficient_model.py"


def get_time():
    current_time = time.strftime(f"%H:%M:%S")
    return current_time


def present_result(result):
    return str(result).replace("**", "^").replace("*", "")


def generate_polynomial(degree, var, scale=3):
    coeffs = [int(random.random() * 2 * scale - scale) for _ in range(degree + 1)]
    polynomial = sum(coeffs[i] * var ** i for i in range(degree + 1))
    return polynomial


def plot_loss(losses, save=None, show=False, mode: ["log", "linear"] = "linear", plot_last=0, xticks=None):
    plt.figure()
    # Plot in logarithmic scale
    if mode == "log":
        plt.yscale("log")
    else:
        plt.ylim(0, max(losses[-plot_last:]) * 1.1)
    plt.plot(range(len(losses[-plot_last:])), losses[-plot_last:], color="salmon")
    plt.title("Loss Function", fontsize=20)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.grid(True)
    # Set the ticks labels to be 1000 to 1000 + plot_last
    if plot_last > 0:
        plt.xticks(ticks=range(0, plot_last, 50),
                   labels=[str(el) for el in range(len(losses) - plot_last - 1, len(losses) - 1, 50)])
    if xticks:
        plt.xticks(xticks, rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    if show:
        plt.show()
    plt.close()


def express_with_coefficients(ps, qs, var):
    """
    Generate a polynomial of the given degree with random coefficients.
    """
    # Set the degrees
    deg_p, deg_q = len(ps) - 1, len(qs) - 1

    # Set the polynomials
    p = sum(ps[i] * var ** i for i in range(deg_p + 1))
    q = sum(qs[i] * var ** i for i in range(deg_q + 1))

    # Set the polynomial R
    r = expand(p.subs(var, q))

    # Get the coefficients of R as a list, from the one of the lowest degree to the one of the highest degree
    coeffs = sp.Poly(r, var).all_coeffs()[::-1]
    coeffs = [str(c) for c in coeffs]

    for i, c in enumerate(coeffs):
        coeffs[i] = c.replace("**", "^")

    return coeffs


def create_efficient_model(exp_list, degree, deg_q):
    # Handle power sign
    for i, exp in enumerate(exp_list):
        exp_list[i] = exp.replace("^", "**")

    # Parse expressions
    for i in range(len(exp_list)):
        # Replace vars with params
        exp_list[i] = exp_list[i].replace("p", "self.P[")
        exp_list[i] = exp_list[i].replace("q", "self.Q[")
        # Create a string for receiving
        new_exp = ""
        # Iterate through the expression to find unclosed brackets
        j, condition = 0, True
        for j in range(len(exp_list[i])):
            char = exp_list[i][j]
            if char == "[":
                condition = False
                new_exp += char
                continue
            if not condition:
                if not char.isdigit():
                    new_exp += "]" + char
                    condition = True
                else:
                    new_exp += char
            else:
                new_exp += char

        if not condition:
            new_exp += "]"

        # Update the expression list
        exp_list[i] = new_exp

    # Implement the model's forward function
    forward_function = ""
    for i, exp in enumerate(exp_list):
        if i > 0:
            forward_function += " " * 8
        forward_function += f"output[{i}] = " + exp + "\n"

    # Open the model file
    with open("efficient_model_template.py", "r") as file:
        model = file.read()

    # Assign the forward in the model
    model = model.replace("# TO DO", forward_function)
    model = model.replace(" = degree", f" = {degree}")
    model = model.replace(" = deg_q", f" = {deg_q}")
    model = model.replace("EfficientPolynomialSearch", f"EfficientPolynomialSearch_{degree}_{deg_q}")
    model = model.replace("import torch\nfrom torch import nn", "")

    # Open the model file
    with open(f"efficient_model.py", "a") as file:
        file.write(model)


def weighted_l1_loss(output, target, weights):
    """
    Args:
        output: Tensor of shape (N,)
        target: Tensor of shape (N,)
        weights: Tensor of shape (N,)
    Returns:
        Scalar tensor (weighted L1 loss)
    """
    loss = torch.abs(output - target)
    weighted_loss = loss * weights
    return weighted_loss.mean()


def suggest_coefficients(n, deg_p):
    """
    Get the coefficient of the highest degree of the polynomial R and the degree of the polynomial P.
    Returns a suggested pair (p, q) such that p * q^deg_q = n.
    """
    revert = n < 0
    n = abs(n)
    start = time.time()
    for p in range(1, n + 1):
        if time.time() - start > 10:
            return None, None
        if n % p == 0:
            q = math.pow(n // p, 1 / deg_p)
            if abs(q - round(q)) < 0.0001:
                if revert:
                    return -p, round(q)
                return p, round(q)


def reduce_solution_variance(Q, P, R):
    # Ge the polynomials' variable
    x = list(Q.free_symbols)[0]
    # Get the coefficients of P
    P_coeffs = sp.Poly(P, x).all_coeffs()
    # Calculate the variance of P
    variance = np.max(np.abs(np.array(P_coeffs)))
    # Get P's degree
    deg_p = len(P_coeffs) - 1

    min_variance = {
        "Q": Q,
        "P": P
    }

    for sign in [-1, 1]:
        new_Q = Q
        strikes = 0
        while True:
            new_Q += sign
            ps = sp.symbols(f"p0:{deg_p + 1}")
            p = sum(ps[i] * x ** i for i in range(deg_p + 1))

            # Create the equation
            eq = sp.Eq(p.subs(x, new_Q), R)
            # Solve the equation
            sol = sp.solve(eq, ps, dict=True)
            if isinstance(sol, list):
                sol = sol[0]
            sol = np.array([sol[ps[i]] for i in range(deg_p + 1)])

            # Calculate the variance
            new_variance = np.max(np.abs(sol))
            if new_variance < variance:
                # If the variance is lower, update the coefficients
                variance = new_variance
                min_variance["Q"] = new_Q
                min_variance["P"] = sum(sol[i] * x ** i for i in range(deg_p + 1))
                continue
            else:
                strikes += 1
                if strikes == 3:
                    break

    return min_variance["Q"], min_variance["P"]


def is_decomposable_kozen_landau(f, x=None):
    """
    Returns True if f(x) is decomposable (exists f = g∘h)
    using Kozen–Landau style critical-point GCD tests.
    Does NOT return the decomposition.
    """
    if x is None:
        x = list(f.free_symbols)[0]

    n = sp.degree(f, gen=x)

    # 1. degrees of possible inner polynomial h
    divisors = [d for d in range(2, n) if n % d == 0]

    if not divisors:
        return False  # degree is prime → impossible to decompose

    fprime = sp.diff(f, x)

    # Find critical points (symbolic roots may be complicated)
    crit_points = sp.solve(sp.Eq(fprime, 0), x)

    # Cannot analyze if derivative has no roots (rare but possible)
    if len(crit_points) == 0:
        return False

    # 2. For each critical point, test its fiber
    for c in crit_points:
        fc = f.subs(x, c)
        g = sp.gcd(sp.simplify(fprime), sp.simplify(f - fc))

        # If gcd has degree ≥ 1 → structure exists → decomposable
        if sp.degree(g) >= 1:
            return True

    return False


def j_to_bool(index, j):
    if j == index:
        return "0"
    else:
        return "1"
