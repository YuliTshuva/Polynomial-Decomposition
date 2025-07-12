import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sympy as sp
from sympy import expand
import torch
import torch.nn as nn
import time
import math

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


def create_efficient_model(exp_list):
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

    # Open the model file
    with open("efficient_model.py", "w") as file:
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
    for p in range(1, n + 1):
        if n % p == 0:
            q = math.pow(n // p, 1/deg_p)
            if q - int(q) < 0.0001:
                if revert:
                    return -p, int(q)
                return p, int(q)
