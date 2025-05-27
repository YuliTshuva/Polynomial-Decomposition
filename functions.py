import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sympy as sp
from sympy import expand
import torch
import torch.nn as nn
import time

rcParams["font.family"] = "Times New Roman"
EFFICIENT_MODEL_PATH = "efficient_model.py"


def get_time():
    current_time = time.strftime(f"%H:%M:%S")
    return current_time


def present_result(result):
    return str(result).replace("**", "^").replace("*", "")


def generate_polynomial(degree, var):
    scale = 3
    coeffs = [int(round(random.random() * 2 * scale - scale, 2)) for _ in range(degree + 1)]
    polynomial = sum(coeffs[i] * var ** i for i in range(degree + 1))
    return polynomial


def plot_loss(losses, save=None, show=False, mode: ["log", "linear"] = "linear", plot_last=0):
    plt.figure()
    # Update the list
    losses = losses[-plot_last:]
    # Plot in logarithmic scale
    if mode == "log":
        plt.yscale("log")
    else:
        plt.ylim(0, max(losses) * 1.1)
    plt.plot(range(len(losses)), losses, color="salmon")
    plt.title("Loss Function", fontsize=20)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
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
