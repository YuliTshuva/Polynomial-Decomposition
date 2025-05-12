import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sympy as sp
from sympy import expand

rcParams["font.family"] = "Times New Roman"


def present_result(result):
    return str(result).replace("**", "^").replace("*", "")


def generate_polynomial(degree, var):
    scale = 3
    coeffs = [int(round(random.random() * 2 * scale - scale, 2)) for _ in range(degree + 1)]
    polynomial = sum(coeffs[i] * var ** i for i in range(degree + 1))
    return polynomial


def plot_loss(losses, save=None, show=False):
    plt.figure()
    # Plot in logarithmic scale
    # plt.yscale("log")
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


def plot_losses(l1, l2, label1, label2, save=None, show=False):
    plt.figure()
    # Plot in logarithmic scale
    plt.yscale("log")
    plt.plot(range(len(l1)), l1, color="hotpink", label=label1)
    plt.plot(range(len(l2)), l2, color="royalblue", label=label2)
    plt.title("Loss Function", fontsize=20)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.legend()
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
