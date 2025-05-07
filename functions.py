import random
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
    plt.yscale("log")
    plt.plot(range(len(losses)), losses, color="salmon")
    plt.title("Loss Function", fontsize=20)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
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
