import random
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"


def present_result(result):
    print(str(result).replace("**", "^").replace("*", ""))


def generate_polynomial(degree, var):
    coeffs = [random.randint(-5, 5) for _ in range(degree + 1)]
    polynomial = sum(coeffs[i] * var ** i for i in range(degree + 1))
    return polynomial


def plot_loss(losses, save=None, show=False):
    plt.figure()
    plt.plot(range(len(losses)), losses, color="salmon")
    plt.title("Loss Function", fontsize=20)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    if save:
        plt.savefig(save)
    if show:
        plt.show()
    plt.close()
