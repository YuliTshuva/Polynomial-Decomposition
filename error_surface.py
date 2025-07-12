"""
Yuli Tshuva
Plot the error surface for the model.
"""

# Imports
import torch
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from sklearn.decomposition import PCA
from efficient_model import EfficientPolynomialSearch
from functions import weighted_l1_loss
from tqdm.auto import tqdm
from matplotlib import rcParams


def plot_error_surface(working_dir, thread=0):
    OUTPUT_DIR = join(working_dir, f"thread_{thread}")
    DEGREE, DEG_Q = 15, 3
    N_SAMPLES = int(1e4)
    WEIGHTS = torch.tensor([1] * DEGREE + [1], dtype=torch.float64)
    rcParams["font.family"] = "Times New Roman"

    # Load the model
    model = EfficientPolynomialSearch(DEGREE, DEG_Q)
    model.load_state_dict(torch.load(join(OUTPUT_DIR, "model.pth")))

    # Read the coefficients from the output file
    with open(join(OUTPUT_DIR, "polynomials.txt"), "r") as f:
        lines = f.readlines()

    # Set the original polynomials
    original_P = [float(a) for a in lines[1].split("[")[1].split("]")[0].strip().split(", ")][::-1]
    original_Q = [float(a) for a in lines[2].split("[")[1].split("]")[0].strip().split(", ")][::-1]

    # Set P and Q
    P = [float(a) for a in lines[7].split("[")[1].split("]")[0].strip().split(", ")][::-1]
    Q = [float(a) for a in lines[8].split("[")[1].split("]")[0].strip().split(", ")][::-1]
    R = [float(a) for a in lines[9].split("[")[1].split("]")[0].strip().split(", ")][::-1]

    # Convert P and Q to tensors
    P = torch.tensor(P, dtype=torch.float64)
    Q = torch.tensor(Q, dtype=torch.float64)
    R = torch.tensor(R, dtype=torch.float64)

    # Calculate the model's loss
    model_loss = weighted_l1_loss(model(), R, WEIGHTS).item()

    # Generate a grid of points in the parameter space
    data = []
    losses = []
    for i in tqdm(range(N_SAMPLES), desc="Generating samples", total=N_SAMPLES):
        # Generate random uniform noise for P and Q in [-scale, scale]
        scale = 2
        P_noise = torch.rand_like(P, dtype=torch.float64) * 2 * scale - scale
        Q_noise = torch.rand_like(Q, dtype=torch.float64) * 2 * scale - scale

        # Create a new polynomial by adding noise to P and Q
        new_P = P + P_noise
        new_Q = Q + Q_noise

        # Add to the data
        data.append(torch.cat((new_P, new_Q)))

        # Assign in the model
        model.P.data = new_P
        model.Q.data = new_Q

        # Calculate the loss
        loss = weighted_l1_loss(model(), R, WEIGHTS)
        losses.append(loss.item())

    # Add the local and global minimums to the data
    data.append(torch.cat((P, Q)))
    data.append(torch.tensor(original_P + original_Q, dtype=torch.float64))

    # Add local minimum and global minimum losses
    losses += [model_loss, 0.0]

    # Apply PCA to the data
    data = torch.stack(data)
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data.numpy())

    # Convert losses to numpy array
    losses = np.array(losses)

    # Plot a heatmap of the loss surface as a scatter plot where the colors of each dot is determined by the loss value
    plt.figure(figsize=(10, 10))
    plt.tricontourf(data_reduced[:, 0], data_reduced[:, 1], np.log10(losses + 1e-10), levels=100, cmap='cool')
    plt.colorbar(label='Log Loss')

    # Add the dot of the original polynomials and the local minimum
    plt.scatter(data_reduced[-2, 0], data_reduced[-2, 1], color='red', label='Local Minimum', s=10)
    plt.scatter(data_reduced[-1, 0], data_reduced[-1, 1], color='green', label='Global Minimum', s=10)

    # Add title and labels
    plt.title('Error Surface', fontsize=20)
    plt.xlabel('First Dimension', fontsize=16)
    plt.ylabel('Second Dimension', fontsize=16)
    plt.savefig(join(OUTPUT_DIR, "error_surface.png"))
    plt.close()
