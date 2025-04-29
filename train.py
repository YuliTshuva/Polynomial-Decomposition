"""
Yuli Tsvhua
Implementing polynomial decomposition using PyTorch.
"""

# Imports
import torch
import torch.optim as optim
from model import PolynomialDecomposition, WeightedMSELoss
import sympy as sp
from sympy import expand
from functions import generate_polynomial, plot_loss
from tqdm.auto import tqdm
from os.path import join

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR = 0.001
EPOCHS = 1000
DEG_P, DEG_Q = 5, 3
DEGREE = DEG_P * DEG_Q
WEIGHTS = torch.linspace(1, DEGREE + 1, steps=DEGREE + 1).to(DEVICE)
WEIGHTS /= torch.sum(WEIGHTS)  # Normalize weights

# Define variable
x = sp.symbols('x')
# Define the polynomials
P, Q = generate_polynomial(degree=DEG_P, var=x), generate_polynomial(degree=DEG_Q, var=x)
R = expand(P.subs(x, Q))
# Get its coefficients
R_coeffs = torch.tensor(sp.Poly(R, x).all_coeffs()[::-1], dtype=torch.float64).to(DEVICE)

# Initialize the model
model = PolynomialDecomposition(degree=DEGREE).to(DEVICE)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# Define loss function
# loss_fn = WeightedMSELoss(WEIGHTS)
loss_fn = torch.nn.MSELoss()

# Set a list of loss functions
losses = []
# Build a training loop
for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch", total=EPOCHS):
    # Forward pass
    optimizer.zero_grad()
    output = model()

    # Compute loss
    loss = loss_fn(output, R_coeffs)
    losses.append(loss.item())

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Plot the loss
    if epoch % 20 == 0:
        plot_loss(losses, save=join("output", "loss.png"))
