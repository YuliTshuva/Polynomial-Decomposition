"""
Yuli Tsvhua
Implementing polynomial decomposition using PyTorch.
"""

# Imports
import torch
import torch.optim as optim
from model import PolynomialDecomposition
import numpy as np
import sympy as sp
from sympy import expand
from functions import *
from tqdm.auto import tqdm
from os.path import join

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR = 1e-3
EPOCHS = int(1e5)
EARLY_STOPPING = 200
EPSILON = 1e-3
SAMPLES = 1000
VAR = 10
DEG_P, DEG_Q = 5, 3
DEGREE = DEG_P * DEG_Q
WEIGHTS = torch.linspace(1, DEGREE + 1, steps=DEGREE + 1).to(DEVICE)
WEIGHTS /= torch.sum(WEIGHTS)  # Normalize weights

# Define variable
x = sp.symbols('x')
# Define the polynomials
P, Q = generate_polynomial(degree=DEG_P, var=x), generate_polynomial(degree=DEG_Q, var=x)
R = expand(P.subs(x, Q))

# Present the polynomials
print("P(x):", present_result(P))
print("Q(x):", present_result(Q))
print("R(x):", present_result(R))

# Initialize the model
model = PolynomialDecomposition(degree=DEGREE).to(DEVICE)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# Define loss function
loss_fn = torch.nn.MSELoss()

# Define training data
X = torch.linspace(-1*VAR, VAR, SAMPLES, dtype=torch.float64).to(DEVICE)
f = sp.lambdify(x, R, modules=["numpy"])
y = torch.tensor(f(X.cpu().numpy()), dtype=torch.float64, requires_grad=True).to(DEVICE)

# Set a list of loss functions
losses, count = [], 0
# Build a training loop
model.train()
for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch", total=EPOCHS):
    # Forward pass
    optimizer.zero_grad()
    output = model(X)

    # Compute loss
    loss = loss_fn(output, y)
    losses.append(loss.item())

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Early stopping
    if epoch > 1:
        if abs(losses[-1] - losses[-2]) < EPSILON:
            count += 1
            if count > EARLY_STOPPING:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            count = 0

    # Plot the loss
    if epoch % 100 == 0:
        plot_loss(losses, save=join("output", "loss.png"))

# Evaluate the model
model.eval()

# Get the weights of the model
P_weights = model.P.detach().cpu().numpy()
Q_weights = model.Q.detach().cpu().numpy()

# Present the results
restored_p = sum(P_weights[i] * x ** i for i in range(len(P_weights)))
restored_q = sum(Q_weights[i] * x ** i for i in range(len(Q_weights)))
print("Restored P(x):", present_result(restored_p))
print("Restored Q(x):", present_result(restored_q))
print("Restored R(x):", present_result(expand(restored_p.subs(x, restored_q))))
