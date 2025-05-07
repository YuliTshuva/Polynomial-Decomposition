"""
Yuli Tsvhua
Implementing polynomial decomposition using PyTorch.
"""

# Imports
import torch
import torch.optim as optim
from model import Polynomial
import sympy as sp
from sympy import expand
from functions import *
from tqdm.auto import tqdm
from os.path import join

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
SAMPLES = int(1e3)
EARLY_STOPPING = int(1e3)
VAR = 2
LR = 1e-3
BATCH_SIZE = 256
EPOCHS = int(1e6) // BATCH_SIZE
OUTPUT_FILE = join("output", "polynomials3.txt")
MODEL_PATH = join("output", "model3.pth")
PLOT_PATH = join("output", "loss3.png")
DEG_P, DEG_Q = 5, 3
DEGREE = DEG_P * DEG_Q

# Define variable
x = sp.symbols('x')
# Define the polynomials
P, Q = generate_polynomial(degree=DEG_P, var=x), generate_polynomial(degree=DEG_Q, var=x)
# Define the polynomial R
R = expand(P.subs(x, Q))

# Find the degree of R
deg_r = R.as_poly(x).degree()
if deg_r < DEGREE:
    raise ValueError(f"Degree of R is less than {DEGREE}. Please increase the degree of P or Q.")

# Create output file in the output directory
with open(OUTPUT_FILE, "w") as f:
    f.write(f"P(x): {present_result(P)}\n")
    f.write(f"Q(x): {present_result(Q)}\n")
    f.write(f"R(x): {present_result(R)}\n")

# Initialize the model
P_x = Polynomial(deg=DEG_P).to(DEVICE)
Q_x = Polynomial(deg=DEG_Q).to(DEVICE)

# Define the optimizer
P_optimizer = optim.Adam(P_x.parameters(), lr=LR)
Q_optimizer = optim.Adam(Q_x.parameters(), lr=LR)

# Define loss function
loss_fn1 = torch.nn.MSELoss()
loss_fn2 = torch.nn.MSELoss()

# Define training data
X = torch.linspace(-1 * VAR, VAR, SAMPLES, dtype=torch.float64).to(DEVICE)
f = sp.lambdify(x, R, modules=["numpy"])
y = torch.tensor(f(X.cpu().numpy()), dtype=torch.float64, requires_grad=True).to(DEVICE)

# Set lists for the losses
P_losses, Q_losses = [], []
# Build a training loop
for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch", total=EPOCHS):
    # Forward pass
    P_x.train()
    Q_x.train()

    # P training
    for batch in range(BATCH_SIZE):
        # Compute the output of P(Q)
        P_out = P_x(Q_x(X))
        # Compute the loss
        loss = loss_fn1(P_out, y)
        # Zero the gradients
        P_optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update the weights
        P_optimizer.step()
        # Store the loss
        P_losses.append(loss.item())

    # Q training
    for batch in range(BATCH_SIZE):
        # Compute the output of Q(P)
        Q_out = P_x(Q_x(X))
        # Compute the loss
        loss = loss_fn2(Q_out, y)
        # Zero the gradients
        Q_optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update the weights
        Q_optimizer.step()
        # Store the loss
        Q_losses.append(loss.item())

    # Evaluate
    if epoch % 50 == 0:
        # Plot the losses
        plot_losses(l1=P_losses, l2=Q_losses, label1="P_loss", label2="Q_loss", save=PLOT_PATH, show=False)

        # Apply evaluation mode
        P_x.eval()
        Q_x.eval()
        # Extract the weights
        P_weights = P_x.P.detach().cpu().numpy()
        Q_weights = Q_x.P.detach().cpu().numpy()
        # Get the polynomials
        restored_p = sum(P_weights[i] * x ** i for i in range(len(P_weights)))
        restored_q = sum(Q_weights[i] * x ** i for i in range(len(Q_weights)))
        # Write in the output file
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n" + "-" * 50 + "\n")
            f.write(f"Epoch {epoch}\n")
            f.write(f"P(x): {present_result(restored_p)}\n")
            f.write(f"Q(x): {present_result(restored_q)}\n")
            f.write(f"R(x): {present_result(expand(restored_p.subs(x, restored_q)))}\n")
