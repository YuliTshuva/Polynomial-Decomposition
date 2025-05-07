"""
Yuli Tsvhua
Implementing polynomial decomposition using PyTorch.
"""

# Imports
import torch
import torch.optim as optim

from model import PolynomialDecomposition
import sympy as sp
from sympy import expand
from functions import *
from tqdm.auto import tqdm
from os.path import join
from torch.optim.lr_scheduler import StepLR

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPOCHS = int(1e6)
SAMPLES = int(1e3)
EARLY_STOPPING, MIN_CHANGE, GAMMA = int(1e3), 1, 0.4
DEG_P, DEG_Q = 5, 3
DEGREE = DEG_P * DEG_Q
VAR = 2
LR, MIN_LR = 1e-1, 1e-5
OUTPUT_FILE = join("output", "polynomials2.txt")
MODEL_PATH = join("output", "model2.pth")
PLOT_PATH = join("output", "loss2.png")

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
model = PolynomialDecomposition(degree=DEGREE, deg_q=3).to(DEVICE)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

# Define loss function
loss_fn = torch.nn.MSELoss()

# Define training data
X = torch.linspace(-1 * VAR, VAR, SAMPLES, dtype=torch.float64).to(DEVICE)
f = sp.lambdify(x, R, modules=["numpy"])
y = torch.tensor(f(X.cpu().numpy()), dtype=torch.float64, requires_grad=True).to(DEVICE)

# Set a list of loss functions
losses = []
count, min_loss, best_epoch = 0, float("inf"), -1
current_lr = LR
# Build a training loop
for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch", total=EPOCHS):
    model.train()

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
    if loss.item() + MIN_CHANGE < min_loss:
        best_epoch = epoch
        count = 0
        min_loss = loss.item()
        # Save the model in the output directory
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        count += 1

    if count > EARLY_STOPPING:
        count = 0
        if current_lr > MIN_LR:
            current_lr *= GAMMA
            scheduler.step()
            print(f"Learning rate decreased to {current_lr:.6f}")
            if current_lr <= 1e-2:
                MIN_CHANGE = 0.01
        else:
            print(f"Early stopping.")
            break

    # Plot the loss
    if epoch % 2000 == 0:
        plot_loss(losses, save=PLOT_PATH)

    # Evaluation
    if epoch % (3 * EPOCHS // 100) == 0:
        model.eval()
        P_weights = model.P.detach().cpu().numpy()
        Q_weights = model.Q.detach().cpu().numpy()
        restored_p = sum(P_weights[i] * x ** i for i in range(len(P_weights)))
        restored_q = sum(Q_weights[i] * x ** i for i in range(len(Q_weights)))

        with open(OUTPUT_FILE, "a") as f:
            f.write("\n" + "-" * 50 + "\n")
            f.write(f"Epoch {epoch} ({int(epoch / EPOCHS * 100)}%): Loss = {loss.item():.3f}\n")
            f.write(f"Restored P(x): {present_result(restored_p)}\n")
            f.write(f"Restored Q(x): {present_result(restored_q)}\n")
            f.write(f"Restored R(x): {present_result(expand(restored_p.subs(x, restored_q)))}\n")

plot_loss(losses, save=PLOT_PATH)

# Load the best model
model.load_state_dict(torch.load(MODEL_PATH))

# Final evaluation
model.eval()
P_weights = model.P.detach().cpu().numpy()
Q_weights = model.Q.detach().cpu().numpy()
restored_p = sum(P_weights[i] * x ** i for i in range(len(P_weights)))
restored_q = sum(Q_weights[i] * x ** i for i in range(len(Q_weights)))
with open(OUTPUT_FILE, "a") as f:
    f.write("\n" + "-" * 50 + "\n")
    f.write(f"Epoch {best_epoch} ({(best_epoch / EPOCHS * 100):.3f}%): Loss = {min_loss:.3f}\n")
    f.write(f"Restored P(x): {present_result(restored_p)}\n")
    f.write(f"Restored Q(x): {present_result(restored_q)}\n")
    f.write(f"Restored R(x): {present_result(expand(restored_p.subs(x, restored_q)))}\n")
