"""
Yuli Tshuva
Addressing the coefficients directly.
"""

# Imports
import torch
import torch.optim as optim
from model import PolynomialSearch
import sympy as sp
from functions import *
from tqdm.auto import tqdm
from os.path import join

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPOCHS = int(1e5)
LR = 1e-1
EARLY_STOPPING, MIN_CHANGE = int(1e3), 1e-1
DEG_P, DEG_Q = 5, 3
DEGREE = DEG_P * DEG_Q
OUTPUT_FILE = join("output", "polynomials5.txt")
LOSS_PLOT = join("output", "loss5.png")
MODEL_PATH = join("output", "model5.pth")

# Define variable
x = sp.symbols('x')
deg_r = 0
while deg_r < DEGREE:
    # Define the polynomials
    P, Q = generate_polynomial(degree=DEG_P, var=x), generate_polynomial(degree=DEG_Q, var=x)
    R = expand(P.subs(x, Q))
    # Get r's coefficients
    Rs = torch.tensor(sp.Poly(R, x).all_coeffs()[::-1], dtype=torch.float64, requires_grad=True).to(DEVICE)

    # Find the degree of R
    deg_r = R.as_poly(x).degree()

# Create output file in the output directory
with open(OUTPUT_FILE, "w") as f:
    f.write(f"P(x): {present_result(P)}\n")
    f.write(f"Q(x): {present_result(Q)}\n")
    f.write(f"R(x): {present_result(R)}\n")

# Initialize the model
model = PolynomialSearch(degree=DEGREE, deg_q=DEG_Q).to(DEVICE)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Define loss function
loss_fn = torch.nn.L1Loss()

# Set a list of loss functions
losses = []
count, min_loss, best_epoch = 0, float("inf"), -1
# Build a training loop
for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch", total=EPOCHS):
    # Training mode
    model.train()

    # Forward pass
    optimizer.zero_grad()
    output = model()

    # Compute loss
    loss = loss_fn(output, Rs)
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
        if LR == 1e-5:
            print(f"Early stopping at epoch {epoch}")
            break
        else:
            print(f"Reducing learning rate at epoch {epoch}")
            LR /= 10
            scheduler.step()
            count = 0

            EARLY_STOPPING *= 10
            EARLY_STOPPING = max(1e-3, EARLY_STOPPING)
            MIN_CHANGE /= 10

    # Plot the loss
    if epoch % 400 == 0:
        plot_loss(losses, save=LOSS_PLOT)

    # Evaluation
    if epoch % (1*EPOCHS // 100) == 0:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\n" + "-" * 50 + "\n")
            f.write(f"Epoch {epoch} ({int(epoch / EPOCHS * 100)}%): Loss = {loss.item():.3f}\n")
            f.write(f"R(x): {torch.round(output, decimals=2)}\n")

plot_loss(losses, save=LOSS_PLOT)

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
