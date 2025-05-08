"""
Yuli Tsvhua
Integer Case.
"""

# Imports
import torch
import torch.optim as optim
from model import *
import sympy as sp
from sympy import expand
from functions import *
from tqdm.auto import tqdm
from os.path import join

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPOCHS = int(1e6)
SAMPLES = int(5e2)
EARLY_STOPPING, MIN_CHANGE = int(3e4), 0.01
DEG_P, DEG_Q = 5, 3
DEGREE = DEG_P * DEG_Q
VAR = 2
LR = 1e-3
LAMBDA = VAR ** (DEGREE//2)
BATCH_SIZE = 16
OUTPUT_FILE = join("output", "polynomials4.txt")
PLOT1_PATH = join("output", "loss4.png")
PLOT2_PATH = join("output", "losses4.png")
MODEL_PATH = join("output", "model4.pth")

# Define variable
x = sp.symbols('x')

deg_r = 0
while deg_r < DEGREE:
    # Define the polynomials
    P, Q = generate_polynomial(degree=DEG_P, var=x), generate_polynomial(degree=DEG_Q, var=x)
    R = expand(P.subs(x, Q))
    # Find the degree of R
    deg_r = R.as_poly(x).degree()

# Create output file in the output directory
with open(OUTPUT_FILE, "w") as f:
    f.write(f"P(x): {present_result(P)}\n")
    f.write(f"Q(x): {present_result(Q)}\n")
    f.write(f"R(x): {present_result(R)}\n")

# Initialize the model
model = PolynomialDecomposition(degree=DEGREE, deg_q=3).to(DEVICE)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# Define loss function
loss_reg_fn = custom_loss
loss_mse_fn = torch.nn.MSELoss()

# Define training data
X = torch.linspace(-1 * VAR, VAR, SAMPLES, dtype=torch.float64).to(DEVICE)
f = sp.lambdify(x, R, modules=["numpy"])
y = torch.tensor(f(X.cpu().numpy()), dtype=torch.float64, requires_grad=True).to(DEVICE)

# Set a list of loss functions
losses = []
mse_losses, reg_losses = [], []
count, min_loss, best_epoch = 0, float("inf"), -1
# Build a training loop
for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch", total=EPOCHS):
    model.train()

    # Forward pass
    optimizer.zero_grad()
    output = model(X)

    # Compute loss
    mse_loss, reg_loss = loss_mse_fn(output, y), loss_reg_fn(model)
    # loss = mse_loss + LAMBDA * reg_loss
    loss = mse_loss
    losses.append(loss.item())
    mse_losses.append(mse_loss.item())
    reg_losses.append(reg_loss.item() * LAMBDA)

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
        print(f"Early stopping at epoch {epoch}")
        break

    if epoch % int(1e4) == 0 and epoch > 3e4:
        # Round all weights of the model
        model.P.data = torch.round(model.P.data)
        model.Q.data = torch.round(model.Q.data)

    # Plot the loss
    if epoch % int(5e3) == 0:
        plot_loss(losses, save=PLOT1_PATH)
        plot_losses(mse_losses, reg_losses, label1="MSE Loss", label2="REG Loss", save=PLOT2_PATH)

    # Evaluation
    if epoch % int(1e4) == 0:
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

# Plot the losses for the last time
plot_loss(losses, save=PLOT1_PATH)
plot_losses(mse_losses, reg_losses, label1="MSE Loss", label2="REG Loss", save=PLOT2_PATH)

# Load the best model
model.load_state_dict(torch.load(MODEL_PATH))
# Round all weights of the model
model.P.data = torch.round(model.P.data)
model.Q.data = torch.round(model.Q.data)

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
