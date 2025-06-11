"""
Yuli Tshuva
Address the cases in 09 where it doesn't work by forcing the coefficient of the largest degrees to behave.
"""

# Imports
import torch.optim as optim
from model import PolynomialSearch
from functions import *
from os.path import join
import os
import shutil
import sympy as sp
import pickle
import random
import time
from tqdm.auto import tqdm
import importlib.util
import sys
from constants import *

# Constants
RESET_ENVIRONMENT = False
SHOW_EVERY = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_DIR = join("../output_dirs", "train_13")
THREAD_DIR = lambda i: join(WORKING_DIR, f"thread_{i}")
OUTPUT_FILE = lambda i: join(THREAD_DIR(i), f"polynomials.txt")
LOSS_PLOT = lambda i: join(THREAD_DIR(i), f"loss.png")
MODEL_PATH = lambda i: join(THREAD_DIR(i), f"model.pth")
STOP_THREAD_FILE = lambda i: join(THREAD_DIR(i), "stop.txt")
DIVISORS_DATA = join("../data", "divisors.pkl")
TIMEOUT = 60 * 10  # 10 minutes

# Read the divisors data
with open(DIVISORS_DATA, "rb") as f:
    DIVISORS = pickle.load(f)

# Sample a degree for R
DEGREE = 18
while DEGREE == 18:
    DEGREE = random.choice(list(DIVISORS.keys())[:20])
print("Degree of R:", DEGREE)
WEIGHTS = torch.tensor([1] * DEGREE + [LAMBDA2]).to(DEVICE)

# Define variable
x = sp.symbols('x')
deg_r = 0
while deg_r < DEGREE:
    # Define the polynomials
    R = generate_polynomial(degree=DEGREE, var=x, scale=100)
    # Get r's coefficients
    Rs = torch.tensor(sp.Poly(R, x).all_coeffs()[::-1], dtype=torch.float64, requires_grad=True).to(DEVICE)
    # Find the degree of R
    deg_r = R.as_poly(x).degree()


def train(train_id: int):
    # Create the working directory
    os.makedirs(THREAD_DIR(train_id), exist_ok=True)

    # Set deg_q by train_id
    deg_q = train_id

    # Create output file in the output directory
    initial_string = "Generated Polynomials:\n"
    initial_string += f"R(x): {present_result(R)}\n"

    # Initialize the model
    model = PolynomialSearch(degree=DEGREE, deg_q=deg_q).to(DEVICE)
    # Get the model's expression list
    exp_list = model.rs
    # Create the efficient version of the model
    create_efficient_model(exp_list)

    # Remove cached module if it exists
    if "efficient_model" in sys.modules:
        del sys.modules["efficient_model"]
    spec = importlib.util.spec_from_file_location("efficient_model", EFFICIENT_MODEL_PATH)
    efficient_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(efficient_model)
    EfficientPolynomialSearch = efficient_model.EfficientPolynomialSearch
    # Create an instance of the model
    model = EfficientPolynomialSearch(degree=DEGREE, deg_q=deg_q).to(DEVICE)

    # Initialize the model parameters
    with open(OUTPUT_FILE(train_id), "w") as f:
        f.write(initial_string)

    # Define the optimizer
    lr, min_change = LR, MIN_CHANGE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Define the loss function
    loss_fn = lambda inputs: weighted_l1_loss(inputs[0], inputs[1], WEIGHTS)

    # Print stating message
    print(f"[{get_time()}][Thread {train_id}]: Started training.")

    # Set a list of loss functions
    losses = []
    count, min_loss = 0, float("inf")
    # Build a training loop
    for epoch in tqdm(range(EPOCHS)):
        # Training mode
        model.train()

        # Forward pass
        optimizer.zero_grad()
        output = model()

        # Compute loss
        if epoch >= START_REGULARIZATION:
            loss = loss_fn([output, Rs]) + LAMBDA1 * model.sparse_optimization()
        else:
            loss = loss_fn([output, Rs])
        losses.append(loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if loss.item() + min_change < min_loss:
            count = 0
            min_loss = loss.item()
            # Save the model in the output directory
            torch.save(model.state_dict(), MODEL_PATH(train_id))

            with open(OUTPUT_FILE(train_id), "w") as f:
                f.write(initial_string)
                f.write("\n" + "-" * 50 + "\n")
                f.write(f"Epoch {epoch}: Loss = {loss.item():.3f}\n")
                f.write(f"P(x): {torch.round(model.P, decimals=3).tolist()[::-1]}\n")
                f.write(f"Q(x): {torch.round(model.Q, decimals=3).tolist()[::-1]}\n")
                f.write(f"R(x): {torch.round(output, decimals=2).tolist()[::-1]}\n")
                diffs = torch.round(torch.abs((output - Rs) * WEIGHTS), decimals=3).tolist()[::-1]
                f.write(f"diffs: {diffs}\n")
        else:
            count += 1

        # Early stopping
        if count > EARLY_STOPPING:
            if lr <= MIN_LR:
                print(f"[{get_time()}][Thread {train_id}]: Early stopping at epoch {epoch}")
                break
            else:
                lr = lr / 10
                print(f"[{get_time()}][Thread {train_id}]: Reducing learning rate to {lr} at epoch {epoch}")
                scheduler.step()
                count = 0
                min_change /= 10

        # Plot the loss
        if epoch % SHOW_EVERY == 0:
            if epoch <= 500:
                plot_loss(losses, save=LOSS_PLOT(train_id))
            else:
                plot_loss(losses, save=LOSS_PLOT(train_id), plot_last=300)

        if os.path.exists(STOP_THREAD_FILE(train_id)):
            print(f"[{get_time()}][Thread {train_id}]: Stopping thread.")
            os.remove(STOP_THREAD_FILE(train_id))
            break

        if min_loss < 2:
            print(f"[{get_time()}][Thread {train_id}]: Found a solution.")
            return True

    return False


def check_solution(qs):
    # We already have R
    q = sum([qs[i] * x ** i for i in range(len(qs))])
    # Find p such that p(q) = R
    ps = sp.var(f"p0:{DEGREE // (len(qs) - 1) + 1}")
    p = sum([ps[i] * x ** i for i in range(len(ps))])
    # Create the equation
    eq = sp.Eq(p.subs(x, q), R)
    # Solve the equation
    sol = sp.solve(eq, ps, dict=True)
    # Check if the solution is valid
    if sol:
        return sol, ps
    else:
        return False, None


def main():
    # Delete the results from the old run
    if RESET_ENVIRONMENT:
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)

    # Run the threads for this run
    divisors = DIVISORS[DEGREE]
    divisors = [d for i, d in enumerate(divisors) if i < (len(divisors) + 1) // 2 and d <= 10]

    # Set a timer for timeout
    train(random.sample(divisors, 1)[0])


if __name__ == "__main__":
    main()
