"""
Yuli Tshuva
Trying to improve my algorithm for large coefficients by rounding Q's coeffs every XXXX epochs.
"""

# Imports
import torch
import torch.optim as optim
from model import PolynomialSearch
from functions import *
from os.path import join
import os
import threading
import shutil
import sympy as sp
import pickle
from constants import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Constants
RESET_ENVIRONMENT = False
NUM_THREADS = 1
SHOW_EVERY = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_DIR = join("output_dirs", "train_14")
STOP_FILE = join(WORKING_DIR, "stop.txt")
THREAD_DIR = lambda i: join(WORKING_DIR, f"thread_{i}")
OUTPUT_FILE = lambda i: join(THREAD_DIR(i), f"polynomials.txt")
LOSS_PLOT = lambda i: join(THREAD_DIR(i), f"loss.png")
MODEL_PATH = lambda i: join(THREAD_DIR(i), f"model.pth")
STOP_THREAD_FILE = lambda i: join(THREAD_DIR(i), "stop.txt")
TERMINATE_THREAD_FILE = lambda i: join(THREAD_DIR(i), "terminate.txt")
SUCCESS_FILE = join(WORKING_DIR, "success.txt")
DEG_P, DEG_Q = 5, 3
DEGREE = DEG_P * DEG_Q
WEIGHTS = torch.tensor([1] * DEGREE + [LAMBDA2]).to(DEVICE)
SCALE = 100

# Define variable
x = sp.symbols('x')
deg_r = 0
while deg_r < DEGREE:
    # Define the polynomials
    P, Q = generate_polynomial(degree=DEG_P, var=x, scale=SCALE), generate_polynomial(degree=DEG_Q, var=x, scale=SCALE)
    R = expand(P.subs(x, Q))
    # Get r's coefficients
    Rs = torch.tensor(sp.Poly(R, x).all_coeffs()[::-1], dtype=torch.float64, requires_grad=True).to(DEVICE)

    # Find the degree of R
    deg_r = R.as_poly(x).degree()


def train(train_id: int):
    # Create the working directory
    os.makedirs(THREAD_DIR(train_id), exist_ok=True)

    # Create output file in the output directory
    initial_string = "Generated Polynomials:\n"
    initial_string += f"P(x): {present_result(P)}\nQ(x): {present_result(Q)}\nR(x): {present_result(R)}\n"

    # Initialize the model
    model = PolynomialSearch(degree=DEGREE, deg_q=DEG_Q).to(DEVICE)
    # Get the model's expression list
    exp_list = model.rs
    # Create the efficient version of the model
    create_efficient_model(exp_list)
    # Import the efficient model created
    from efficient_model import EfficientPolynomialSearch
    model = EfficientPolynomialSearch(degree=DEGREE, deg_q=DEG_Q).to(DEVICE)

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
    count, min_loss = 0, float("inf")
    losses, epochs = [], [0]
    # Build a training loop
    epoch = -1
    while epoch < EPOCHS:
        # Add epochs
        epoch += 1

        # Training mode
        model.train()

        # Forward pass
        optimizer.zero_grad()
        output = model()

        # Compute loss
        if epoch >= START_REGULARIZATION:
            loss = loss_fn([output, Rs]) + LAMBDA1 * model.q_integer_regularization()
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
                f.write(f"P(x): {model.P.tolist()[::-1]}\n")
                f.write(f"Q(x): {model.Q.tolist()[::-1]}\n")
                f.write(f"R(x): {torch.round(output, decimals=3).tolist()[::-1]}\n")
                diffs = torch.abs(output - Rs).tolist()[::-1]
                f.write(f"diffs: {diffs}\n")
        else:
            count += 1

        # Early stopping
        if count > EARLY_STOPPING:
            if lr <= MIN_LR:
                print(f"[{get_time()}][Thread {train_id}]: Early stopping at epoch {epoch}")
                plot_loss(losses, save=LOSS_PLOT(train_id), mode="log", xticks=epochs)
                return
            else:
                lr = lr / 10
                print(f"[{get_time()}][Thread {train_id}]: Reducing learning rate to {lr} at epoch {epoch}")
                epochs.append(epoch)
                scheduler.step()
                count = 0
                min_change /= 10

        if epoch % SHOW_EVERY == 0:
            plot_loss(losses, save=LOSS_PLOT(train_id), mode="log")

        if epoch % SHOW_EVERY == 0:
            # Round the model's Q coefficients
            model.Q.data = torch.round(model.Q.data)

            solution, ps_var = check_solution(torch.round(model.Q).tolist())
            if solution:
                if isinstance(solution, list):
                    solution = solution[0]
                try:
                    ps_result = [solution[ps_var[i]] for i in range(len(ps_var))][::-1]
                except:
                    continue
                print(f"[{get_time()}][Thread {train_id}] Found a solution!")
                with open(OUTPUT_FILE(train_id), "w") as f:
                    f.write(initial_string)
                    f.write("\n" + "-" * 50 + "\n")
                    f.write(f"Epoch {epoch}: Loss = 0\n")
                    f.write(f"p(x) = {ps_result}.\n")
                    f.write(f"Q(x) = {torch.round(model.Q).tolist()[::-1]}\n")
                return


def check_solution(qs):
    # We already have R
    q = sum([qs[i] * x ** i for i in range(len(qs))])
    # Find p such that p(q) = R
    ps = sp.var(f"p0:{DEG_P + 1}")
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


def find_close_solution(thread_id: int = 0):
    qs, ps = sp.symbols(f"q0:{DEG_Q + 1}"), sp.symbols(f"p0:{DEG_P + 1}")
    # Create the polynomial
    q = sum([qs[i] * x ** i for i in range(len(qs))])
    p = sum([ps[i] * x ** i for i in range(len(ps))])
    # Create the equation
    eq = sp.Eq(p.subs(x, q), R)
    # Solve the equation
    sol = sp.solve(eq, ps + qs, dict=True)
    # Check if the solution is valid
    if sol:
        pickle.dump(sol + [ps, qs], open(join(THREAD_DIR(thread_id), "solution.pkl"), "wb"))


def main():
    # Delete the results from the old run
    if RESET_ENVIRONMENT:
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)

    thread = threading.Thread(target=find_close_solution, args=(0,))
    thread.start()

    # Run the threads for this run
    train(0)

    if thread.is_alive():
        raise Exception("Stop the program after training is done.")
    else:
        print("Found exact solution, check the 'solution.pkl' file.")


if __name__ == "__main__":
    main()
