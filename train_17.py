"""
Yuli Tshuva
Trying to improve my algorithm for large coefficients by rounding Q's coeffs every XXXX epochs.
"""

# Imports
import sys
import torch
import torch.optim as optim
from model import PolynomialSearch
from functions import *
from os.path import join
import os
import shutil
import sympy as sp
import pickle
from find_closest_solution import find_closest_solution
import importlib
import efficient_model
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hyperparameters
EPOCHS = int(1e4)
LR, MIN_LR = 10, 1e-3
EARLY_STOPPING, MIN_CHANGE = int(3e2), 2
LAMBDA1, LAMBDA2, LAMBDA3 = 1, 1, 1e3
P_REG, Q_REG = 0, 1
FORCE_COEFFICIENTS = 4000

# Adjust for hyperparameters optimization
if len(sys.argv) > 7:
    LR = float(sys.argv[7])
    MIN_LR = float(sys.argv[8])
    EARLY_STOPPING = int(sys.argv[9])
    LAMBDA1 = float(sys.argv[10])
    LAMBDA2 = float(sys.argv[11])
    LAMBDA3 = float(sys.argv[12])
    FORCE_COEFFICIENTS = int(sys.argv[13])

USE_PARTS = {
    "guess coefficients": sys.argv[4] == "1",
    "use regularization": sys.argv[5] == "1",
    "round coefficients": sys.argv[6] == "1"
}

# Constants
RESET_ENVIRONMENT = False
NUM_THREADS = 1
SHOW_EVERY = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if len(sys.argv) <= 7:
    WORKING_DIR = join("output_dirs", f"train_17_{'1' if USE_PARTS['guess coefficients'] else '0'}{'1' if USE_PARTS['use regularization'] else '0'}{'1' if USE_PARTS['round coefficients'] else '0'}")
else:
    WORKING_DIR = sys.argv[14]
THREAD_DIR = lambda i: join(WORKING_DIR, f"thread_{i}")
OUTPUT_FILE = lambda i: join(THREAD_DIR(i), f"polynomials.txt")
LOSS_PLOT = lambda i: join(THREAD_DIR(i), f"loss.png")
MODEL_PATH = lambda i: join(THREAD_DIR(i), f"model.pth")
STOP_THREAD_FILE = lambda i: join(THREAD_DIR(i), "stop.txt")


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


# Define variable
x = sp.symbols('x')

# Load the input polynomials
P, Q = sys.argv[1], sys.argv[2]
# Convert the input polynomials from string to sympy expressions
P = sp.sympify(P)
Q = sp.sympify(Q)

# Calculate the polynomial degrees
DEG_P, DEG_Q = sp.Poly(P, x).degree(), sp.Poly(Q, x).degree()
DEGREE = DEG_P * DEG_Q
WEIGHTS = torch.tensor([1] * DEGREE + [LAMBDA3]).to(DEVICE)

# Calculate R
R = expand(P.subs(x, Q))
# Present the polynomials' coefficients as real values and not like fraction
P = [int(c) for c in sp.Poly(P, x).all_coeffs()]
Q = [int(c) for c in sp.Poly(Q, x).all_coeffs()]
# Get r's coefficients
Rs = torch.tensor(sp.Poly(R, x).all_coeffs()[::-1], dtype=torch.float64, requires_grad=True).to(DEVICE)

if not USE_PARTS["use regularization"]:
    LAMBDA1, LAMBDA2 = 0, 0


def train(train_id: int):
    # Create the working directory
    os.makedirs(THREAD_DIR(train_id), exist_ok=True)

    # Create output file in the output directory
    initial_string = "Generated Polynomials:\n"
    R_coefficients = Rs.tolist()[::-1]
    initial_string += f"P(x): {P}\nQ(x): {Q}\nR(x): {R_coefficients}\n"

    # Search for coefficients of P and Q
    c_p, c_q = suggest_coefficients(int(R_coefficients[0]), DEG_P)

    # Set module and class name
    module_name = "efficient_model"
    class_name = f"EfficientPolynomialSearch_{DEGREE}_{DEG_Q}"

    # Check if such a model exists
    with open(module_name+".py", "r") as file:
        models_available = file.read()

    # Check if the model already been created
    if class_name in models_available:
        # Import the module dynamically
        module = importlib.import_module(module_name)
        # Get the class dynamically
        cls = getattr(module, class_name)
        # Optionally instantiate it
        model = cls().to(DEVICE)
    # Create the model and instantiate it
    else:
        # Initialize the model to get its expression list
        model = PolynomialSearch(degree=DEGREE, deg_q=DEG_Q).to(DEVICE)
        exp_list = model.rs
        # Create the efficient version of the model
        create_efficient_model(exp_list, degree=DEGREE, deg_q=DEG_Q)
        # Import the efficient model created
        importlib.reload(efficient_model)
        # Import the module dynamically
        module = importlib.import_module(module_name)
        # Get the class dynamically
        cls = getattr(module, class_name)
        # Optionally instantiate it
        model = cls().to(DEVICE)

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

        # Set coefficients
        if epoch < FORCE_COEFFICIENTS and c_p and USE_PARTS["guess coefficients"]:
            model.P.data[-1] = c_p
            model.Q.data[-1] = c_q

        # Forward pass
        optimizer.zero_grad()
        output = model()

        # Compute loss
        loss = loss_fn([output, Rs]) + LAMBDA1 * model.q_ln(Q_REG) + LAMBDA2 * model.p_ln(P_REG)
        losses.append(loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Set coefficients
        if epoch < FORCE_COEFFICIENTS and c_p and USE_PARTS["guess coefficients"]:
            model.P.data[-1] = c_p
            model.Q.data[-1] = c_q

        if loss.item() + min_change < min_loss:
            count = 0
            min_loss = loss.item()
            # Save the model in the output directory
            # torch.save(model.state_dict(), MODEL_PATH(train_id))

            with open(OUTPUT_FILE(train_id), "w") as f:
                f.write(initial_string)
                f.write("\n" + "-" * 50 + "\n")
                f.write(f"Epoch {epoch}: Loss = {loss.item()}\n")
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
                # plot_loss(losses, save=LOSS_PLOT(train_id), mode="log", xticks=epochs)
                return
            else:
                lr = lr / 10
                print(f"[{get_time()}][Thread {train_id}]: Reducing learning rate to {lr} at epoch {epoch}")
                epochs.append(epoch)
                scheduler.step()
                count = 0
                min_change /= 10

        # # Plot loss and check solution
        # if epoch % SHOW_EVERY == 0:
        #     plot_loss(losses, save=LOSS_PLOT(train_id), mode="log")

        if epoch % SHOW_EVERY == 0 and USE_PARTS["round coefficients"]:
            solution, ps_var = check_solution(torch.round(model.Q).tolist())
            if solution:
                if isinstance(solution, list):
                    solution = solution[0]
                try:
                    ps_result = [int(solution[ps_var[i]]) for i in range(len(ps_var))][::-1]
                except:
                    continue
                print(f"[{get_time()}][Thread {train_id}] Found a solution!")
                with open(OUTPUT_FILE(train_id), "w") as f:
                    f.write(initial_string)
                    f.write("\n" + "-" * 50 + "\n")
                    f.write(f"Epoch {epoch}: Loss = 0\n")
                    f.write(f"p(x) = {ps_result}.\n")
                    f.write(f"Q(x) = {torch.round(model.Q).tolist()[::-1]}\n")
                return True

        if os.path.exists(STOP_THREAD_FILE(train_id)):
            print(f"[{get_time()}][Thread {train_id}]: Stopping thread.")
            os.remove(STOP_THREAD_FILE(train_id))
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
    os.makedirs(WORKING_DIR, exist_ok=True)
    if RESET_ENVIRONMENT:
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)

    # Find thread id
    thread_id = int(sys.argv[3])

    # Measure runtime
    start_time = time.time()

    # Run the threads for this run
    found_optimal_solution = train(thread_id)

    # Stop runtime measurement
    runtime = time.time() - start_time

    # Save runtime to a file
    with open(OUTPUT_FILE(thread_id), "a") as f:
        f.write(f"\nTotal runtime: {runtime:.2f} seconds\n")

    # # If an optimal solution was found, we can stop here
    # if not found_optimal_solution:
    #     # Find closed form solution
    #     find_close_solution(thread_id)
    #     # Find the closest solution
    #     find_closest_solution(THREAD_DIR(thread_id), DEGREE, DEG_Q)


if __name__ == "__main__":
    main()
