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
import shutil
import sympy as sp
import pickle
from constants import *
from find_closest_solution import find_closest_solution
from rank_directions import rank_directions

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hyperparameters
EPOCHS = int(2e6)
LR, MIN_LR = 10, 1e-10
EARLY_STOPPING, MIN_CHANGE = int(3e2), 2
LAMBDA1, LAMBDA2, P_REG = 1, 1e6, 0
P_REG, Q_REG, HIGHEST_Q_REG = 0, 1, 1
LAMBDA3 = 1e6
TRAIN_Q, START_TUNING_P = 3, 2000

# Constants
RESET_ENVIRONMENT = False
NUM_THREADS = 1
SHOW_EVERY = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_DIR = join("output_dirs", "train_17")
THREAD_DIR = lambda i: join(WORKING_DIR, f"thread_{i}")
OUTPUT_FILE = lambda i: join(THREAD_DIR(i), f"polynomials.txt")
LOSS_PLOT = lambda i: join(THREAD_DIR(i), f"loss.png")
MODEL_PATH = lambda i: join(THREAD_DIR(i), f"model.pth")
STOP_THREAD_FILE = lambda i: join(THREAD_DIR(i), "stop.txt")
DEG_P, DEG_Q = 5, 3
DEGREE = DEG_P * DEG_Q
WEIGHTS = torch.tensor([1] * DEGREE + [LAMBDA2]).to(DEVICE)
SCALE = 100


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


# Define variable
x = sp.symbols('x')
deg_r = 0
while deg_r < DEGREE:
    # Define the polynomials
    P, Q = generate_polynomial(degree=DEG_P, var=x, scale=SCALE), generate_polynomial(degree=DEG_Q, var=x, scale=SCALE)
    # Calculate R
    R = expand(P.subs(x, Q))
    # Present the polynomials' coefficients as real values and not like fraction
    P = [int(c) for c in sp.Poly(P, x).all_coeffs()]
    Q = [int(c) for c in sp.Poly(Q, x).all_coeffs()]
    # Get r's coefficients
    Rs = torch.tensor(sp.Poly(R, x).all_coeffs()[::-1], dtype=torch.float64, requires_grad=True).to(DEVICE)

    # Find the degree of R
    deg_r = R.as_poly(x).degree()


def train(train_id: int):
    # Create the working directory
    os.makedirs(THREAD_DIR(train_id), exist_ok=True)

    # Create output file in the output directory
    initial_string = "Generated Polynomials:\n"
    R_coefficients = Rs.tolist()[::-1]
    initial_string += f"P(x): {P}\nQ(x): {Q}\nR(x): {R_coefficients}\n"

    # Search for coefficients of P and Q
    c_p, c_q = suggest_coefficients(int(R_coefficients[0]), DEG_P)

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

        # Set coefficients
        if epoch < 2000:
            model.P.data[-1] = c_p
            model.Q.data[-1] = c_q

        # Forward pass
        optimizer.zero_grad()
        output = model()

        # Compute loss
        loss = (loss_fn([output, Rs]) + LAMBDA1 * model.q_l1_p_ln(P_REG, Q_REG) +
                LAMBDA3 * model.q_high_degree_regularization(HIGHEST_Q_REG))
        losses.append(loss.item())

        # Backward pass and optimization
        loss.backward()
        if epoch >= START_TUNING_P and epoch % TRAIN_Q != 0:
            highest_coef = model.Q[-1].item()
            optimizer.step()
            model.Q.data[-1] = highest_coef
        else:
            optimizer.step()

        # Set coefficients
        if epoch < 2000:
            model.P.data[-1] = c_p
            model.Q.data[-1] = c_q

        if loss.item() + min_change < min_loss:
            count = 0
            min_loss = loss.item()
            # Save the model in the output directory
            torch.save(model.state_dict(), MODEL_PATH(train_id))

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
    thread_id = max([int(d.split('_')[-1]) for d in os.listdir(WORKING_DIR) if d.startswith("thread_")], default=-1) + 1

    # Run the threads for this run
    found_optimal_solution = train(thread_id)
    # If an optimal solution was found, we can stop here
    if not found_optimal_solution:
        # Find closed form solution
        find_close_solution(thread_id)
        # Find the closest solution
        find_closest_solution(THREAD_DIR(thread_id))
        # Rank the directions
        rank_directions(THREAD_DIR(thread_id))


if __name__ == "__main__":
    main()
