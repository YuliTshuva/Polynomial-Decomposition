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
import threading
import shutil
import sympy as sp
import pickle
import random
import time

# Hyperparameters
EPOCHS = int(1e6)
LR = 1e-1
EARLY_STOPPING, MIN_CHANGE = int(4e2), 1e-1
START_REGULARIZATION = 300
LAMBDA1, LAMBDA2 = 1, 1e1  # Weight of the sparsity, Weight of the largest coeffs

# Constants
RESET_ENVIRONMENT = True
SHOW_EVERY = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_DIR = join("output_dirs", "train_13")
STOP_FILE = join(WORKING_DIR, "stop.txt")
THREAD_DIR = lambda i: join(WORKING_DIR, f"thread_{i}")
OUTPUT_FILE = lambda i: join(THREAD_DIR(i), f"polynomials.txt")
LOSS_PLOT = lambda i: join(THREAD_DIR(i), f"loss.png")
MODEL_PATH = lambda i: join(THREAD_DIR(i), f"model.pth")
STOP_THREAD_FILE = lambda i: join(THREAD_DIR(i), "stop.txt")
TERMINATE_THREAD_FILE = lambda i: join(THREAD_DIR(i), "terminate.txt")
SUCCESS_FILE = join(WORKING_DIR, "success.txt")
DIVISORS_DATA = join("data", "divisors.pkl")
TIMEOUT = 60 * 10  # 10 minutes

# Read the divisors data
with open(DIVISORS_DATA, "rb") as f:
    DIVISORS = pickle.load(f)

# Sample a degree for R
DEGREE = random.choice(list(DIVISORS.keys()))
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

# Stop flag
stop_event = threading.Event()
# Lock for thread safety
lock = threading.Lock()


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
    with lock:
        # Create the efficient version of the model
        create_efficient_model(exp_list)
        # Import the efficient model created
        from efficient_model import EfficientPolynomialSearch
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
                f.write(f"R(x): {torch.round(output, decimals=3).tolist()[::-1]}\n")
                diffs = torch.round(torch.abs((output - Rs) * WEIGHTS), decimals=3).tolist()[::-1]
                f.write(f"diffs: {diffs}\n")
        else:
            count += 1

        # Early stopping
        if count > EARLY_STOPPING:
            if lr == 1e-5:
                print(f"[{get_time()}][Thread {train_id}]: Early stopping at epoch {epoch}")
                start_new_thread(train_id)
                return
            else:
                print(f"[{get_time()}][Thread {train_id}]: Reducing learning rate at epoch {epoch}")
                lr = lr / 10
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
            print(f"[{get_time()}][Thread {train_id}]: Stopping thread and Starting a new one.")
            os.remove(STOP_THREAD_FILE(train_id))
            start_new_thread(train_id)
            return

        if os.path.exists(TERMINATE_THREAD_FILE(train_id)):
            print(f"[{get_time()}][Thread {train_id}]: Terminating thread.")
            os.remove(TERMINATE_THREAD_FILE(train_id))
            return

        if stop_event.is_set():
            print(f"[{get_time()}][Thread {train_id}]: Stopping thread.")
            return

        if min_loss < 1e-1:
            print(f"[{get_time()}][Thread {train_id}]: Found optimal solution.")
            break

        if epoch >= 800 and epoch % 200 == 0:
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
                # Stop all other events
                stop_event.set()
                return

    # Add the minimum loss to the list of losses
    losses.append(min_loss)

    # Plot the losses
    plot_loss(losses, save=LOSS_PLOT(train_id))

    # Save the success file
    with open(SUCCESS_FILE, "w") as f:
        f.write(f"Success at thread: {train_id}.")

    # Stop all other events
    stop_event.set()


def start_new_thread(i):
    if not os.path.exists(SUCCESS_FILE):
        thread = threading.Thread(target=train, args=(i,))
        thread.start()


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
    for deg_q in divisors:
        if deg_q > DEGREE // 2 or deg_q >= 10:
            break
        thread = threading.Thread(target=train, args=(deg_q,))
        thread.start()

    # Set a timer for timeout
    start = time.time()
    while True:
        if os.path.exists(STOP_FILE):
            stop_event.set()
            print("Stopping all threads.")
            break

        threads = threading.enumerate()
        running_threads = [t for t in threads if t is not threading.main_thread()]
        if not running_threads:
            print(f"There are no running threads.")
            break

        if time.time() - start > TIMEOUT:
            print(f"Timeout reached. Stopping all threads.")
            raise TimeoutError("Training took too long. Stopping all threads.")


if __name__ == "__main__":
    main()
