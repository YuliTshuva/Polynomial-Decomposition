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
import os
import threading
import time
import shutil

# Hyperparameters
EPOCHS = int(1e5)
LR = 1e-1
EARLY_STOPPING, MIN_CHANGE = int(1e3), 1e-1
DEG_P, DEG_Q = 5, 3

# Constants
NUM_THREADS = 4
SHOW_EVERY = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_DIR = join("output_dirs", "train_6")
STOP_FILE = join(WORKING_DIR, "stop.txt")
THREAD_DIR = lambda i: join(WORKING_DIR, f"thread_{i}")
OUTPUT_FILE = lambda i: join(THREAD_DIR(i), f"polynomials.txt")
LOSS_PLOT = lambda i: join(THREAD_DIR(i), f"loss.png")
MODEL_PATH = lambda i: join(THREAD_DIR(i), f"model.pth")
STOP_THREAD_FILE = lambda i: join(THREAD_DIR(i), "stop.txt")
SUCCESS_FILE = join(WORKING_DIR, "success.txt")
DEGREE = DEG_P * DEG_Q

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

# Stop flag
stop_event = threading.Event()


def train(train_id: int):
    # Create the working directory
    os.makedirs(THREAD_DIR(train_id), exist_ok=True)

    # Create output file in the output directory
    initial_string = f"P(x): {present_result(P)}\nQ(x): {present_result(Q)}\nR(x): {present_result(R)}\n"
    with open(OUTPUT_FILE(train_id), "w") as f:
        f.write(initial_string)

    # Initialize the model
    model = PolynomialSearch(degree=DEGREE, deg_q=DEG_Q).to(DEVICE)

    # Define the optimizer
    lr, min_change = LR, MIN_CHANGE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Define loss function
    loss_fn = torch.nn.L1Loss()

    # Set a list of loss functions
    losses = []
    count, min_loss, best_epoch = 0, float("inf"), -1
    # Build a training loop
    for epoch in range(EPOCHS):
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
        if loss.item() + min_change < min_loss:
            best_epoch = epoch
            count = 0
            min_loss = loss.item()
            # Save the model in the output directory
            torch.save(model.state_dict(), MODEL_PATH(train_id))
        else:
            count += 1

        if count > EARLY_STOPPING:
            if lr == 1e-5:
                print(f"[Thread {train_id}]: Early stopping at epoch {epoch}")
                break
            else:
                print(f"[Thread {train_id}]: Reducing learning rate at epoch {epoch}")
                lr = lr / 10
                scheduler.step()
                count = 0
                min_change /= 10

        # Plot the loss
        if epoch % SHOW_EVERY == 0:
            plot_loss(losses, save=LOSS_PLOT(train_id))

            with open(OUTPUT_FILE(train_id), "w") as f:
                f.write(initial_string)
                f.write("\n" + "-" * 50 + "\n")
                f.write(f"Epoch {epoch} ({int(epoch / EPOCHS * 100)}%): Loss = {loss.item():.3f}\n")
                f.write(f"P(x): {torch.round(model.P, decimals=3).tolist()[::-1]}\n")
                f.write(f"Q(x): {torch.round(model.Q, decimals=3).tolist()[::-1]}\n")
                f.write(f"R(x): {torch.round(output, decimals=3).tolist()[::-1]}\n")

        if epoch > 500 and min_loss > 1:
            print(f"[Thread {train_id}]: Stopping thread and Starting a new one.")
            start_new_thread(train_id)
            return

        if os.path.exists(STOP_THREAD_FILE(train_id)):
            print(f"[Thread {train_id}]: Stopping thread and Starting a new one.")
            os.remove(STOP_THREAD_FILE(train_id))
            start_new_thread(train_id)
            return

        if stop_event.is_set():
            print(f"[Thread {train_id}]: Stopping thread.")
            return

        if min_loss < 1e-1:
            print(f"[Thread {train_id}]: Found optimal solution.")
            break

    # Add the minimum loss to the list of losses
    losses.append(min_loss)

    # Plot the losses
    plot_loss(losses, save=LOSS_PLOT(train_id))

    # Load the best model
    model.load_state_dict(torch.load(MODEL_PATH(train_id)))

    # Final evaluation
    model.eval()
    P_weights = model.P.detach().cpu().numpy()
    Q_weights = model.Q.detach().cpu().numpy()
    restored_p = sum(P_weights[i] * x ** i for i in range(len(P_weights)))
    restored_q = sum(Q_weights[i] * x ** i for i in range(len(Q_weights)))
    with open(OUTPUT_FILE(train_id), "a") as f:
        f.write("\n" + "-" * 50 + "\n")
        f.write(f"Epoch {best_epoch} ({(best_epoch / EPOCHS * 100):.3f}%): Loss = {min_loss:.3f}\n")
        f.write(f"Restored P(x): {present_result(restored_p)}\n")
        f.write(f"Restored Q(x): {present_result(restored_q)}\n")
        f.write(f"Restored R(x): {present_result(expand(restored_p.subs(x, restored_q)))}\n")

    # Save the success file
    with open(SUCCESS_FILE, "w") as f:
        f.write(f"Success at thread: {train_id}.")

    # Stop all other events
    stop_event.set()


def start_new_thread(i):
    if not os.path.exists(SUCCESS_FILE):
        thread = threading.Thread(target=train, args=(i,))
        thread.start()


def main():
    for i in range(NUM_THREADS):
        thread = threading.Thread(target=train, args=(i,))
        thread.start()

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


if __name__ == "__main__":
    main()
