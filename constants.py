# Hyperparameters
EPOCHS = int(1e6)
LR, MIN_LR = 1e-1, 1e-10
EARLY_STOPPING, MIN_CHANGE = int(4e2), 1e-1
START_REGULARIZATION = 300
LAMBDA1, LAMBDA2 = 1, 1e6  # Weight of the sparsity, Weight of the largest coeffs
