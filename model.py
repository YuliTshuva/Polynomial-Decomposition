"""
Yuli Tshuva.
Designing the model for polynomial decomposition using PyTorch.
"""

import torch
import torch.nn as nn


class PolynomialDecomposition(nn.Module):
    """
    Polynomial decomposition using PyTorch.
    """

    def __init__(self, degree: int):
        """
        Initialize the polynomial decomposition module.

        Args:
            degree (int): Degree of the polynomial.
        """
        super(PolynomialDecomposition, self).__init__()
        # Set the degree of R
        self.degree = degree
        # Define the coefficients for P and Q
        self.P = nn.Parameter(torch.randn(degree // 2 + 1, dtype=torch.float64))
        self.Q = nn.Parameter(torch.randn(degree // 2 + 1, dtype=torch.float64))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Compute Q(x)
        Qx = sum(self.Q[i] * input ** i for i in range(len(self.Q)))  # Q(x)

        # Now compute P(Q(x)) = R(x)
        Rx = sum(self.P[i] * Qx ** i for i in range(len(self.P)))  # P(Q(x)) = R(x)

        return Rx  # Already differentiable!
