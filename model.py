"""
Yuli Tshuva.
Designing the model for polynomial decomposition using PyTorch.
"""

import torch
import torch.nn as nn
import sympy as sp
from sympy import expand


def polynomial_decomposition(degree: int) -> callable:
    """
    Polynomial decomposition using SymPy.

    Args:
        degree (int): Degree of the polynomial.

    Returns:
        tuple: Coefficients of the polynomial P and Q.
    """
    # Define the variable
    x = sp.symbols('x')
    a_i, b_i = sp.symbols([f'a{i}' for i in range(degree//2 + 1)]), sp.symbols([f'b{i}' for i in range(degree//2 + 1)])
    # Define the polynomial P and Q
    P = sum(a_i[i] * x ** i for i in range(degree//2 + 1))
    Q = sum(b_i[i] * x ** i for i in range(degree//2 + 1))
    # Compute the polynomial composition
    R = expand(P.subs(x, Q))
    # Extract coefficients
    coeffs = sp.Poly(R, x).all_coeffs()[::-1]

    def compose(P: torch.tensor, Q: torch.tensor) -> torch.tensor:
        """
        Compose two polynomials P and Q.

        Args:
            P (list): Coefficients of polynomial P.
            Q (list): Coefficients of polynomial Q.

        Returns:
            list: Coefficients of the composed polynomial P(Q).
        """
        # Substitute Q's values into b and P's values into a
        R = torch.tensor([coeffs[i].subs([(k, v) for k, v in zip(b_i, Q)] + [(k, v) for k, v in zip(a_i, P)])
                          for i in range(degree + 1)], device=P.device, dtype=torch.float64, requires_grad=True)
        return R

    return compose


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
        self.P = nn.Parameter(torch.randn(degree//2 + 1))
        self.Q = nn.Parameter(torch.randn(degree//2 + 1))
        # Create a function that describes the polynomial composition
        self.compose = polynomial_decomposition(degree)

    def forward(self) -> torch.Tensor:
        """
        Forward pass for polynomial decomposition.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after polynomial decomposition.
        """
        # Compute the polynomial composition
        return self.compose(self.P, self.Q)


# Define a loss function
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, y_pred, y_true):
        return torch.mean(self.weights * (y_pred - y_true) ** 2)
