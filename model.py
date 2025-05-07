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

    def __init__(self, degree: int, deg_q: int = None):
        """
        Initialize the polynomial decomposition module.

        Args:
            degree (int): Degree of the polynomial.
        """
        super(PolynomialDecomposition, self).__init__()
        # Set the degree of R
        self.degree = degree

        # Set the degree of Q
        if not deg_q:
            deg_q = degree // 2
            deg_p = degree // 2
        else:
            deg_p = degree // deg_q

        # Define the coefficients for P and Q
        self.P = nn.Parameter(torch.randn(deg_p + 1, dtype=torch.float64))
        self.Q = nn.Parameter(torch.randn(deg_q + 1, dtype=torch.float64))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        Qx = PolynomialDecomposition.horner_eval(self.Q, input)
        Rx = PolynomialDecomposition.horner_eval(self.P, Qx)
        return Rx

    @staticmethod
    def horner_eval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for c in coeffs.flip(0):
            result = result * x + c
        return result


class Polynomial(nn.Module):
    """
    Polynomial decomposition using PyTorch.
    """

    def __init__(self, deg: int):
        """
        Initialize the polynomial decomposition module.

        Args:
            degree (int): Degree of the polynomial.
        """
        super(Polynomial, self).__init__()
        # Set the degree of R
        self.degree = deg

        # Define the coefficients for P and Q
        self.P = nn.Parameter(torch.randn(deg + 1, dtype=torch.float64))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        Px = PolynomialDecomposition.horner_eval(self.P, input)
        return Px


class LnLoss(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true) ** self.n)
