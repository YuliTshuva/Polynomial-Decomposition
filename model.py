"""
Yuli Tshuva.
Designing the model for polynomial decomposition using PyTorch.
"""

import torch
import torch.nn as nn
import sympy as sp
from functions import *


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


def custom_loss(model):
    # Collect all weights
    reg_loss = 0.0
    for param in model.parameters():
        if param.requires_grad:
            # Penalize distance from nearest integer
            rounded = torch.round(param)
            reg_loss += torch.sum((10 * (param - rounded)) ** 2)

    # Total loss
    return reg_loss


class PolynomialSearch(nn.Module):
    """
    Polynomial decomposition using PyTorch.
    """

    def __init__(self, degree: int, deg_q: int = None):
        """
        Initialize the polynomial decomposition module.

        Args:
            degree (int): Degree of the polynomial.
        """
        super(PolynomialSearch, self).__init__()
        # Set the degree of R
        self.degree = degree
        self.deg_q = deg_q
        self.deg_p = degree // deg_q

        # Define the coefficients for P and Q
        self.P = nn.Parameter(torch.randn(self.deg_p + 1, dtype=torch.float64))
        self.Q = nn.Parameter(torch.randn(self.deg_q + 1, dtype=torch.float64))

        # Define the coefficients
        self.x = sp.symbols("x")
        self.qs = sp.symbols('q0:%d' % (self.deg_q + 1))
        self.ps = sp.symbols('p0:%d' % (self.deg_p + 1))
        self.rs = express_with_coefficients(ps=self.ps, qs=self.qs, var=self.x)

        # Match dict
        self.qs_to_idx = {str(k): v for k, v in zip(self.qs, range(self.deg_q + 1))}
        self.ps_to_idx = {str(k): v for k, v in zip(self.ps, range(self.deg_p + 1))}

    def forward(self) -> torch.Tensor:
        output = torch.zeros(len(self.rs), dtype=torch.float64)
        for i, r in enumerate(self.rs):
            for summed in r.split(" + "):
                temp = 1
                for mul in summed.split("*"):
                    if "^" in mul:
                        mul = mul.split("^")
                        temp *= self.Q[self.qs_to_idx[mul[0]]] ** int(mul[1])
                    else:
                        if mul in self.ps_to_idx:
                            temp *= self.P[self.ps_to_idx[mul]]
                        elif mul in self.qs_to_idx:
                            temp *= self.Q[self.qs_to_idx[mul]]
                        else:
                            temp *= torch.tensor(float(mul), dtype=torch.float64)
                output[i] += temp
        return output
