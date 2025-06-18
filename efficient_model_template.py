import torch
from torch import nn


class EfficientPolynomialSearch(nn.Module):
    """
    Polynomial decomposition using PyTorch.

    If the initialization is good, it is perfect. Else it will just not work.
    """

    def __init__(self, degree: int, deg_q: int = None):
        """
        Initialize the polynomial decomposition module.

        Args:
            degree (int): Degree of the polynomial.
        """
        super(EfficientPolynomialSearch, self).__init__()
        # Set the degree of R
        self.degree = degree
        self.deg_q = deg_q
        self.deg_p = degree // deg_q

        # Define the coefficients for P and Q
        self.P = nn.Parameter(torch.randn(self.deg_p + 1, dtype=torch.float64) / 10000)
        self.Q = nn.Parameter(torch.randn(self.deg_q + 1, dtype=torch.float64) / 10000)

    def forward(self) -> torch.Tensor:
        output = torch.zeros(self.degree + 1, dtype=torch.float64)

        # Update the output by the polynomial
        # TO DO

        return output

    def integer_regularization(self) -> torch.Tensor:
        # Penalize distance from nearest integer
        reg = torch.sum(torch.abs(torch.round(self.P) - self.P)) + torch.sum(torch.abs(torch.round(self.Q) - self.Q))
        reg /= len(self.P) + len(self.Q)
        return reg

    def sparse_optimization(self) -> torch.Tensor:
        # Penalize weights' absolute value
        reg = torch.sum(torch.abs(self.P)) + torch.sum(torch.abs(self.Q))
        reg /= len(self.P) + len(self.Q)
        return reg

    def q_l1_p_ln(self, n) -> torch.Tensor:
        # Penalize weights' absolute value
        reg = torch.sum(torch.abs(torch.pow(self.P, n))) + torch.sum(torch.abs(self.Q))
        reg /= len(self.P) + len(self.Q)
        return reg

    def p_integer_regularization(self) -> torch.Tensor:
        # Penalize the coefficients of Q
        reg = torch.sum(torch.abs((torch.round(self.P) - self.P))) / len(self.P)
        return reg

    def q_high_degree_regularization(self) -> torch.Tensor:
        # Penalize high degree coefficients of Q
        return (self.Q[-1] - torch.round(self.Q[-1])) ** 2
