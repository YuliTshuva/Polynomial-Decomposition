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

    def q_l1_p_ln(self, pn, qn) -> torch.Tensor:
        # Penalize weights' absolute value
        reg = torch.sum(torch.abs(torch.pow(self.P, pn)))/len(self.P)
        reg += torch.sum(torch.abs(torch.pow(self.Q, qn)))/len(self.Q)
        return reg

    def q_high_degree_regularization(self, n) -> torch.Tensor:
        # Penalize high degree coefficients of Q
        return torch.abs(torch.pow(self.Q[-1] - torch.round(self.Q[-1]), n))
