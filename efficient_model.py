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
        output[0] = self.P[0] + self.P[1]*self.Q[0] + self.P[2]*self.Q[0]**2 + self.P[3]*self.Q[0]**3 + self.P[4]*self.Q[0]**4 + self.P[5]*self.Q[0]**5
        output[1] = self.P[1]*self.Q[1] + 2*self.P[2]*self.Q[0]*self.Q[1] + 3*self.P[3]*self.Q[0]**2*self.Q[1] + 4*self.P[4]*self.Q[0]**3*self.Q[1] + 5*self.P[5]*self.Q[0]**4*self.Q[1]
        output[2] = self.P[1]*self.Q[2] + 2*self.P[2]*self.Q[0]*self.Q[2] + self.P[2]*self.Q[1]**2 + 3*self.P[3]*self.Q[0]**2*self.Q[2] + 3*self.P[3]*self.Q[0]*self.Q[1]**2 + 4*self.P[4]*self.Q[0]**3*self.Q[2] + 6*self.P[4]*self.Q[0]**2*self.Q[1]**2 + 5*self.P[5]*self.Q[0]**4*self.Q[2] + 10*self.P[5]*self.Q[0]**3*self.Q[1]**2
        output[3] = self.P[1]*self.Q[3] + 2*self.P[2]*self.Q[0]*self.Q[3] + 2*self.P[2]*self.Q[1]*self.Q[2] + 3*self.P[3]*self.Q[0]**2*self.Q[3] + 6*self.P[3]*self.Q[0]*self.Q[1]*self.Q[2] + self.P[3]*self.Q[1]**3 + 4*self.P[4]*self.Q[0]**3*self.Q[3] + 12*self.P[4]*self.Q[0]**2*self.Q[1]*self.Q[2] + 4*self.P[4]*self.Q[0]*self.Q[1]**3 + 5*self.P[5]*self.Q[0]**4*self.Q[3] + 20*self.P[5]*self.Q[0]**3*self.Q[1]*self.Q[2] + 10*self.P[5]*self.Q[0]**2*self.Q[1]**3
        output[4] = 2*self.P[2]*self.Q[1]*self.Q[3] + self.P[2]*self.Q[2]**2 + 6*self.P[3]*self.Q[0]*self.Q[1]*self.Q[3] + 3*self.P[3]*self.Q[0]*self.Q[2]**2 + 3*self.P[3]*self.Q[1]**2*self.Q[2] + 12*self.P[4]*self.Q[0]**2*self.Q[1]*self.Q[3] + 6*self.P[4]*self.Q[0]**2*self.Q[2]**2 + 12*self.P[4]*self.Q[0]*self.Q[1]**2*self.Q[2] + self.P[4]*self.Q[1]**4 + 20*self.P[5]*self.Q[0]**3*self.Q[1]*self.Q[3] + 10*self.P[5]*self.Q[0]**3*self.Q[2]**2 + 30*self.P[5]*self.Q[0]**2*self.Q[1]**2*self.Q[2] + 5*self.P[5]*self.Q[0]*self.Q[1]**4
        output[5] = 2*self.P[2]*self.Q[2]*self.Q[3] + 6*self.P[3]*self.Q[0]*self.Q[2]*self.Q[3] + 3*self.P[3]*self.Q[1]**2*self.Q[3] + 3*self.P[3]*self.Q[1]*self.Q[2]**2 + 12*self.P[4]*self.Q[0]**2*self.Q[2]*self.Q[3] + 12*self.P[4]*self.Q[0]*self.Q[1]**2*self.Q[3] + 12*self.P[4]*self.Q[0]*self.Q[1]*self.Q[2]**2 + 4*self.P[4]*self.Q[1]**3*self.Q[2] + 20*self.P[5]*self.Q[0]**3*self.Q[2]*self.Q[3] + 30*self.P[5]*self.Q[0]**2*self.Q[1]**2*self.Q[3] + 30*self.P[5]*self.Q[0]**2*self.Q[1]*self.Q[2]**2 + 20*self.P[5]*self.Q[0]*self.Q[1]**3*self.Q[2] + self.P[5]*self.Q[1]**5
        output[6] = self.P[2]*self.Q[3]**2 + 3*self.P[3]*self.Q[0]*self.Q[3]**2 + 6*self.P[3]*self.Q[1]*self.Q[2]*self.Q[3] + self.P[3]*self.Q[2]**3 + 6*self.P[4]*self.Q[0]**2*self.Q[3]**2 + 24*self.P[4]*self.Q[0]*self.Q[1]*self.Q[2]*self.Q[3] + 4*self.P[4]*self.Q[0]*self.Q[2]**3 + 4*self.P[4]*self.Q[1]**3*self.Q[3] + 6*self.P[4]*self.Q[1]**2*self.Q[2]**2 + 10*self.P[5]*self.Q[0]**3*self.Q[3]**2 + 60*self.P[5]*self.Q[0]**2*self.Q[1]*self.Q[2]*self.Q[3] + 10*self.P[5]*self.Q[0]**2*self.Q[2]**3 + 20*self.P[5]*self.Q[0]*self.Q[1]**3*self.Q[3] + 30*self.P[5]*self.Q[0]*self.Q[1]**2*self.Q[2]**2 + 5*self.P[5]*self.Q[1]**4*self.Q[2]
        output[7] = 3*self.P[3]*self.Q[1]*self.Q[3]**2 + 3*self.P[3]*self.Q[2]**2*self.Q[3] + 12*self.P[4]*self.Q[0]*self.Q[1]*self.Q[3]**2 + 12*self.P[4]*self.Q[0]*self.Q[2]**2*self.Q[3] + 12*self.P[4]*self.Q[1]**2*self.Q[2]*self.Q[3] + 4*self.P[4]*self.Q[1]*self.Q[2]**3 + 30*self.P[5]*self.Q[0]**2*self.Q[1]*self.Q[3]**2 + 30*self.P[5]*self.Q[0]**2*self.Q[2]**2*self.Q[3] + 60*self.P[5]*self.Q[0]*self.Q[1]**2*self.Q[2]*self.Q[3] + 20*self.P[5]*self.Q[0]*self.Q[1]*self.Q[2]**3 + 5*self.P[5]*self.Q[1]**4*self.Q[3] + 10*self.P[5]*self.Q[1]**3*self.Q[2]**2
        output[8] = 3*self.P[3]*self.Q[2]*self.Q[3]**2 + 12*self.P[4]*self.Q[0]*self.Q[2]*self.Q[3]**2 + 6*self.P[4]*self.Q[1]**2*self.Q[3]**2 + 12*self.P[4]*self.Q[1]*self.Q[2]**2*self.Q[3] + self.P[4]*self.Q[2]**4 + 30*self.P[5]*self.Q[0]**2*self.Q[2]*self.Q[3]**2 + 30*self.P[5]*self.Q[0]*self.Q[1]**2*self.Q[3]**2 + 60*self.P[5]*self.Q[0]*self.Q[1]*self.Q[2]**2*self.Q[3] + 5*self.P[5]*self.Q[0]*self.Q[2]**4 + 20*self.P[5]*self.Q[1]**3*self.Q[2]*self.Q[3] + 10*self.P[5]*self.Q[1]**2*self.Q[2]**3
        output[9] = self.P[3]*self.Q[3]**3 + 4*self.P[4]*self.Q[0]*self.Q[3]**3 + 12*self.P[4]*self.Q[1]*self.Q[2]*self.Q[3]**2 + 4*self.P[4]*self.Q[2]**3*self.Q[3] + 10*self.P[5]*self.Q[0]**2*self.Q[3]**3 + 60*self.P[5]*self.Q[0]*self.Q[1]*self.Q[2]*self.Q[3]**2 + 20*self.P[5]*self.Q[0]*self.Q[2]**3*self.Q[3] + 10*self.P[5]*self.Q[1]**3*self.Q[3]**2 + 30*self.P[5]*self.Q[1]**2*self.Q[2]**2*self.Q[3] + 5*self.P[5]*self.Q[1]*self.Q[2]**4
        output[10] = 4*self.P[4]*self.Q[1]*self.Q[3]**3 + 6*self.P[4]*self.Q[2]**2*self.Q[3]**2 + 20*self.P[5]*self.Q[0]*self.Q[1]*self.Q[3]**3 + 30*self.P[5]*self.Q[0]*self.Q[2]**2*self.Q[3]**2 + 30*self.P[5]*self.Q[1]**2*self.Q[2]*self.Q[3]**2 + 20*self.P[5]*self.Q[1]*self.Q[2]**3*self.Q[3] + self.P[5]*self.Q[2]**5
        output[11] = 4*self.P[4]*self.Q[2]*self.Q[3]**3 + 20*self.P[5]*self.Q[0]*self.Q[2]*self.Q[3]**3 + 10*self.P[5]*self.Q[1]**2*self.Q[3]**3 + 30*self.P[5]*self.Q[1]*self.Q[2]**2*self.Q[3]**2 + 5*self.P[5]*self.Q[2]**4*self.Q[3]
        output[12] = self.P[4]*self.Q[3]**4 + 5*self.P[5]*self.Q[0]*self.Q[3]**4 + 20*self.P[5]*self.Q[1]*self.Q[2]*self.Q[3]**3 + 10*self.P[5]*self.Q[2]**3*self.Q[3]**2
        output[13] = 5*self.P[5]*self.Q[1]*self.Q[3]**4 + 10*self.P[5]*self.Q[2]**2*self.Q[3]**3
        output[14] = 5*self.P[5]*self.Q[2]*self.Q[3]**4
        output[15] = self.P[5]*self.Q[3]**5


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

    def q_integer_regularization(self) -> torch.Tensor:
        # Penalize the coefficients of Q
        reg = torch.sum(torch.abs(torch.round(self.Q) - self.Q)) / len(self.Q)
        return reg

    def q_high_degree_regularization(self) -> torch.Tensor:
        # Penalize high degree coefficients of Q
        return (self.Q[-1] - torch.round(self.Q[-1])) ** 2
