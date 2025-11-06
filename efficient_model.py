import torch
from torch import nn





class EfficientPolynomialSearch_15_3(nn.Module):
    """
    Polynomial decomposition using PyTorch.

    If the initialization is good, it is perfect. Else it will just not work.
    """

    def __init__(self):
        """
        Initialize the polynomial decomposition module.

        Args:
            degree (int): Degree of the polynomial.
        """
        super(EfficientPolynomialSearch_15_3, self).__init__()
        # Set slack values for compilation
        degree, deg_q = 16, 4
        # Set the degree of R
        self.degree = 15
        self.deg_q = 3
        self.deg_p = self.degree // self.deg_q

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

    def q_l1_p_ln(self, pn, qn) -> torch.Tensor:
        # Penalize weights' absolute value
        reg = torch.sum(torch.abs(torch.pow(self.P, pn)))/len(self.P)
        reg += torch.sum(torch.abs(torch.pow(self.Q, qn)))/len(self.Q)
        return reg

    def q_ln(self, qn) -> torch.Tensor:
        # Penalize weights' absolute value
        reg = torch.sum(torch.abs(torch.pow(self.Q, qn)))/len(self.Q)
        return reg

    def p_ln(self, pn) -> torch.Tensor:
        # Penalize weights' absolute value
        reg = torch.sum(torch.abs(torch.pow(self.P, pn)))/len(self.P)
        return reg

    def q_high_degree_regularization(self, n) -> torch.Tensor:
        # Penalize high degree coefficients of Q
        return torch.abs(torch.pow(self.Q[-1] - torch.round(self.Q[-1]), n))
