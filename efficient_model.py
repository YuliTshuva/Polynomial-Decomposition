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



class EfficientPolynomialSearch_18_3(nn.Module):
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
        super(EfficientPolynomialSearch_18_3, self).__init__()
        # Set slack values for compilation
        degree, deg_q = 16, 4
        # Set the degree of R
        self.degree = 18
        self.deg_q = 3
        self.deg_p = self.degree // self.deg_q

        # Define the coefficients for P and Q
        self.P = nn.Parameter(torch.randn(self.deg_p + 1, dtype=torch.float64) / 10000)
        self.Q = nn.Parameter(torch.randn(self.deg_q + 1, dtype=torch.float64) / 10000)

    def forward(self) -> torch.Tensor:
        output = torch.zeros(self.degree + 1, dtype=torch.float64)

        # Update the output by the polynomial
        output[0] = self.P[0] + self.P[1]*self.Q[0] + self.P[2]*self.Q[0]**2 + self.P[3]*self.Q[0]**3 + self.P[4]*self.Q[0]**4 + self.P[5]*self.Q[0]**5 + self.P[6]*self.Q[0]**6
        output[1] = self.P[1]*self.Q[1] + 2*self.P[2]*self.Q[0]*self.Q[1] + 3*self.P[3]*self.Q[0]**2*self.Q[1] + 4*self.P[4]*self.Q[0]**3*self.Q[1] + 5*self.P[5]*self.Q[0]**4*self.Q[1] + 6*self.P[6]*self.Q[0]**5*self.Q[1]
        output[2] = self.P[1]*self.Q[2] + 2*self.P[2]*self.Q[0]*self.Q[2] + self.P[2]*self.Q[1]**2 + 3*self.P[3]*self.Q[0]**2*self.Q[2] + 3*self.P[3]*self.Q[0]*self.Q[1]**2 + 4*self.P[4]*self.Q[0]**3*self.Q[2] + 6*self.P[4]*self.Q[0]**2*self.Q[1]**2 + 5*self.P[5]*self.Q[0]**4*self.Q[2] + 10*self.P[5]*self.Q[0]**3*self.Q[1]**2 + 6*self.P[6]*self.Q[0]**5*self.Q[2] + 15*self.P[6]*self.Q[0]**4*self.Q[1]**2
        output[3] = self.P[1]*self.Q[3] + 2*self.P[2]*self.Q[0]*self.Q[3] + 2*self.P[2]*self.Q[1]*self.Q[2] + 3*self.P[3]*self.Q[0]**2*self.Q[3] + 6*self.P[3]*self.Q[0]*self.Q[1]*self.Q[2] + self.P[3]*self.Q[1]**3 + 4*self.P[4]*self.Q[0]**3*self.Q[3] + 12*self.P[4]*self.Q[0]**2*self.Q[1]*self.Q[2] + 4*self.P[4]*self.Q[0]*self.Q[1]**3 + 5*self.P[5]*self.Q[0]**4*self.Q[3] + 20*self.P[5]*self.Q[0]**3*self.Q[1]*self.Q[2] + 10*self.P[5]*self.Q[0]**2*self.Q[1]**3 + 6*self.P[6]*self.Q[0]**5*self.Q[3] + 30*self.P[6]*self.Q[0]**4*self.Q[1]*self.Q[2] + 20*self.P[6]*self.Q[0]**3*self.Q[1]**3
        output[4] = 2*self.P[2]*self.Q[1]*self.Q[3] + self.P[2]*self.Q[2]**2 + 6*self.P[3]*self.Q[0]*self.Q[1]*self.Q[3] + 3*self.P[3]*self.Q[0]*self.Q[2]**2 + 3*self.P[3]*self.Q[1]**2*self.Q[2] + 12*self.P[4]*self.Q[0]**2*self.Q[1]*self.Q[3] + 6*self.P[4]*self.Q[0]**2*self.Q[2]**2 + 12*self.P[4]*self.Q[0]*self.Q[1]**2*self.Q[2] + self.P[4]*self.Q[1]**4 + 20*self.P[5]*self.Q[0]**3*self.Q[1]*self.Q[3] + 10*self.P[5]*self.Q[0]**3*self.Q[2]**2 + 30*self.P[5]*self.Q[0]**2*self.Q[1]**2*self.Q[2] + 5*self.P[5]*self.Q[0]*self.Q[1]**4 + 30*self.P[6]*self.Q[0]**4*self.Q[1]*self.Q[3] + 15*self.P[6]*self.Q[0]**4*self.Q[2]**2 + 60*self.P[6]*self.Q[0]**3*self.Q[1]**2*self.Q[2] + 15*self.P[6]*self.Q[0]**2*self.Q[1]**4
        output[5] = 2*self.P[2]*self.Q[2]*self.Q[3] + 6*self.P[3]*self.Q[0]*self.Q[2]*self.Q[3] + 3*self.P[3]*self.Q[1]**2*self.Q[3] + 3*self.P[3]*self.Q[1]*self.Q[2]**2 + 12*self.P[4]*self.Q[0]**2*self.Q[2]*self.Q[3] + 12*self.P[4]*self.Q[0]*self.Q[1]**2*self.Q[3] + 12*self.P[4]*self.Q[0]*self.Q[1]*self.Q[2]**2 + 4*self.P[4]*self.Q[1]**3*self.Q[2] + 20*self.P[5]*self.Q[0]**3*self.Q[2]*self.Q[3] + 30*self.P[5]*self.Q[0]**2*self.Q[1]**2*self.Q[3] + 30*self.P[5]*self.Q[0]**2*self.Q[1]*self.Q[2]**2 + 20*self.P[5]*self.Q[0]*self.Q[1]**3*self.Q[2] + self.P[5]*self.Q[1]**5 + 30*self.P[6]*self.Q[0]**4*self.Q[2]*self.Q[3] + 60*self.P[6]*self.Q[0]**3*self.Q[1]**2*self.Q[3] + 60*self.P[6]*self.Q[0]**3*self.Q[1]*self.Q[2]**2 + 60*self.P[6]*self.Q[0]**2*self.Q[1]**3*self.Q[2] + 6*self.P[6]*self.Q[0]*self.Q[1]**5
        output[6] = self.P[2]*self.Q[3]**2 + 3*self.P[3]*self.Q[0]*self.Q[3]**2 + 6*self.P[3]*self.Q[1]*self.Q[2]*self.Q[3] + self.P[3]*self.Q[2]**3 + 6*self.P[4]*self.Q[0]**2*self.Q[3]**2 + 24*self.P[4]*self.Q[0]*self.Q[1]*self.Q[2]*self.Q[3] + 4*self.P[4]*self.Q[0]*self.Q[2]**3 + 4*self.P[4]*self.Q[1]**3*self.Q[3] + 6*self.P[4]*self.Q[1]**2*self.Q[2]**2 + 10*self.P[5]*self.Q[0]**3*self.Q[3]**2 + 60*self.P[5]*self.Q[0]**2*self.Q[1]*self.Q[2]*self.Q[3] + 10*self.P[5]*self.Q[0]**2*self.Q[2]**3 + 20*self.P[5]*self.Q[0]*self.Q[1]**3*self.Q[3] + 30*self.P[5]*self.Q[0]*self.Q[1]**2*self.Q[2]**2 + 5*self.P[5]*self.Q[1]**4*self.Q[2] + 15*self.P[6]*self.Q[0]**4*self.Q[3]**2 + 120*self.P[6]*self.Q[0]**3*self.Q[1]*self.Q[2]*self.Q[3] + 20*self.P[6]*self.Q[0]**3*self.Q[2]**3 + 60*self.P[6]*self.Q[0]**2*self.Q[1]**3*self.Q[3] + 90*self.P[6]*self.Q[0]**2*self.Q[1]**2*self.Q[2]**2 + 30*self.P[6]*self.Q[0]*self.Q[1]**4*self.Q[2] + self.P[6]*self.Q[1]**6
        output[7] = 3*self.P[3]*self.Q[1]*self.Q[3]**2 + 3*self.P[3]*self.Q[2]**2*self.Q[3] + 12*self.P[4]*self.Q[0]*self.Q[1]*self.Q[3]**2 + 12*self.P[4]*self.Q[0]*self.Q[2]**2*self.Q[3] + 12*self.P[4]*self.Q[1]**2*self.Q[2]*self.Q[3] + 4*self.P[4]*self.Q[1]*self.Q[2]**3 + 30*self.P[5]*self.Q[0]**2*self.Q[1]*self.Q[3]**2 + 30*self.P[5]*self.Q[0]**2*self.Q[2]**2*self.Q[3] + 60*self.P[5]*self.Q[0]*self.Q[1]**2*self.Q[2]*self.Q[3] + 20*self.P[5]*self.Q[0]*self.Q[1]*self.Q[2]**3 + 5*self.P[5]*self.Q[1]**4*self.Q[3] + 10*self.P[5]*self.Q[1]**3*self.Q[2]**2 + 60*self.P[6]*self.Q[0]**3*self.Q[1]*self.Q[3]**2 + 60*self.P[6]*self.Q[0]**3*self.Q[2]**2*self.Q[3] + 180*self.P[6]*self.Q[0]**2*self.Q[1]**2*self.Q[2]*self.Q[3] + 60*self.P[6]*self.Q[0]**2*self.Q[1]*self.Q[2]**3 + 30*self.P[6]*self.Q[0]*self.Q[1]**4*self.Q[3] + 60*self.P[6]*self.Q[0]*self.Q[1]**3*self.Q[2]**2 + 6*self.P[6]*self.Q[1]**5*self.Q[2]
        output[8] = 3*self.P[3]*self.Q[2]*self.Q[3]**2 + 12*self.P[4]*self.Q[0]*self.Q[2]*self.Q[3]**2 + 6*self.P[4]*self.Q[1]**2*self.Q[3]**2 + 12*self.P[4]*self.Q[1]*self.Q[2]**2*self.Q[3] + self.P[4]*self.Q[2]**4 + 30*self.P[5]*self.Q[0]**2*self.Q[2]*self.Q[3]**2 + 30*self.P[5]*self.Q[0]*self.Q[1]**2*self.Q[3]**2 + 60*self.P[5]*self.Q[0]*self.Q[1]*self.Q[2]**2*self.Q[3] + 5*self.P[5]*self.Q[0]*self.Q[2]**4 + 20*self.P[5]*self.Q[1]**3*self.Q[2]*self.Q[3] + 10*self.P[5]*self.Q[1]**2*self.Q[2]**3 + 60*self.P[6]*self.Q[0]**3*self.Q[2]*self.Q[3]**2 + 90*self.P[6]*self.Q[0]**2*self.Q[1]**2*self.Q[3]**2 + 180*self.P[6]*self.Q[0]**2*self.Q[1]*self.Q[2]**2*self.Q[3] + 15*self.P[6]*self.Q[0]**2*self.Q[2]**4 + 120*self.P[6]*self.Q[0]*self.Q[1]**3*self.Q[2]*self.Q[3] + 60*self.P[6]*self.Q[0]*self.Q[1]**2*self.Q[2]**3 + 6*self.P[6]*self.Q[1]**5*self.Q[3] + 15*self.P[6]*self.Q[1]**4*self.Q[2]**2
        output[9] = self.P[3]*self.Q[3]**3 + 4*self.P[4]*self.Q[0]*self.Q[3]**3 + 12*self.P[4]*self.Q[1]*self.Q[2]*self.Q[3]**2 + 4*self.P[4]*self.Q[2]**3*self.Q[3] + 10*self.P[5]*self.Q[0]**2*self.Q[3]**3 + 60*self.P[5]*self.Q[0]*self.Q[1]*self.Q[2]*self.Q[3]**2 + 20*self.P[5]*self.Q[0]*self.Q[2]**3*self.Q[3] + 10*self.P[5]*self.Q[1]**3*self.Q[3]**2 + 30*self.P[5]*self.Q[1]**2*self.Q[2]**2*self.Q[3] + 5*self.P[5]*self.Q[1]*self.Q[2]**4 + 20*self.P[6]*self.Q[0]**3*self.Q[3]**3 + 180*self.P[6]*self.Q[0]**2*self.Q[1]*self.Q[2]*self.Q[3]**2 + 60*self.P[6]*self.Q[0]**2*self.Q[2]**3*self.Q[3] + 60*self.P[6]*self.Q[0]*self.Q[1]**3*self.Q[3]**2 + 180*self.P[6]*self.Q[0]*self.Q[1]**2*self.Q[2]**2*self.Q[3] + 30*self.P[6]*self.Q[0]*self.Q[1]*self.Q[2]**4 + 30*self.P[6]*self.Q[1]**4*self.Q[2]*self.Q[3] + 20*self.P[6]*self.Q[1]**3*self.Q[2]**3
        output[10] = 4*self.P[4]*self.Q[1]*self.Q[3]**3 + 6*self.P[4]*self.Q[2]**2*self.Q[3]**2 + 20*self.P[5]*self.Q[0]*self.Q[1]*self.Q[3]**3 + 30*self.P[5]*self.Q[0]*self.Q[2]**2*self.Q[3]**2 + 30*self.P[5]*self.Q[1]**2*self.Q[2]*self.Q[3]**2 + 20*self.P[5]*self.Q[1]*self.Q[2]**3*self.Q[3] + self.P[5]*self.Q[2]**5 + 60*self.P[6]*self.Q[0]**2*self.Q[1]*self.Q[3]**3 + 90*self.P[6]*self.Q[0]**2*self.Q[2]**2*self.Q[3]**2 + 180*self.P[6]*self.Q[0]*self.Q[1]**2*self.Q[2]*self.Q[3]**2 + 120*self.P[6]*self.Q[0]*self.Q[1]*self.Q[2]**3*self.Q[3] + 6*self.P[6]*self.Q[0]*self.Q[2]**5 + 15*self.P[6]*self.Q[1]**4*self.Q[3]**2 + 60*self.P[6]*self.Q[1]**3*self.Q[2]**2*self.Q[3] + 15*self.P[6]*self.Q[1]**2*self.Q[2]**4
        output[11] = 4*self.P[4]*self.Q[2]*self.Q[3]**3 + 20*self.P[5]*self.Q[0]*self.Q[2]*self.Q[3]**3 + 10*self.P[5]*self.Q[1]**2*self.Q[3]**3 + 30*self.P[5]*self.Q[1]*self.Q[2]**2*self.Q[3]**2 + 5*self.P[5]*self.Q[2]**4*self.Q[3] + 60*self.P[6]*self.Q[0]**2*self.Q[2]*self.Q[3]**3 + 60*self.P[6]*self.Q[0]*self.Q[1]**2*self.Q[3]**3 + 180*self.P[6]*self.Q[0]*self.Q[1]*self.Q[2]**2*self.Q[3]**2 + 30*self.P[6]*self.Q[0]*self.Q[2]**4*self.Q[3] + 60*self.P[6]*self.Q[1]**3*self.Q[2]*self.Q[3]**2 + 60*self.P[6]*self.Q[1]**2*self.Q[2]**3*self.Q[3] + 6*self.P[6]*self.Q[1]*self.Q[2]**5
        output[12] = self.P[4]*self.Q[3]**4 + 5*self.P[5]*self.Q[0]*self.Q[3]**4 + 20*self.P[5]*self.Q[1]*self.Q[2]*self.Q[3]**3 + 10*self.P[5]*self.Q[2]**3*self.Q[3]**2 + 15*self.P[6]*self.Q[0]**2*self.Q[3]**4 + 120*self.P[6]*self.Q[0]*self.Q[1]*self.Q[2]*self.Q[3]**3 + 60*self.P[6]*self.Q[0]*self.Q[2]**3*self.Q[3]**2 + 20*self.P[6]*self.Q[1]**3*self.Q[3]**3 + 90*self.P[6]*self.Q[1]**2*self.Q[2]**2*self.Q[3]**2 + 30*self.P[6]*self.Q[1]*self.Q[2]**4*self.Q[3] + self.P[6]*self.Q[2]**6
        output[13] = 5*self.P[5]*self.Q[1]*self.Q[3]**4 + 10*self.P[5]*self.Q[2]**2*self.Q[3]**3 + 30*self.P[6]*self.Q[0]*self.Q[1]*self.Q[3]**4 + 60*self.P[6]*self.Q[0]*self.Q[2]**2*self.Q[3]**3 + 60*self.P[6]*self.Q[1]**2*self.Q[2]*self.Q[3]**3 + 60*self.P[6]*self.Q[1]*self.Q[2]**3*self.Q[3]**2 + 6*self.P[6]*self.Q[2]**5*self.Q[3]
        output[14] = 5*self.P[5]*self.Q[2]*self.Q[3]**4 + 30*self.P[6]*self.Q[0]*self.Q[2]*self.Q[3]**4 + 15*self.P[6]*self.Q[1]**2*self.Q[3]**4 + 60*self.P[6]*self.Q[1]*self.Q[2]**2*self.Q[3]**3 + 15*self.P[6]*self.Q[2]**4*self.Q[3]**2
        output[15] = self.P[5]*self.Q[3]**5 + 6*self.P[6]*self.Q[0]*self.Q[3]**5 + 30*self.P[6]*self.Q[1]*self.Q[2]*self.Q[3]**4 + 20*self.P[6]*self.Q[2]**3*self.Q[3]**3
        output[16] = 6*self.P[6]*self.Q[1]*self.Q[3]**5 + 15*self.P[6]*self.Q[2]**2*self.Q[3]**4
        output[17] = 6*self.P[6]*self.Q[2]*self.Q[3]**5
        output[18] = self.P[6]*self.Q[3]**6


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



class EfficientPolynomialSearch_12_3(nn.Module):
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
        super(EfficientPolynomialSearch_12_3, self).__init__()
        # Set slack values for compilation
        degree, deg_q = 16, 4
        # Set the degree of R
        self.degree = 12
        self.deg_q = 3
        self.deg_p = self.degree // self.deg_q

        # Define the coefficients for P and Q
        self.P = nn.Parameter(torch.randn(self.deg_p + 1, dtype=torch.float64) / 10000)
        self.Q = nn.Parameter(torch.randn(self.deg_q + 1, dtype=torch.float64) / 10000)

    def forward(self) -> torch.Tensor:
        output = torch.zeros(self.degree + 1, dtype=torch.float64)

        # Update the output by the polynomial
        output[0] = self.P[0] + self.P[1]*self.Q[0] + self.P[2]*self.Q[0]**2 + self.P[3]*self.Q[0]**3 + self.P[4]*self.Q[0]**4
        output[1] = self.P[1]*self.Q[1] + 2*self.P[2]*self.Q[0]*self.Q[1] + 3*self.P[3]*self.Q[0]**2*self.Q[1] + 4*self.P[4]*self.Q[0]**3*self.Q[1]
        output[2] = self.P[1]*self.Q[2] + 2*self.P[2]*self.Q[0]*self.Q[2] + self.P[2]*self.Q[1]**2 + 3*self.P[3]*self.Q[0]**2*self.Q[2] + 3*self.P[3]*self.Q[0]*self.Q[1]**2 + 4*self.P[4]*self.Q[0]**3*self.Q[2] + 6*self.P[4]*self.Q[0]**2*self.Q[1]**2
        output[3] = self.P[1]*self.Q[3] + 2*self.P[2]*self.Q[0]*self.Q[3] + 2*self.P[2]*self.Q[1]*self.Q[2] + 3*self.P[3]*self.Q[0]**2*self.Q[3] + 6*self.P[3]*self.Q[0]*self.Q[1]*self.Q[2] + self.P[3]*self.Q[1]**3 + 4*self.P[4]*self.Q[0]**3*self.Q[3] + 12*self.P[4]*self.Q[0]**2*self.Q[1]*self.Q[2] + 4*self.P[4]*self.Q[0]*self.Q[1]**3
        output[4] = 2*self.P[2]*self.Q[1]*self.Q[3] + self.P[2]*self.Q[2]**2 + 6*self.P[3]*self.Q[0]*self.Q[1]*self.Q[3] + 3*self.P[3]*self.Q[0]*self.Q[2]**2 + 3*self.P[3]*self.Q[1]**2*self.Q[2] + 12*self.P[4]*self.Q[0]**2*self.Q[1]*self.Q[3] + 6*self.P[4]*self.Q[0]**2*self.Q[2]**2 + 12*self.P[4]*self.Q[0]*self.Q[1]**2*self.Q[2] + self.P[4]*self.Q[1]**4
        output[5] = 2*self.P[2]*self.Q[2]*self.Q[3] + 6*self.P[3]*self.Q[0]*self.Q[2]*self.Q[3] + 3*self.P[3]*self.Q[1]**2*self.Q[3] + 3*self.P[3]*self.Q[1]*self.Q[2]**2 + 12*self.P[4]*self.Q[0]**2*self.Q[2]*self.Q[3] + 12*self.P[4]*self.Q[0]*self.Q[1]**2*self.Q[3] + 12*self.P[4]*self.Q[0]*self.Q[1]*self.Q[2]**2 + 4*self.P[4]*self.Q[1]**3*self.Q[2]
        output[6] = self.P[2]*self.Q[3]**2 + 3*self.P[3]*self.Q[0]*self.Q[3]**2 + 6*self.P[3]*self.Q[1]*self.Q[2]*self.Q[3] + self.P[3]*self.Q[2]**3 + 6*self.P[4]*self.Q[0]**2*self.Q[3]**2 + 24*self.P[4]*self.Q[0]*self.Q[1]*self.Q[2]*self.Q[3] + 4*self.P[4]*self.Q[0]*self.Q[2]**3 + 4*self.P[4]*self.Q[1]**3*self.Q[3] + 6*self.P[4]*self.Q[1]**2*self.Q[2]**2
        output[7] = 3*self.P[3]*self.Q[1]*self.Q[3]**2 + 3*self.P[3]*self.Q[2]**2*self.Q[3] + 12*self.P[4]*self.Q[0]*self.Q[1]*self.Q[3]**2 + 12*self.P[4]*self.Q[0]*self.Q[2]**2*self.Q[3] + 12*self.P[4]*self.Q[1]**2*self.Q[2]*self.Q[3] + 4*self.P[4]*self.Q[1]*self.Q[2]**3
        output[8] = 3*self.P[3]*self.Q[2]*self.Q[3]**2 + 12*self.P[4]*self.Q[0]*self.Q[2]*self.Q[3]**2 + 6*self.P[4]*self.Q[1]**2*self.Q[3]**2 + 12*self.P[4]*self.Q[1]*self.Q[2]**2*self.Q[3] + self.P[4]*self.Q[2]**4
        output[9] = self.P[3]*self.Q[3]**3 + 4*self.P[4]*self.Q[0]*self.Q[3]**3 + 12*self.P[4]*self.Q[1]*self.Q[2]*self.Q[3]**2 + 4*self.P[4]*self.Q[2]**3*self.Q[3]
        output[10] = 4*self.P[4]*self.Q[1]*self.Q[3]**3 + 6*self.P[4]*self.Q[2]**2*self.Q[3]**2
        output[11] = 4*self.P[4]*self.Q[2]*self.Q[3]**3
        output[12] = self.P[4]*self.Q[3]**4


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



class EfficientPolynomialSearch_16_4(nn.Module):
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
        super(EfficientPolynomialSearch_16_4, self).__init__()
        # Set slack values for compilation
        degree, deg_q = 16, 4
        # Set the degree of R
        self.degree = 16
        self.deg_q = 4
        self.deg_p = self.degree // self.deg_q

        # Define the coefficients for P and Q
        self.P = nn.Parameter(torch.randn(self.deg_p + 1, dtype=torch.float64) / 10000)
        self.Q = nn.Parameter(torch.randn(self.deg_q + 1, dtype=torch.float64) / 10000)

    def forward(self) -> torch.Tensor:
        output = torch.zeros(self.degree + 1, dtype=torch.float64)

        # Update the output by the polynomial
        output[0] = self.P[0] + self.P[1]*self.Q[0] + self.P[2]*self.Q[0]**2 + self.P[3]*self.Q[0]**3 + self.P[4]*self.Q[0]**4
        output[1] = self.P[1]*self.Q[1] + 2*self.P[2]*self.Q[0]*self.Q[1] + 3*self.P[3]*self.Q[0]**2*self.Q[1] + 4*self.P[4]*self.Q[0]**3*self.Q[1]
        output[2] = self.P[1]*self.Q[2] + 2*self.P[2]*self.Q[0]*self.Q[2] + self.P[2]*self.Q[1]**2 + 3*self.P[3]*self.Q[0]**2*self.Q[2] + 3*self.P[3]*self.Q[0]*self.Q[1]**2 + 4*self.P[4]*self.Q[0]**3*self.Q[2] + 6*self.P[4]*self.Q[0]**2*self.Q[1]**2
        output[3] = self.P[1]*self.Q[3] + 2*self.P[2]*self.Q[0]*self.Q[3] + 2*self.P[2]*self.Q[1]*self.Q[2] + 3*self.P[3]*self.Q[0]**2*self.Q[3] + 6*self.P[3]*self.Q[0]*self.Q[1]*self.Q[2] + self.P[3]*self.Q[1]**3 + 4*self.P[4]*self.Q[0]**3*self.Q[3] + 12*self.P[4]*self.Q[0]**2*self.Q[1]*self.Q[2] + 4*self.P[4]*self.Q[0]*self.Q[1]**3
        output[4] = self.P[1]*self.Q[4] + 2*self.P[2]*self.Q[0]*self.Q[4] + 2*self.P[2]*self.Q[1]*self.Q[3] + self.P[2]*self.Q[2]**2 + 3*self.P[3]*self.Q[0]**2*self.Q[4] + 6*self.P[3]*self.Q[0]*self.Q[1]*self.Q[3] + 3*self.P[3]*self.Q[0]*self.Q[2]**2 + 3*self.P[3]*self.Q[1]**2*self.Q[2] + 4*self.P[4]*self.Q[0]**3*self.Q[4] + 12*self.P[4]*self.Q[0]**2*self.Q[1]*self.Q[3] + 6*self.P[4]*self.Q[0]**2*self.Q[2]**2 + 12*self.P[4]*self.Q[0]*self.Q[1]**2*self.Q[2] + self.P[4]*self.Q[1]**4
        output[5] = 2*self.P[2]*self.Q[1]*self.Q[4] + 2*self.P[2]*self.Q[2]*self.Q[3] + 6*self.P[3]*self.Q[0]*self.Q[1]*self.Q[4] + 6*self.P[3]*self.Q[0]*self.Q[2]*self.Q[3] + 3*self.P[3]*self.Q[1]**2*self.Q[3] + 3*self.P[3]*self.Q[1]*self.Q[2]**2 + 12*self.P[4]*self.Q[0]**2*self.Q[1]*self.Q[4] + 12*self.P[4]*self.Q[0]**2*self.Q[2]*self.Q[3] + 12*self.P[4]*self.Q[0]*self.Q[1]**2*self.Q[3] + 12*self.P[4]*self.Q[0]*self.Q[1]*self.Q[2]**2 + 4*self.P[4]*self.Q[1]**3*self.Q[2]
        output[6] = 2*self.P[2]*self.Q[2]*self.Q[4] + self.P[2]*self.Q[3]**2 + 6*self.P[3]*self.Q[0]*self.Q[2]*self.Q[4] + 3*self.P[3]*self.Q[0]*self.Q[3]**2 + 3*self.P[3]*self.Q[1]**2*self.Q[4] + 6*self.P[3]*self.Q[1]*self.Q[2]*self.Q[3] + self.P[3]*self.Q[2]**3 + 12*self.P[4]*self.Q[0]**2*self.Q[2]*self.Q[4] + 6*self.P[4]*self.Q[0]**2*self.Q[3]**2 + 12*self.P[4]*self.Q[0]*self.Q[1]**2*self.Q[4] + 24*self.P[4]*self.Q[0]*self.Q[1]*self.Q[2]*self.Q[3] + 4*self.P[4]*self.Q[0]*self.Q[2]**3 + 4*self.P[4]*self.Q[1]**3*self.Q[3] + 6*self.P[4]*self.Q[1]**2*self.Q[2]**2
        output[7] = 2*self.P[2]*self.Q[3]*self.Q[4] + 6*self.P[3]*self.Q[0]*self.Q[3]*self.Q[4] + 6*self.P[3]*self.Q[1]*self.Q[2]*self.Q[4] + 3*self.P[3]*self.Q[1]*self.Q[3]**2 + 3*self.P[3]*self.Q[2]**2*self.Q[3] + 12*self.P[4]*self.Q[0]**2*self.Q[3]*self.Q[4] + 24*self.P[4]*self.Q[0]*self.Q[1]*self.Q[2]*self.Q[4] + 12*self.P[4]*self.Q[0]*self.Q[1]*self.Q[3]**2 + 12*self.P[4]*self.Q[0]*self.Q[2]**2*self.Q[3] + 4*self.P[4]*self.Q[1]**3*self.Q[4] + 12*self.P[4]*self.Q[1]**2*self.Q[2]*self.Q[3] + 4*self.P[4]*self.Q[1]*self.Q[2]**3
        output[8] = self.P[2]*self.Q[4]**2 + 3*self.P[3]*self.Q[0]*self.Q[4]**2 + 6*self.P[3]*self.Q[1]*self.Q[3]*self.Q[4] + 3*self.P[3]*self.Q[2]**2*self.Q[4] + 3*self.P[3]*self.Q[2]*self.Q[3]**2 + 6*self.P[4]*self.Q[0]**2*self.Q[4]**2 + 24*self.P[4]*self.Q[0]*self.Q[1]*self.Q[3]*self.Q[4] + 12*self.P[4]*self.Q[0]*self.Q[2]**2*self.Q[4] + 12*self.P[4]*self.Q[0]*self.Q[2]*self.Q[3]**2 + 12*self.P[4]*self.Q[1]**2*self.Q[2]*self.Q[4] + 6*self.P[4]*self.Q[1]**2*self.Q[3]**2 + 12*self.P[4]*self.Q[1]*self.Q[2]**2*self.Q[3] + self.P[4]*self.Q[2]**4
        output[9] = 3*self.P[3]*self.Q[1]*self.Q[4]**2 + 6*self.P[3]*self.Q[2]*self.Q[3]*self.Q[4] + self.P[3]*self.Q[3]**3 + 12*self.P[4]*self.Q[0]*self.Q[1]*self.Q[4]**2 + 24*self.P[4]*self.Q[0]*self.Q[2]*self.Q[3]*self.Q[4] + 4*self.P[4]*self.Q[0]*self.Q[3]**3 + 12*self.P[4]*self.Q[1]**2*self.Q[3]*self.Q[4] + 12*self.P[4]*self.Q[1]*self.Q[2]**2*self.Q[4] + 12*self.P[4]*self.Q[1]*self.Q[2]*self.Q[3]**2 + 4*self.P[4]*self.Q[2]**3*self.Q[3]
        output[10] = 3*self.P[3]*self.Q[2]*self.Q[4]**2 + 3*self.P[3]*self.Q[3]**2*self.Q[4] + 12*self.P[4]*self.Q[0]*self.Q[2]*self.Q[4]**2 + 12*self.P[4]*self.Q[0]*self.Q[3]**2*self.Q[4] + 6*self.P[4]*self.Q[1]**2*self.Q[4]**2 + 24*self.P[4]*self.Q[1]*self.Q[2]*self.Q[3]*self.Q[4] + 4*self.P[4]*self.Q[1]*self.Q[3]**3 + 4*self.P[4]*self.Q[2]**3*self.Q[4] + 6*self.P[4]*self.Q[2]**2*self.Q[3]**2
        output[11] = 3*self.P[3]*self.Q[3]*self.Q[4]**2 + 12*self.P[4]*self.Q[0]*self.Q[3]*self.Q[4]**2 + 12*self.P[4]*self.Q[1]*self.Q[2]*self.Q[4]**2 + 12*self.P[4]*self.Q[1]*self.Q[3]**2*self.Q[4] + 12*self.P[4]*self.Q[2]**2*self.Q[3]*self.Q[4] + 4*self.P[4]*self.Q[2]*self.Q[3]**3
        output[12] = self.P[3]*self.Q[4]**3 + 4*self.P[4]*self.Q[0]*self.Q[4]**3 + 12*self.P[4]*self.Q[1]*self.Q[3]*self.Q[4]**2 + 6*self.P[4]*self.Q[2]**2*self.Q[4]**2 + 12*self.P[4]*self.Q[2]*self.Q[3]**2*self.Q[4] + self.P[4]*self.Q[3]**4
        output[13] = 4*self.P[4]*self.Q[1]*self.Q[4]**3 + 12*self.P[4]*self.Q[2]*self.Q[3]*self.Q[4]**2 + 4*self.P[4]*self.Q[3]**3*self.Q[4]
        output[14] = 4*self.P[4]*self.Q[2]*self.Q[4]**3 + 6*self.P[4]*self.Q[3]**2*self.Q[4]**2
        output[15] = 4*self.P[4]*self.Q[3]*self.Q[4]**3
        output[16] = self.P[4]*self.Q[4]**4


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



class EfficientPolynomialSearch_16_2(nn.Module):
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
        super(EfficientPolynomialSearch_16_2, self).__init__()
        # Set slack values for compilation
        degree, deg_q = 16, 4
        # Set the degree of R
        self.degree = 16
        self.deg_q = 2
        self.deg_p = self.degree // self.deg_q

        # Define the coefficients for P and Q
        self.P = nn.Parameter(torch.randn(self.deg_p + 1, dtype=torch.float64) / 10000)
        self.Q = nn.Parameter(torch.randn(self.deg_q + 1, dtype=torch.float64) / 10000)

    def forward(self) -> torch.Tensor:
        output = torch.zeros(self.degree + 1, dtype=torch.float64)

        # Update the output by the polynomial
        output[0] = self.P[0] + self.P[1]*self.Q[0] + self.P[2]*self.Q[0]**2 + self.P[3]*self.Q[0]**3 + self.P[4]*self.Q[0]**4 + self.P[5]*self.Q[0]**5 + self.P[6]*self.Q[0]**6 + self.P[7]*self.Q[0]**7 + self.P[8]*self.Q[0]**8
        output[1] = self.P[1]*self.Q[1] + 2*self.P[2]*self.Q[0]*self.Q[1] + 3*self.P[3]*self.Q[0]**2*self.Q[1] + 4*self.P[4]*self.Q[0]**3*self.Q[1] + 5*self.P[5]*self.Q[0]**4*self.Q[1] + 6*self.P[6]*self.Q[0]**5*self.Q[1] + 7*self.P[7]*self.Q[0]**6*self.Q[1] + 8*self.P[8]*self.Q[0]**7*self.Q[1]
        output[2] = self.P[1]*self.Q[2] + 2*self.P[2]*self.Q[0]*self.Q[2] + self.P[2]*self.Q[1]**2 + 3*self.P[3]*self.Q[0]**2*self.Q[2] + 3*self.P[3]*self.Q[0]*self.Q[1]**2 + 4*self.P[4]*self.Q[0]**3*self.Q[2] + 6*self.P[4]*self.Q[0]**2*self.Q[1]**2 + 5*self.P[5]*self.Q[0]**4*self.Q[2] + 10*self.P[5]*self.Q[0]**3*self.Q[1]**2 + 6*self.P[6]*self.Q[0]**5*self.Q[2] + 15*self.P[6]*self.Q[0]**4*self.Q[1]**2 + 7*self.P[7]*self.Q[0]**6*self.Q[2] + 21*self.P[7]*self.Q[0]**5*self.Q[1]**2 + 8*self.P[8]*self.Q[0]**7*self.Q[2] + 28*self.P[8]*self.Q[0]**6*self.Q[1]**2
        output[3] = 2*self.P[2]*self.Q[1]*self.Q[2] + 6*self.P[3]*self.Q[0]*self.Q[1]*self.Q[2] + self.P[3]*self.Q[1]**3 + 12*self.P[4]*self.Q[0]**2*self.Q[1]*self.Q[2] + 4*self.P[4]*self.Q[0]*self.Q[1]**3 + 20*self.P[5]*self.Q[0]**3*self.Q[1]*self.Q[2] + 10*self.P[5]*self.Q[0]**2*self.Q[1]**3 + 30*self.P[6]*self.Q[0]**4*self.Q[1]*self.Q[2] + 20*self.P[6]*self.Q[0]**3*self.Q[1]**3 + 42*self.P[7]*self.Q[0]**5*self.Q[1]*self.Q[2] + 35*self.P[7]*self.Q[0]**4*self.Q[1]**3 + 56*self.P[8]*self.Q[0]**6*self.Q[1]*self.Q[2] + 56*self.P[8]*self.Q[0]**5*self.Q[1]**3
        output[4] = self.P[2]*self.Q[2]**2 + 3*self.P[3]*self.Q[0]*self.Q[2]**2 + 3*self.P[3]*self.Q[1]**2*self.Q[2] + 6*self.P[4]*self.Q[0]**2*self.Q[2]**2 + 12*self.P[4]*self.Q[0]*self.Q[1]**2*self.Q[2] + self.P[4]*self.Q[1]**4 + 10*self.P[5]*self.Q[0]**3*self.Q[2]**2 + 30*self.P[5]*self.Q[0]**2*self.Q[1]**2*self.Q[2] + 5*self.P[5]*self.Q[0]*self.Q[1]**4 + 15*self.P[6]*self.Q[0]**4*self.Q[2]**2 + 60*self.P[6]*self.Q[0]**3*self.Q[1]**2*self.Q[2] + 15*self.P[6]*self.Q[0]**2*self.Q[1]**4 + 21*self.P[7]*self.Q[0]**5*self.Q[2]**2 + 105*self.P[7]*self.Q[0]**4*self.Q[1]**2*self.Q[2] + 35*self.P[7]*self.Q[0]**3*self.Q[1]**4 + 28*self.P[8]*self.Q[0]**6*self.Q[2]**2 + 168*self.P[8]*self.Q[0]**5*self.Q[1]**2*self.Q[2] + 70*self.P[8]*self.Q[0]**4*self.Q[1]**4
        output[5] = 3*self.P[3]*self.Q[1]*self.Q[2]**2 + 12*self.P[4]*self.Q[0]*self.Q[1]*self.Q[2]**2 + 4*self.P[4]*self.Q[1]**3*self.Q[2] + 30*self.P[5]*self.Q[0]**2*self.Q[1]*self.Q[2]**2 + 20*self.P[5]*self.Q[0]*self.Q[1]**3*self.Q[2] + self.P[5]*self.Q[1]**5 + 60*self.P[6]*self.Q[0]**3*self.Q[1]*self.Q[2]**2 + 60*self.P[6]*self.Q[0]**2*self.Q[1]**3*self.Q[2] + 6*self.P[6]*self.Q[0]*self.Q[1]**5 + 105*self.P[7]*self.Q[0]**4*self.Q[1]*self.Q[2]**2 + 140*self.P[7]*self.Q[0]**3*self.Q[1]**3*self.Q[2] + 21*self.P[7]*self.Q[0]**2*self.Q[1]**5 + 168*self.P[8]*self.Q[0]**5*self.Q[1]*self.Q[2]**2 + 280*self.P[8]*self.Q[0]**4*self.Q[1]**3*self.Q[2] + 56*self.P[8]*self.Q[0]**3*self.Q[1]**5
        output[6] = self.P[3]*self.Q[2]**3 + 4*self.P[4]*self.Q[0]*self.Q[2]**3 + 6*self.P[4]*self.Q[1]**2*self.Q[2]**2 + 10*self.P[5]*self.Q[0]**2*self.Q[2]**3 + 30*self.P[5]*self.Q[0]*self.Q[1]**2*self.Q[2]**2 + 5*self.P[5]*self.Q[1]**4*self.Q[2] + 20*self.P[6]*self.Q[0]**3*self.Q[2]**3 + 90*self.P[6]*self.Q[0]**2*self.Q[1]**2*self.Q[2]**2 + 30*self.P[6]*self.Q[0]*self.Q[1]**4*self.Q[2] + self.P[6]*self.Q[1]**6 + 35*self.P[7]*self.Q[0]**4*self.Q[2]**3 + 210*self.P[7]*self.Q[0]**3*self.Q[1]**2*self.Q[2]**2 + 105*self.P[7]*self.Q[0]**2*self.Q[1]**4*self.Q[2] + 7*self.P[7]*self.Q[0]*self.Q[1]**6 + 56*self.P[8]*self.Q[0]**5*self.Q[2]**3 + 420*self.P[8]*self.Q[0]**4*self.Q[1]**2*self.Q[2]**2 + 280*self.P[8]*self.Q[0]**3*self.Q[1]**4*self.Q[2] + 28*self.P[8]*self.Q[0]**2*self.Q[1]**6
        output[7] = 4*self.P[4]*self.Q[1]*self.Q[2]**3 + 20*self.P[5]*self.Q[0]*self.Q[1]*self.Q[2]**3 + 10*self.P[5]*self.Q[1]**3*self.Q[2]**2 + 60*self.P[6]*self.Q[0]**2*self.Q[1]*self.Q[2]**3 + 60*self.P[6]*self.Q[0]*self.Q[1]**3*self.Q[2]**2 + 6*self.P[6]*self.Q[1]**5*self.Q[2] + 140*self.P[7]*self.Q[0]**3*self.Q[1]*self.Q[2]**3 + 210*self.P[7]*self.Q[0]**2*self.Q[1]**3*self.Q[2]**2 + 42*self.P[7]*self.Q[0]*self.Q[1]**5*self.Q[2] + self.P[7]*self.Q[1]**7 + 280*self.P[8]*self.Q[0]**4*self.Q[1]*self.Q[2]**3 + 560*self.P[8]*self.Q[0]**3*self.Q[1]**3*self.Q[2]**2 + 168*self.P[8]*self.Q[0]**2*self.Q[1]**5*self.Q[2] + 8*self.P[8]*self.Q[0]*self.Q[1]**7
        output[8] = self.P[4]*self.Q[2]**4 + 5*self.P[5]*self.Q[0]*self.Q[2]**4 + 10*self.P[5]*self.Q[1]**2*self.Q[2]**3 + 15*self.P[6]*self.Q[0]**2*self.Q[2]**4 + 60*self.P[6]*self.Q[0]*self.Q[1]**2*self.Q[2]**3 + 15*self.P[6]*self.Q[1]**4*self.Q[2]**2 + 35*self.P[7]*self.Q[0]**3*self.Q[2]**4 + 210*self.P[7]*self.Q[0]**2*self.Q[1]**2*self.Q[2]**3 + 105*self.P[7]*self.Q[0]*self.Q[1]**4*self.Q[2]**2 + 7*self.P[7]*self.Q[1]**6*self.Q[2] + 70*self.P[8]*self.Q[0]**4*self.Q[2]**4 + 560*self.P[8]*self.Q[0]**3*self.Q[1]**2*self.Q[2]**3 + 420*self.P[8]*self.Q[0]**2*self.Q[1]**4*self.Q[2]**2 + 56*self.P[8]*self.Q[0]*self.Q[1]**6*self.Q[2] + self.P[8]*self.Q[1]**8
        output[9] = 5*self.P[5]*self.Q[1]*self.Q[2]**4 + 30*self.P[6]*self.Q[0]*self.Q[1]*self.Q[2]**4 + 20*self.P[6]*self.Q[1]**3*self.Q[2]**3 + 105*self.P[7]*self.Q[0]**2*self.Q[1]*self.Q[2]**4 + 140*self.P[7]*self.Q[0]*self.Q[1]**3*self.Q[2]**3 + 21*self.P[7]*self.Q[1]**5*self.Q[2]**2 + 280*self.P[8]*self.Q[0]**3*self.Q[1]*self.Q[2]**4 + 560*self.P[8]*self.Q[0]**2*self.Q[1]**3*self.Q[2]**3 + 168*self.P[8]*self.Q[0]*self.Q[1]**5*self.Q[2]**2 + 8*self.P[8]*self.Q[1]**7*self.Q[2]
        output[10] = self.P[5]*self.Q[2]**5 + 6*self.P[6]*self.Q[0]*self.Q[2]**5 + 15*self.P[6]*self.Q[1]**2*self.Q[2]**4 + 21*self.P[7]*self.Q[0]**2*self.Q[2]**5 + 105*self.P[7]*self.Q[0]*self.Q[1]**2*self.Q[2]**4 + 35*self.P[7]*self.Q[1]**4*self.Q[2]**3 + 56*self.P[8]*self.Q[0]**3*self.Q[2]**5 + 420*self.P[8]*self.Q[0]**2*self.Q[1]**2*self.Q[2]**4 + 280*self.P[8]*self.Q[0]*self.Q[1]**4*self.Q[2]**3 + 28*self.P[8]*self.Q[1]**6*self.Q[2]**2
        output[11] = 6*self.P[6]*self.Q[1]*self.Q[2]**5 + 42*self.P[7]*self.Q[0]*self.Q[1]*self.Q[2]**5 + 35*self.P[7]*self.Q[1]**3*self.Q[2]**4 + 168*self.P[8]*self.Q[0]**2*self.Q[1]*self.Q[2]**5 + 280*self.P[8]*self.Q[0]*self.Q[1]**3*self.Q[2]**4 + 56*self.P[8]*self.Q[1]**5*self.Q[2]**3
        output[12] = self.P[6]*self.Q[2]**6 + 7*self.P[7]*self.Q[0]*self.Q[2]**6 + 21*self.P[7]*self.Q[1]**2*self.Q[2]**5 + 28*self.P[8]*self.Q[0]**2*self.Q[2]**6 + 168*self.P[8]*self.Q[0]*self.Q[1]**2*self.Q[2]**5 + 70*self.P[8]*self.Q[1]**4*self.Q[2]**4
        output[13] = 7*self.P[7]*self.Q[1]*self.Q[2]**6 + 56*self.P[8]*self.Q[0]*self.Q[1]*self.Q[2]**6 + 56*self.P[8]*self.Q[1]**3*self.Q[2]**5
        output[14] = self.P[7]*self.Q[2]**7 + 8*self.P[8]*self.Q[0]*self.Q[2]**7 + 28*self.P[8]*self.Q[1]**2*self.Q[2]**6
        output[15] = 8*self.P[8]*self.Q[1]*self.Q[2]**7
        output[16] = self.P[8]*self.Q[2]**8


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
