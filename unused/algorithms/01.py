"""
Yuli Tshuva
First Attempt with SymPy.
Assigning polynomials.
"""

from sympy import symbols, expand
from functions import *

# Define symbolic variables
x, y = symbols('x y')

# Define the polynomials
P = 3*y**2 + (2*y) + 1  # First polynomial: P(y)
Q = 6*x**2 + (5*x) + 4  # Second polynomial: Q(x)

# Substitute y = Q(x) into P(y)
result = expand(P.subs(y, Q))

present_result(result)

