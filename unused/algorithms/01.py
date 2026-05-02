"""
Yuli Tshuva
First Attempt with SymPy.
Assigning polynomials.
"""

from sympy import symbols, expand
from functions import *

# Define symbolic variables
x, y = symbols('x y')
g0, g1, g2, h0, h1, h2 = symbols('g0 g1 g2 h0 h1 h2')

# Define the polynomials
P = g2*y**2 + (g1*y) + g0  # First polynomial: P(y)
Q = h2*x**2 + (h1*x) + h0  # Second polynomial: Q(x)

# Substitute y = Q(x) into P(y)
result = expand(P.subs(y, Q))

# Group the result by powers of x
result = result.collect(x)

print(result)

