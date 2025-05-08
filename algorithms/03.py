"""
Trying to improve 02
"""

import numpy as np
import sympy as sp
from sympy import expand, symbols
from functions import *
import time
import math

DEG_P, DEG_Q = 15, 2  # Maximum degree for polynomials
CONFIDENCE = 0.8


def recover_P(PQ_expr, Q_expr, var=sp.Symbol('x')):
    # Define polynomials
    x = var
    Qx = sp.Poly(Q, x)
    PQx = sp.Poly(PQ, x)

    # Estimate degree of P(x): deg(PQ) = deg(P) * deg(Q) → deg(P) = deg(PQ) // deg(Q)
    deg_Q = Qx.degree()
    deg_PQ = PQx.degree()
    deg_P = deg_PQ // deg_Q

    p = sp.symbols(f'p0:{deg_P + 1}')
    P_expr = sum(p[i] * x ** i for i in range(deg_P + 1))

    num_points = math.ceil((2 - CONFIDENCE) * (deg_P + 1))
    equations = []

    for i in range(-num_points // 2, num_points // 2 + 1):
        Q_val = Q_expr.subs(x, i)
        lhs = P_expr.subs(x, Q_val)
        rhs = PQ_expr.subs(x, i)
        equations.append(sp.Eq(lhs, rhs))

    solution = sp.linsolve(equations, p)
    if not solution:
        return None

    coeffs = list(solution)[0]
    return sp.Poly(sum(coeffs[i] * x ** i for i in range(deg_P + 1)), x)


def recover_Q(PQ_expr, P_expr, deg_Q, var=sp.Symbol('x')):
    x = var
    q = sp.symbols(f'q0:{deg_Q + 1}')
    Q_expr = sum(q[i] * x ** i for i in range(deg_Q + 1))

    num_points = math.ceil((2 - CONFIDENCE) * (deg_Q + 1))
    equations = []

    for i in range(-num_points // 2, num_points // 2 + 1):
        Q_val = Q_expr.subs(x, i)
        lhs = P_expr.subs(x, Q_val)
        rhs = PQ_expr.subs(x, i)
        equations.append(sp.Eq(lhs, rhs))

    solution = sp.solve(equations, q)
    if not solution:
        return None

    coeffs = list(solution)[0]
    return sp.Poly(sum(coeffs[i] * x ** i for i in range(deg_Q + 1)), x)


# Create symbols
x, y = symbols('x y')

# Define the polynomials
Q = generate_polynomial(DEG_Q, x)
P = generate_polynomial(DEG_P, x)

start = time.time()
PQ = expand(P.subs(x, Q))
end = time.time()
print("P(Q(x)):")
present_result(PQ)
print(f'Took {end - start:.3f} seconds to develop P(Q(x)).')

# Recover P(x)
start = time.time()
P_recovered = recover_P(PQ, Q, x)
end = time.time()

print("\n", "-" * 50, "\n")

print("P(x):")
present_result(P)
print("Recovered P(x):")
# convert P recovered to expression
P_recovered = P_recovered.as_expr()
present_result(P_recovered)
print("Took", end - start, "seconds to recover P(x).")

print("\n", "-" * 50, "\n")

# Recover Q(x)
start = time.time()
Q_recovered = recover_Q(PQ, P, DEG_Q, x)
end = time.time()
print("Took", end - start, "seconds to recover Q(x).")

print("Q(x):")
present_result(Q)
print("Recovered Q(x):")
# convert Q recovered to expression
Q_recovered = Q_recovered.as_expr()
present_result(Q_recovered)
