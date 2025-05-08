import sympy as sp
from sympy import expand, symbols
from functions import *
import random
import time


def recover_p_from_composition(PQ, Q, var=sp.Symbol('x')):
    # Define polynomials
    x = var
    Qx = sp.Poly(Q, x)
    PQx = sp.Poly(PQ, x)

    # Estimate degree of P(x): deg(PQ) = deg(P) * deg(Q) → deg(P) = deg(PQ) // deg(Q)
    deg_Q = Qx.degree()
    deg_PQ = PQx.degree()
    deg_P = deg_PQ // deg_Q

    # Create unknown coefficients for P(x)
    coeffs = symbols(f'a0:{deg_P + 1}')  # a0, a1, ..., a_deg_P
    Px = sum(coeffs[i] * x ** i for i in range(deg_P + 1))

    # Compose P(Q(x))
    P_of_Q = Px.subs(x, Q).expand()
    P_of_Q_poly = sp.Poly(P_of_Q, x)
    PQx_poly = sp.Poly(PQ, x)

    # Create equations by matching coefficients
    eqns = []
    for exp in range(max(PQx_poly.degree(), P_of_Q_poly.degree()) + 1):
        lhs = P_of_Q_poly.coeff_monomial(x ** exp)
        rhs = PQx_poly.coeff_monomial(x ** exp)
        eqns.append(sp.Eq(lhs, rhs))

    # Solve for coefficients of P(x)
    solution = sp.solve(eqns, coeffs)

    # Substitute solution into P(x)
    P_final = sp.Poly(Px.subs(solution), x)

    return P_final


def recover_Q_from_composition(PQx_expr, Px_expr, degree_Q_guess, var=sp.Symbol('x')):
    x = var

    # Define symbolic coefficients for Q(x)
    q_coeffs = sp.symbols(f'q0:{degree_Q_guess + 1}')
    Qx_expr = sum(q_coeffs[i] * x ** i for i in range(degree_Q_guess + 1))

    # Compose P(Q(x))
    Px_poly = sp.Poly(Px_expr, x)
    P_of_Q = Px_expr.subs(x, Qx_expr).expand()
    P_of_Q_poly = sp.Poly(P_of_Q, x)
    PQx_poly = sp.Poly(PQx_expr, x)

    # Match coefficients
    max_deg = max(P_of_Q_poly.degree(), PQx_poly.degree())
    eqns = []
    for deg in range(max_deg + 1):
        lhs = P_of_Q_poly.coeff_monomial(x ** deg)
        rhs = PQx_poly.coeff_monomial(x ** deg)
        eqns.append(sp.Eq(lhs, rhs))

    # Solve for q_coeffs
    solution = sp.solve(eqns, q_coeffs)

    if not solution:
        return None  # Could not solve for Q

    # Normalize the solution format
    if isinstance(solution, list):
        solution = solution[0]  # take first if it's a list of solutions
    if isinstance(solution, tuple):
        solution = dict(zip(q_coeffs, solution))  # convert tuple to dict

    Qx_final = sum(solution[q] * x ** i for i, q in enumerate(q_coeffs))
    return sp.Poly(Qx_final, x)


# Create symbols
x, y = symbols('x y')

# Define the polynomials
deg_Q, deg_P = 5, 5
Q = generate_polynomial(deg_Q, x)
P = generate_polynomial(deg_P, x)
PQ = expand(P.subs(x, Q))

print("P(Q(x)):")
present_result(PQ)

# Recover P(x)
start = time.time()
P_recovered = recover_p_from_composition(PQ, Q, x)
end = time.time()
print("Took", end - start, "seconds to recover P(x).")

print("\n", "-" * 50, "\n")

print("P(x):")
present_result(P)
print("Recovered P(x):")
# convert P recovered to expression
P_recovered = P_recovered.as_expr()
present_result(P_recovered)

print("\n", "-" * 50, "\n")

# Recover Q(x)
start = time.time()
Q_recovered = recover_Q_from_composition(PQ, P, deg_Q, x)
end = time.time()
print("Took", end - start, "seconds to recover Q(x).")

print("Q(x):")
present_result(Q)
print("Recovered Q(x):")
# convert Q recovered to expression
Q_recovered = Q_recovered.as_expr()
present_result(Q_recovered)
