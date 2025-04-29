"""
The Kozen–Landau algorithm
"""

from sympy import symbols, Poly, simplify, Symbol, Eq, solve, expand
import pickle
from os.path import join
from functions import *
from tqdm.auto import tqdm

OUTPUT_PATH = join("output", "decompose.pkl")
x = symbols('x')


def decompose(A_poly):
    A = Poly(A_poly, x)
    n = A.degree()

    # Define a decompositions dictionary
    decompositions = {}

    # Try all proper divisors of the degree
    for d in tqdm(range(1, n), total=n-1):
        if n % d != 0:
            continue
        k = n // d

        # Define C(x) as monic degree d: x^d + a_{d-1}x^{d-1} + ... + a0
        a = symbols(f'a0:{d}')
        C = x ** d + sum(a[i] * x ** i for i in range(1, d))

        # Define B(x) = b0 + b1*x + ... + bk*x^k
        b = symbols(f'b0:{k}')
        B = C**k + sum(b[i] * C ** i for i in range(k))

        # Try to match A(x) = B(C(x))
        B_expanded = Poly(expand(B), x)
        A_coeffs = A.all_coeffs()
        B_coeffs = B_expanded.all_coeffs()

        # Pad shorter list with zeros
        diff = len(A_coeffs) - len(B_coeffs)
        if diff > 0:
            B_coeffs = [0] * diff + B_coeffs
        elif diff < 0:
            continue  # B is too large to match A

        # Set up equations to solve
        equations = [Eq(a_c, b_c) for a_c, b_c in zip(A_coeffs, B_coeffs)]

        try:
            sol = solve(equations, a + b, dict=True)
            if sol:
                # Extract solution and identify free symbols
                concrete_sol = sol[0]

                # Now evaluate the polynomials
                C_eval = simplify(C.subs(concrete_sol))
                B_eval = simplify(x**k + sum(b[i] * x ** i for i in range(k)).subs(concrete_sol))

                decompositions[d] = {
                    'C': C_eval,
                    'B': B_eval,
                }

        except Exception:
            continue

    return decompositions


# Example: A(x) = (x^2 + 1)^2 = x^4 + 2x^2 + 1
A = expand(generate_polynomial(20, x))
decompositions = decompose(A)

for key, val in decompositions.items():
    print(key, ":", val)


# Save the decompositions to a file
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(decompositions, f)
