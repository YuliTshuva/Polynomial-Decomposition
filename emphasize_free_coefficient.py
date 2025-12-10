"""
P(x): [-59, -44, -8, -21, -45, -16]
Q(x): [47, -26, 58, 51]
"""

import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from os.path import join
import numpy as np

rcParams["font.family"] = "Times New Roman"

x = sp.symbols("x")
p0, p1, p2, p3, p4 = sp.symbols("p0 p1 p2 p3 p4")

c = 51


def solution_for_c(c):
    p = p0 + p1 * x + p2 * x ** 2 + p3 * x ** 3 + p4 * x ** 4 - 59 * x ** 5
    q = 47 * x ** 3 - 26 * x ** 2 + 58 * x + c

    R = [-13531355413.0, 37427153270.0, -124900107230.0, 134024269071.0, -202775379472.0, -99425264576.0, 41478905686.0,
         -431139381588.0, -169283178396.0, -163424044063.0, -453301229236.0, -249550989008.0, -157950418958.0,
         -213096742104.0, -117110437956.0, -20655276793.0][::-1]
    r = sum([R[i] * x ** i for i in range(len(R))])

    # Create the equation
    eq = sp.Eq(p.subs(x, q), r)
    # Solve the equation
    sol = sp.solve(eq, [p0, p1, p2, p3, p4], dict=True)
    if isinstance(sol, list):
        sol = sol[0]
    try:
        list_sol = [-59, sol[p4], sol[p3], sol[p2], sol[p1], sol[p0]]
    except:
        list_sol = None

    return list_sol


def plot_bar(ps, deviation, save_path=None):
    plt.figure()
    plt.bar(range(len(ps)), np.abs(ps), color="royalblue", edgecolor="black")
    plt.xticks(range(len(ps)), [f"p{5 - i}" for i in range(len(ps))])
    plt.yticks(np.abs(ps).astype(np.int64))
    plt.xlabel("Coefficients", fontsize=15)
    plt.ylabel("Value", fontsize=15)
    sign = "+" if deviation >= 0 else "-"
    plt.title(f"P's coefficients for c = 51 {sign} {abs(deviation)}", fontsize=20)
    if save_path:
        plt.savefig(save_path)
    plt.close()


# Calculate solution for varying cs (around the target c)
sol_to_c = {}
for i in range(10):
    sol_to_c[c + i] = solution_for_c(c + i)
    sol_to_c[c - i] = solution_for_c(c - i)

working_dir = join("plots", "constant_explained")
os.makedirs(working_dir, exist_ok=True)

# for c_i in sol_to_c:
#     sol = sol_to_c[c_i]
#     if sol is None:
#         continue
#     deviation = c_i - c
#     plot_bar(sol, deviation, save_path=join(working_dir, f"c_{c_i}.png"))

# Plot in one plot the solutions for c = 51 +/- 1, 2
deviations = [0, 1, 2]
fig, ax = plt.subplots(nrows=2, ncols=2*len(deviations) - 1, figsize=(20, 5))
for idx, deviation in enumerate(deviations):
    for sign in [-1, 1]:
        c_i = c + sign * deviation
        sol = sol_to_c[c_i]
        if sol is None:
            continue
        center = len(deviations) - 1
        a = np.array(sol)
        ax[center + sign * deviation].bar(range(len(sol)), np.array(a > 0, dtype=np.float64) * a, color="royalblue", edgecolor="black")
        ax[center + sign * deviation].bar(range(len(sol)),  np.array(a < 0, dtype=np.float64) * -a, color="red", edgecolor="black")
        ax[center + sign * deviation].set_xticks(range(len(sol)))
        ax[center + sign * deviation].set_xticklabels([f"b{5 - i}" for i in range(len(sol))])
        ax[center + sign * deviation].set_yticks(ticks=np.abs(sol).astype(np.int64), labels=np.abs(sol).astype(np.int64))
        ax[center + sign * deviation].set_xlabel("Coefficients", fontsize=18)
        if center + sign * deviation == 0:
            ax[center + sign * deviation].set_ylabel("Value", fontsize=18)
        sign_str = "+" if sign == 1 else "-"
        ax[center + sign * deviation].set_title(f"$c_0$ = 51 {sign_str} {deviation}", fontsize=24)

plt.suptitle("g's coefficients for varying free coefficient", fontsize=35)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(join(working_dir, f"all_c_variations.png"))
plt.show()
