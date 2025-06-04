import numpy as np
import sympy as sp
from sympy import lambdify

# --- Parameters ---
theta_val = np.pi / 4           # 45 degrees
x_val, y_val = 0.01, 0.0        # internal magnet position
x_m_val, y_m_val = 0.0, 0.0     # external magnet at origin
psi_val = 0.0                   # external dipole pointing right (along x-axis)
mu0 = 4 * np.pi * 1e-7
MAGNET_M = 1.0                  # normalized magnetic moment
r_val = np.sqrt((x_val - x_m_val)**2 + (y_val - y_m_val)**2)
C_val = mu0 * MAGNET_M / (4 * np.pi * r_val**3)

# --- Symbolic expression for derivative ---
theta, x, y, x_m, y_m, psi = sp.symbols('theta x y x_m y_m psi')
C = sp.Symbol('C')

px = x - x_m
py = y - y_m
r = sp.sqrt(px**2 + py**2)
a = px / r
b = py / r

# External magnet unit vector
m_hat = sp.Matrix([sp.cos(psi), sp.sin(psi)])
p_hat = sp.Matrix([a, b])
dot_pm = p_hat.dot(m_hat)
b_vec = C * (3 * dot_pm * p_hat - m_hat)

# Internal dipole and its derivative
Rm = sp.Matrix([sp.cos(theta), sp.sin(theta)])
first_term_expr = sp.simplify(sp.diff(Rm.dot(b_vec), theta))

# Lambdify
f_first_term = lambdify((theta, x, y, x_m, y_m, psi, C), first_term_expr, modules='numpy')

# Evaluate symbolic derivative
val1 = f_first_term(theta_val, x_val, y_val, x_m_val, y_m_val, psi_val, C_val)
print("Symbolic expanded value:", val1)

# --- Manual computation using Bx, By ---
px = x_val - x_m_val
py = y_val - y_m_val
r3 = r_val**3
a = px / r_val
b = py / r_val
dot_pm = a * np.cos(psi_val) + b * np.sin(psi_val)

Bx = C_val * (3 * dot_pm * a - np.cos(psi_val))
By = C_val * (3 * dot_pm * b - np.sin(psi_val))

val2 = -np.sin(theta_val) * Bx + np.cos(theta_val) * By
print("Manual value:", val2)
