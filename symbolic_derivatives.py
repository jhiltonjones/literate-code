import numpy as np
import sympy as sp
import dill as pickle

# --- Step 1: Rebuild symbolic second term ---
# Redefine symbolic variables
theta, x, y, x_m, y_m, psi = sp.symbols('theta x y x_m y_m psi')
C = sp.Symbol('C')
C_grad = sp.Symbol('C_grad')
px = x - x_m
py = y - y_m
r = sp.sqrt(px**2 + py**2)
a = px / r
b = py / r

# Define external magnet unit vector
m_hat = sp.Matrix([sp.cos(psi), sp.sin(psi)])
p_hat = sp.Matrix([a, b])

# Magnetic field vector b
dot_pm = p_hat.dot(m_hat)
b_vec = C * (3 * dot_pm * p_hat - m_hat)



# Internal rotated dipole and dx/dtheta
# Re-define for clarity
theta = sp.Symbol('theta')
Rm = sp.Matrix([sp.cos(theta), sp.sin(theta)])
Rm_dtheta = sp.Matrix([-sp.sin(theta), sp.cos(theta)])

# dot_product = Rm.dot(b_vec)
# dot_derivative = sp.simplify(sp.diff(dot_product, theta))
# manual_expected = sp.simplify(Rm_dtheta.dot(b_vec))

# # Check equality

dx_dtheta = sp.Matrix([-sp.sin(theta), sp.cos(theta)])

# Gradient of b with respect to x, y
grad_b = sp.Matrix([[sp.diff(bi, x), sp.diff(bi, y)] for bi in b_vec])
print("Solve first term")
# Second term of chain rule
first_term_expr = sp.simplify(sp.diff(Rm.dot(b_vec), theta))
# print(f"First term is : {first_term_expr}")
# assert first_term_expr.equals(manual_expected), "Derivative does not match manual expected value"

print("Solve first term2")
f_first_term = sp.lambdify(
    (theta, x, y, x_m, y_m, psi, C),
    first_term_expr,
    modules="numpy"
)


m1 = sp.cos(psi)
m2 = sp.sin(psi)
a_sym = px / r
b_sym = py / r

# Scalar dot product
dot_pm = a_sym * m1 + b_sym * m2

# Outer products (manual scalars)
ppT = sp.Matrix([[a_sym * a_sym, a_sym * b_sym],
                 [b_sym * a_sym, b_sym * b_sym]])

pmT = sp.Matrix([[a_sym * m1, a_sym * m2],
                 [b_sym * m1, b_sym * m2]])

mpT = sp.Matrix([[m1 * a_sym, m1 * b_sym],
                 [m2 * a_sym, m2 * b_sym]])

I2 = sp.eye(2)
Z = I2 - 5 * ppT

grad_b_explicit = C_grad * (
    pmT + dot_pm * I2 + Z * mpT
)

Rm = sp.Matrix([sp.cos(theta), sp.sin(theta)])
dx_dtheta = sp.Matrix([-sp.sin(theta), sp.cos(theta)])

# Second term
second_term_expr_explicit = sp.simplify((grad_b_explicit.T * Rm).dot(dx_dtheta))
print("Solve second term1")
# Lambdify
f_second_term = sp.lambdify(
    (theta, x, y, x_m, y_m, psi, C_grad),
    second_term_expr_explicit,
    modules='numpy'
)
print("Solve second term2")

with open('magnetic_field_terms.pkl', 'wb') as f:
    pickle.dump((f_first_term, f_second_term), f)
