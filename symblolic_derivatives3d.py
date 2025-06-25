import sympy as sp
import dill as pickle

# --- Step 1: Define full 3D symbolic variables ---
x, y, z = sp.symbols('x y z')
x_m, y_m, z_m = sp.symbols('x_m y_m z_m')
psi, phi = sp.symbols('psi phi')  # spherical coordinates for external dipole
alpha, beta = sp.symbols('alpha beta')  # new angles for internal dipole
C = sp.Symbol('C')
C_grad = sp.Symbol('C_grad')

# Position vector and unit vector p̂
px = x - x_m
py = y - y_m
pz = z - z_m
r_vec = sp.Matrix([px, py, pz])
r = sp.sqrt(r_vec.dot(r_vec))
p_hat = r_vec / r

# External dipole vector (unit, spherical)
m_hat = sp.Matrix([
    sp.sin(psi) * sp.cos(phi),
    sp.sin(psi) * sp.sin(phi),
    sp.cos(psi)
])

# Magnetic field from dipole
dot_pm = p_hat.dot(m_hat)
b_vec = C * (3 * dot_pm * p_hat - m_hat)

# Internal dipole orientation (general 3D, spherical)
Rm = sp.Matrix([
    sp.sin(alpha) * sp.cos(beta),
    sp.sin(alpha) * sp.sin(beta),
    sp.cos(alpha)
])

# Partial derivatives with respect to internal angles
Rm_dalpha = Rm.diff(alpha)
Rm_dbeta = Rm.diff(beta)

# First term: gradient of (Rm ⋅ B) with respect to internal angles
dot_product = Rm.dot(b_vec)
first_term_alpha = sp.simplify(dot_product.diff(alpha))
first_term_beta = sp.simplify(dot_product.diff(beta))

f_first_term_alpha = sp.lambdify(
    (alpha, beta, x, y, z, x_m, y_m, z_m, psi, phi, C),
    first_term_alpha,
    modules='numpy'
)
f_first_term_beta = sp.lambdify(
    (alpha, beta, x, y, z, x_m, y_m, z_m, psi, phi, C),
    first_term_beta,
    modules='numpy'
)

# Gradient of B
pmT = p_hat * m_hat.T
mpT = m_hat * p_hat.T
ppT = p_hat * p_hat.T
I3 = sp.eye(3)
Z = I3 - 5 * ppT

grad_b_explicit = C_grad * (pmT + dot_pm * I3 + Z * mpT)

# Second term: torque from field gradient
second_term_expr_alpha = sp.simplify((grad_b_explicit.T * Rm).dot(Rm_dalpha))
second_term_expr_beta = sp.simplify((grad_b_explicit.T * Rm).dot(Rm_dbeta))

f_second_term_alpha = sp.lambdify(
    (alpha, beta, x, y, z, x_m, y_m, z_m, psi, phi, C_grad),
    second_term_expr_alpha,
    modules='numpy'
)
f_second_term_beta = sp.lambdify(
    (alpha, beta, x, y, z, x_m, y_m, z_m, psi, phi, C_grad),
    second_term_expr_beta,
    modules='numpy'
)

# Save the new full 3D functions
with open('magnetic_field_terms_3D_full.pkl', 'wb') as f:
    pickle.dump(
        (f_first_term_alpha, f_first_term_beta, f_second_term_alpha, f_second_term_beta), f
    )
"3D symbolic magnetic torque terms (α, β) saved."
