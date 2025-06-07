import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

# Constants
L = 0.05  # rod length in meters
r = 0.00054
E = 3.0e6
A = jnp.pi * r**2
I = jnp.pi * r**4 / 4.0
MU0 = 4 * jnp.pi * 1e-7
M = 8000
MAGNET_M = 318

# Beam ODE definition
def dF_dtheta(theta, x, y, x_m, y_m, psi):
    px = x - x_m
    py = y - y_m
    r2 = px**2 + py**2
    r = jnp.sqrt(r2 + 1e-10)  # avoid zero division
    C_val = MU0 * MAGNET_M / (4 * jnp.pi * r**3)

    a = px / r
    b = py / r
    m_hat = jnp.array([jnp.cos(psi), jnp.sin(psi)])
    p_hat = jnp.array([a, b])
    dot_pm = jnp.dot(p_hat, m_hat)
    b_vec = C_val * (3 * dot_pm * p_hat - m_hat)

    m_local = jnp.array([jnp.cos(theta), jnp.sin(theta)])
    dR_dtheta = jnp.array([-jnp.sin(theta), jnp.cos(theta)])
    return jnp.dot(dR_dtheta, b_vec)

def beam_ode(state, s, magnet_params):
    theta, dtheta, x, y = state
    x_m, y_m, psi = magnet_params
    ddtheta = -(A * M / (E * I)) * dF_dtheta(theta, x, y, x_m, y_m, psi)
    dx = jnp.cos(theta)
    dy = jnp.sin(theta)
    return jnp.array([dtheta, ddtheta, dx, dy])

# Integration wrapper
def solve_beam(x_m, y_m, psi, k0_init=0.0, N=1000):
    s_vals = jnp.linspace(0, L, N)
    init_state = jnp.array([0.0, k0_init, 0.0, 0.0])
    sol = odeint(beam_ode, init_state, s_vals, jnp.array([x_m, y_m, psi]))
    theta_vals = sol[:, 0]
    return s_vals, theta_vals

# Jacobian computation
def theta_L_from_params(params):
    x_m, y_m, psi = params
    s_vals, theta_vals = solve_beam(x_m, y_m, psi)
    return theta_vals[-1]

grad_theta_L = jax.grad(theta_L_from_params)

# Evaluate
params = jnp.array([0.08, 0.03, jnp.arctan2(0.03, 0.08)])
theta_L = theta_L_from_params(params)
grad_vals = grad_theta_L(params)

print(theta_L, grad_vals)
