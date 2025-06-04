import numpy as np

# Physical constants and parameters
mu0 = 4*np.pi*1e-7  # vacuum permeability (H/m or N/A^2)
EI = 1.0e-3         # Bending stiffness (EI) of the rod [N·m^2] – set appropriately
L = 0.3             # Total rod length [m]
# Magnet segment parameters
magnet_length = 0.05                      # Length of rod that contains the permanent magnet [m]
magnet_start = L - magnet_length          # Arc-length position where the magnet segment begins
magnetization = 4e5                      # Magnet’s magnetization [A/m] (along its length)
cross_area = 3.14e-6                     # Cross-sectional area of rod [m^2] (example value)
m_per_length = magnetization * cross_area  # Dipole moment per unit length [A·m^2/m] in magnet segment

# External magnet parameters
ext_position = np.array([0.1, 0.15, 0.0])   # Position of external magnet [m]
ext_dipole = np.array([0.0, -1.0, 0.0])    # External magnet’s dipole orientation (unit vector pointing South→North)
ext_magnitude = 1.0                       # External magnet’s dipole moment magnitude [A·m^2]
ext_moment = ext_magnitude * ext_dipole   # Dipole moment vector of external magnet [A·m^2]
# (Ensure ext_dipole is chosen such that this dipole attracts the rod’s magnet. 
# For example, if the rod’s magnet points +x when straight, ext_dipole should point -x toward the rod’s magnet to attract.)

def dipole_field(point):
    """Compute magnetic field B (as a numpy 3-vector) at a given point due to the external magnet's dipole."""
    r_vec = point - ext_position
    r = np.linalg.norm(r_vec)
    if r < 1e-12:
        return np.zeros(3)
    # Dipole field formula: B = (mu0/4πr^3) * [3(m_ext·\hat{r}) \hat{r} - m_ext]
    r_hat = r_vec / r
    m_dot_r = np.dot(ext_moment, r_hat)
    B = (mu0 / (4*np.pi * r**3)) * (3 * m_dot_r * r_hat - ext_moment)
    return B

def dipole_field_gradient(point):
    """Compute the gradient of the external dipole field at a point. 
    Returns a 3x3 matrix of ∂B_i/∂x_j (in Cartesian coords). Uses analytic partial derivatives for efficiency."""
    # For simplicity, assume external dipole moment ext_moment = [mx, my, mz] and we'll use the derived formula for partials.
    r_vec = point - ext_position
    x, y, z = r_vec  # components of r_vec
    r2 = x*x + y*y + z*z
    r = np.sqrt(r2)
    if r < 1e-12:
        return np.zeros((3, 3))
    mx, my, mz = ext_moment
    C = mu0 / (4*np.pi)
    # Common factors for gradient (to avoid repetition)
    r5 = r**5
    r7 = r**7
    m_dot_r = mx*x + my*y + mz*z
    # Compute partial derivatives (analytical expressions for dipole field gradient)
    # ∂B_x/∂x
    dBx_dx = C * (5*x * (mx * r2 - 3 * x * m_dot_r) + r2 * (4 * mx * x + 3 * my * y + 3 * mz * z)) / r7
    # ∂B_x/∂y
    dBx_dy = C * (5*y * (mx * r2 - 3 * x * m_dot_r) + r2 * ( -2 * mx * y + 3 * my * x           )) / r7
    # ∂B_x/∂z
    dBx_dz = C * (5*z * (mx * r2 - 3 * x * m_dot_r) + r2 * ( -2 * mx * z + 3 * mz * x           )) / r7
    # ∂B_y/∂x
    dBy_dx = C * (5*x * (my * r2 - 3 * y * m_dot_r) + r2 * ( 3 * mx * y            - 2 * my * x )) / r7
    # ∂B_y/∂y
    dBy_dy = C * (5*y * (my * r2 - 3 * y * m_dot_r) + r2 * ( 3 * mx * x + 4 * my * y + 3 * mz * z)) / r7
    # ∂B_y/∂z
    dBy_dz = C * (5*z * (my * r2 - 3 * y * m_dot_r) + r2 * (           - 2 * my * z + 3 * mz * y)) / r7
    # ∂B_z/∂x
    dBz_dx = C * (5*x * (mz * r2 - 3 * z * m_dot_r) + r2 * ( 3 * mx * z            - 2 * mz * x )) / r7
    # ∂B_z/∂y
    dBz_dy = C * (5*y * (mz * r2 - 3 * z * m_dot_r) + r2 * ( 3 * my * z            - 2 * mz * y )) / r7
    # ∂B_z/∂z
    dBz_dz = C * (5*z * (mz * r2 - 3 * z * m_dot_r) + r2 * ( 3 * mx * x + 3 * my * y + 4 * mz * z)) / r7
    # Assemble into a 3x3 Jacobian matrix
    gradB = np.array([[dBx_dx, dBx_dy, dBx_dz],
                      [dBy_dx, dBy_dy, dBy_dz],
                      [dBz_dx, dBz_dy, dBz_dz]])
    return gradB

def rod_odes(s, Y):
    """
    ODE system for the rod:
    Y = [x, y, theta, Nx, Ny, M_int] at arc-length s.
    Returns dY/ds.
    """
    x, y, theta, Nx, Ny, M_int = Y
    # Unit tangent vector (cosθ, sinθ) since the rod is inextensible
    tx = np.cos(theta)
    ty = np.sin(theta)
    # Intrinsic curvature relation: dθ/ds = M_int/EI  (since M = EI * curvature)
    dtheta_ds = M_int / EI

    # Initialize external force/torque per length
    f_ext = np.array([0.0, 0.0, 0.0])   # [f_x, f_y, f_z] per unit length
    c_ext = 0.0                        # external torque (about z-axis) per unit length

    # If within the magnetized segment, compute magnetic forces/torques
    if s >= magnet_start:
        # Permanent magnet's dipole (per unit length) aligned with local tangent
        m_vec = m_per_length * np.array([tx, ty, 0.0])  # magnetization vector (A·m^2/m) in global coords at this segment
        # External magnetic field at this point
        point = np.array([x, y, 0.0])
        B = dipole_field(point)
        # Magnetic torque per unit length: m_vec × B  (extrinsic torque density)
        T_ext = np.cross(m_vec, B)
        c_ext = T_ext[2]  # z-component of torque (out-of-plane)
        # Magnetic force per unit length: f_ext = (m_vec · ∇)B  (using field gradient)
        gradB = dipole_field_gradient(point)
        # m_vec · ∇B gives a vector whose components: (m·∇)B)_i = Σ_j m_j ∂B_i/∂x_j
        # Compute that via matrix multiplication of m_vec (1x3) with gradB (3x3)
        f_ext = m_vec.dot(gradB)  # this yields [f_x, f_y, f_z]

    # Elastic (intrinsic) forces and moments:
    # Force equilibrium: dN/ds + f_ext = 0  => dN/ds = -f_ext  (N = [Nx, Ny, 0])
    dNx_ds = -f_ext[0]
    dNy_ds = -f_ext[1]
    # Moment equilibrium: dM_int/ds + (t × N)_z + c_ext = 0  => dM_int/ds = -(t_x * Ny - t_y * Nx) - c_ext
    # (t × N)_z = t_x * N_y - t_y * N_x is the internal torsion from shear forces
    p_cross_N = tx * Ny - ty * Nx
    dM_int_ds = - (p_cross_N + c_ext)

    # Position kinematics: dp/ds = t (tangent vector)
    dx_ds = tx
    dy_ds = ty

    return np.array([dx_ds, dy_ds, dtheta_ds, dNx_ds, dNy_ds, dM_int_ds])

# Shooting method setup: solve two-point boundary value problem
# Boundary conditions:
#   at s=0 (base): x(0)=0, y(0)=0, θ(0) = θ_base (assumed 0 here), 
#                 unknown Nx(0), Ny(0), M_int(0) (reaction forces/moment at base)
#   at s=L (tip): free end ⇒ internal force = 0 (Nx(L)=0, Ny(L)=0) and bending moment = 0 (M_int(L)=0).
# We will use a Newton shooting method: guess initial [Nx0, Ny0, M0], integrate, and correct guess via Jacobian.

# Boundary conditions at base (clamped position and direction)
theta_base = 0.0
base_state = np.array([0.0, 0.0, theta_base])  # we know initial x, y, theta
# Initial guess for unknown base forces/moment
guess = np.array([0.0, 0.0, 0.0])  # [Nx0, Ny0, M0] initial guess

# Function to integrate the ODE given a guess for [Nx0, Ny0, M0]
def integrate_rod(Nx0, Ny0, M0):
    Y0 = np.concatenate((base_state, [Nx0, Ny0, M0]))
    # Use a simple integrator (e.g., explicit Euler or small-step RK4) to integrate from s=0 to s=L
    n_steps = 1000
    ds = L / n_steps
    Y = Y0.copy()
    for i in range(n_steps):
        s = i * ds
        k1 = rod_odes(s, Y)
        k2 = rod_odes(s + 0.5*ds, Y + 0.5*ds*k1)
        k3 = rod_odes(s + 0.5*ds, Y + 0.5*ds*k2)
        k4 = rod_odes(s + ds, Y + ds*k3)
        Y += (ds/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return Y  # returns state at s=L

# Shooting residual function: returns the error in tip boundary conditions for a given guess
def tip_residual(guess):
    Nx0, Ny0, M0 = guess
    Y_end = integrate_rod(Nx0, Ny0, M0)
    # Tip internal forces and moment (desired to be zero)
    Nx_L, Ny_L, M_L = Y_end[3], Y_end[4], Y_end[5]
    return np.array([Nx_L, Ny_L, M_L])

# Newton-Raphson iterations for shooting
max_iters = 20
tol = 1e-6
for it in range(max_iters):
    res = tip_residual(guess)
    if np.linalg.norm(res) < tol:
        break  # convergence achieved
    # Compute Jacobian matrix of residuals w.r.t. guess using sensitivity equations (integrate derivatives)
    # We integrate the sensitivity ODEs alongside the main ODE to get ∂(Nx(L),Ny(L),M(L)) / ∂(Nx0,Ny0,M0).
    # Setup combined state [Y, ∂Y/∂Nx0, ∂Y/∂Ny0, ∂Y/∂M0] and integrate.
    # For brevity, we compute the Jacobian via finite differences here (analytical sensitivity integration is similar but omitted for clarity).
    J = np.zeros((3, 3))
    delta = 1e-6
    for j in range(3):
        pert = np.zeros(3); pert[j] = delta
        res_plus = tip_residual(guess + pert)
        res_minus = tip_residual(guess - pert)
        J[:, j] = (res_plus - res_minus) / (2 * delta)
    # Solve for correction: J * delta_guess = -res
    try:
        delta_guess = np.linalg.solve(J, -res)
    except np.linalg.LinAlgError:
        delta_guess = np.linalg.lstsq(J, -res, rcond=None)[0]
    guess += delta_guess

# At this point, `guess` contains the converged [Nx0, Ny0, M0] that satisfy the boundary conditions.
Nx0, Ny0, M0 = guess
print("Converged base reactions:", Nx0, Ny0, M0)
# Integrate one more time with the converged initial conditions to obtain the final rod shape
Y0 = np.concatenate((base_state, [Nx0, Ny0, M0]))
s_values = np.linspace(0, L, 201)
solution = []  # will store (x,y) positions
Y = Y0.copy()
for i in range(len(s_values)-1):
    s = s_values[i]
    ds = s_values[i+1] - s
    # Integrate one step (RK4)
    k1 = rod_odes(s, Y)
    k2 = rod_odes(s + 0.5*ds, Y + 0.5*ds*k1)
    k3 = rod_odes(s + 0.5*ds, Y + 0.5*ds*k2)
    k4 = rod_odes(s + ds, Y + ds*k3)
    Y += (ds/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    solution.append((Y[0], Y[1]))
# `solution` now contains the (x, y) coordinates of the rod along its length.
import matplotlib.pyplot as plt

# Convert solution to NumPy array for plotting
solution = np.array(solution)
x_vals = solution[:, 0]
y_vals = solution[:, 1]

# Plotting the rod's centerline
plt.figure(figsize=(6, 6))
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='Rod Shape')

# Plot the external magnet position
plt.plot(ext_position[0], ext_position[1], 'ro', label='External Magnet')

# Optionally, show the direction of the external magnet's dipole
arrow_scale = 0.02
plt.arrow(
    ext_position[0], ext_position[1],
    ext_dipole[0]*arrow_scale, ext_dipole[1]*arrow_scale,
    color='red', head_width=0.005, length_includes_head=True, label='Magnet Dipole'
)

# Final plot settings
plt.axis('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Bending of Magnetic Catheter Toward External Magnet')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
