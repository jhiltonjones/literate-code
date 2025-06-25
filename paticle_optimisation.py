import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

L = 0.01          # length in meters
r = 0.00048       # radius in meters
D = 2 * r         # diameter in meters
Mp = 640e3       # magnetization in A/m 
E = 2.2e6         # Young's modulus in Pascals
nu = 0.5          # Poisson’s ratio 
# G0 = E / (2 * (1 + nu))  
G0 = 609990.96
mu_0 = 4 * np.pi * 1e-7  
M_a = 453.12500000000006     
psi = np.deg2rad(0)    
d = 0.152
R = 0.05  
B_re = 1.45  

p = np.array([0.2, 0, 0])  
p_norm = np.linalg.norm(p)
p_unit = p / p_norm

# Dipole orientation unit 
m_a_unit = np.array([np.cos(psi), np.sin(psi), 0])

def compute_magnetic_moment(diameter_mm, height_mm, grade='N52'):
    r = (diameter_mm / 1000) / 2  
    h = height_mm / 1000        

    volume = np.pi * r**2 * h 

    if grade.upper() == 'N52':
        M_s = 1.3e6  
    else:
        raise ValueError(f"Unsupported magnet grade: {grade}")

    M_a = volume * M_s  
    return M_a
M_a = compute_magnetic_moment(100, 50)
print(f"Magnetisation {M_a}, l/d = {L/D}")

def magnetic_field(p, m_a_unit, M_a, mu_0):
    p_norm = np.linalg.norm(p)
    p_hat = p / p_norm
    outer = 3 * np.outer(p_hat, p_hat) - np.eye(3)
    term1 = (mu_0 * M_a) / (4 * np.pi * p_norm**3)
    B = term1 * outer @ m_a_unit
    return B
def axial_field_cylinder(d, R=0.05, B_re=1.45):
    """
    Computes axial magnetic field along the centerline of a cylindrical magnet.
    
    Parameters:
    - d: distance from center of magnet along axis (in meters)
    - R: half-length of the magnet (e.g., 0.05 for 100 mm long magnet)
    - B_re: remanent surface field of the magnet in Tesla (e.g. 1.45 T)

    Returns:
    - B: magnetic field strength in Tesla at distance d
    """
    d_by_R = d / R
    term1 = (d_by_R + 1) / np.sqrt((d_by_R + 1)**2 + 1)
    term2 = (d_by_R - 1) / np.sqrt((d_by_R - 1)**2 + 1)
    return 0.5 * B_re * (term1 - term2)

psi_values = np.linspace(0, np.pi, 90) 
B_store = -np.inf
psi_best = 0

for psi_deg in psi_values:
    psi_rad = np.deg2rad(psi_deg)
    m_a_unit = np.array([np.cos(psi_rad), np.sin(psi_rad), 0])
    # B_3d = magnetic_field(p, m_a_unit, M_a, mu_0)
    # B = np.linalg.norm(B_3d)
    B = axial_field_cylinder(d, R, B_re)
    # print(f"Axial magnetic field at d = {d} m: B = {B*1000:.2f} mT")
    if B > B_store:
        B_store = B
        psi_best = psi_deg

print("Max B =", B_store, "T at psi =", psi_best, "rad")


def Mr(phi):
    return phi * Mp

def G(phi):
    if phi >= 1/1.35: 
        return np.inf
    return G0 * np.exp((2.5 * phi) / (1 - 1.35 * phi))

def theta_L(phi):
    mr = Mr(phi)
    g = G(phi)
    if g == np.inf or g == 0:
        return 0
    term1 = (mr * B) / g
    term2 = (L / D)**2
    return (8/3) * term1 * term2

result = minimize_scalar(lambda phi: -theta_L(phi), bounds=(0, 1), method='bounded')

optimal_phi = result.x
max_theta = theta_L(optimal_phi)

print(f"Optimal particle fraction phi = {optimal_phi:.4f}")
print(f"Maximum bending angle theta_L = {max_theta:.4f} rad")
print(f"Maximum bending angle theta_L = {np.rad2deg(max_theta):.4f} deg")

phis = np.linspace(0.001, 0.4, 300)
thetas = [theta_L(p) for p in phis]
plt.figure()
plt.plot(phis, np.rad2deg(thetas))
plt.xlabel("Particle fraction φ")
plt.ylabel("Bending angle θ_L")
plt.title("Bending angle vs. particle concentration")
plt.grid(True)
plt.show()
def energy_density(phi):
    mr = Mr(phi)
    g = G(phi)
    term1 = (mr**2 * B**2)/g
    term2 = (L/D)**2
    return (16/9)*term1*term2
result_2 = minimize_scalar(lambda phi: -energy_density(phi), bounds=(0,1.0), method='bounded')
optimal_phi_energy_density = result_2.x
max_energy_density = energy_density(optimal_phi_energy_density)
print(f"Optimal phi for the energy density {optimal_phi_energy_density}")
print(f"Maximum bending angle theta_L for optimal energy density particle = {max_energy_density:.4f} rad")
phis = np.linspace(0.001, 0.4, 300)
energies = [energy_density(p) for p in phis]
plt.figure()
plt.plot(phis, energies, color='darkgreen')
plt.xlabel("Particle fraction φ for energy density")
plt.ylabel(r"Energy Density $\mathcal{W}$ (J·m$^{-3}$)")
plt.title("Energy Density vs. Particle Concentration")
plt.grid(True)
plt.tight_layout()
plt.show()

phi_values = np.linspace(0.01, 0.4, 100)
M_over_G = [Mr(phi) / G(phi) for phi in phi_values]

plt.figure(figsize=(6, 4))
plt.plot(phi_values * 100, M_over_G, label="Analytical", color="red")
plt.axvline(x=20.7, color="black", linestyle="--", label=r"$\phi = 20.7\%$")
plt.xlabel("Particle volume fraction φ (%)")
plt.ylabel(r"$\tilde{M}_r / G$")
plt.title(r"Remanent Magnetization to Shear Modulus vs. $\phi$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

phi = 0.2
print(f"M_r at phi=0.2: {Mr(0.2)} A/m")
print(f"G(0.2) = {G(0.2):.2f} Pa")

def compute_force_stiffness_angle(phi,
                                  D_range,
                                  D_inner=0.5e-3,
                                  t_mag=0.5e-3,
                                  L=0.01,
                                  Mp=640e3,
                                  B_update=0.05,
                                  E_inner=2.5e6,    
                                  E_mag=2.2e6):
    forces = []
    stiffnesses = []
    angles = []

    for D_outer in D_range:
        D_mag_inner = D_outer - 2 * t_mag
        if D_mag_inner <= D_inner:
            forces.append(np.nan)
            stiffnesses.append(np.nan)
            angles.append(np.nan)
            continue

        Mr = phi * Mp
        A_mag = (np.pi / 4) * (D_outer**2 - D_mag_inner**2)
        F_eq = Mr * B_update * A_mag

        # Moments of inertia
        I_mag = (np.pi / 64) * (D_outer**4 - D_mag_inner**4)
        I_inner = (np.pi / 64) * (D_mag_inner**4 - D_inner**4)
        EI_total = E_mag * I_mag + E_inner * I_inner

        # Use force-based deflection angle formula
        theta_rad = F_eq * L**2 / (2 * EI_total)
        theta_deg = np.rad2deg(theta_rad)

        forces.append(F_eq)
        stiffnesses.append(EI_total)
        angles.append(theta_deg)

    return np.array(forces), np.array(stiffnesses), np.array(angles)


# --- Configuration ---
phi = 0.2
D_range = np.linspace(1e-3, 5e-3, 100)  # Outer diameters from 1 mm to 5 mm

# --- Compute ---
forces, stiffnesses, angles = compute_force_stiffness_angle(phi, D_range)

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(10,6))

# Equivalent Force plot
color1 = 'tab:red'
ax1.set_xlabel('Outer Diameter $D_{\\text{outer}}$ (m)')
ax1.set_ylabel('Equivalent Force $F_{\\mathrm{eq}}$ (N)', color=color1)
ax1.plot(D_range, forces, color=color1, label='Equivalent Force')
ax1.tick_params(axis='y', labelcolor=color1)

# EI plot
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Bending Stiffness $EI$ (N·m$^2$)', color=color2)
ax2.plot(D_range, stiffnesses, color=color2, label='Bending Stiffness')
ax2.tick_params(axis='y', labelcolor=color2)

# Bending angle plot on third axis
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))  # Offset to avoid overlap
color3 = 'tab:green'
ax3.set_ylabel('Bending Angle $\\theta$ (degrees)', color=color3)
ax3.plot(D_range, angles, color=color3, linestyle='--', label='Bending Angle')
ax3.tick_params(axis='y', labelcolor=color3)

# Labels and layout
plt.title("Equivalent Force, Bending Stiffness, and Bending Angle vs Outer Diameter")
fig.tight_layout()
plt.grid(True)
plt.show()




