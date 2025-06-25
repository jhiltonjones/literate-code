import numpy as np
import matplotlib.pyplot as plt

# Constants
mu_0 = 4 * np.pi * 1e-7  # H/m
B_re = 1  # Tesla
R = 0.025  # meters, 
# M_a = 342.86  # A·m² (from dipole moment)

def magnetic_field(p, m_a_unit, M_a, mu_0):
    p_norm = np.linalg.norm(p)
    p_hat = p / p_norm
    outer = 3 * np.outer(p_hat, p_hat) - np.eye(3)
    term1 = (mu_0 * M_a) / (4 * np.pi * p_norm**3)
    B = term1 * outer @ m_a_unit
    return B

# Cylindrical axial field model
def axial_field_cylinder(d, R=0.05):
    d_by_R = d / R
    term1 = (d_by_R + 1) / np.sqrt((d_by_R + 1)**2 + 1)
    term2 = (d_by_R - 1) / np.sqrt((d_by_R - 1)**2 + 1)
    return 0.5 * B_re * (term1 - term2)
def vol_cyclinder(r,h):
    return np.pi * r**2 *h
def m_a_calc():
    volume = vol_cyclinder(0.04, 0.09)
    print(f"Volume of the magnet is; {volume}")
    return (B_re / mu_0)*volume
M_a = m_a_calc()
print(f"M_a is {M_a}")
distances = np.linspace(0.05, 0.3, 200)  
m_a_unit = np.array([1, 0, 0])  
p = np.array([0.15,0,0])
B_dipole = magnetic_field(p, m_a_unit, M_a, mu_0)
B_cylinder = axial_field_cylinder(p[0], R)

print(f"The magnetic field at {p[0]} is {B_dipole}  with the dipole equation")
print(f"The magnetic field at {p[0]} is {B_cylinder}  with the cylinder equation")
dipole_fields = []
cylinder_fields = []

for d in distances:
    p = np.array([d, 0, 0])
    B_dipole = magnetic_field(p, m_a_unit, M_a, mu_0)
    B_cylinder = axial_field_cylinder(d, R)
    
    dipole_fields.append(np.linalg.norm(B_dipole))
    cylinder_fields.append(B_cylinder)

plt.figure(figsize=(8, 5))
plt.plot(distances * 1000, np.array(dipole_fields) * 1000, label="Magnetic Field Approximation", lw=2)
# plt.plot(distances * 1000, np.array(cylinder_fields) * 1000, label="Axial Cylinder Field", lw=2, linestyle="--")
plt.xlabel("Distance from Magnet Center (mm)")
plt.ylabel("Magnetic Field Magnitude (mT)")
plt.title("Magnetic Field Aproximation ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
