import numpy as np

def force_from_paper(r_vec, angle_deg):
    """Matches exactly the MATLAB version you posted"""
    
    # Magnet constants
    mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (T·m/A)
    Br = 1.2                # Remanence (T)
    r = 0.04                # EPM radius (m)
    h = 0.06                # EPM height (m)
    r_i = 0.0005            # IPM radius (m)
    h_i = 0.005             # IPM height (m)

    V_E = np.pi * r**2 * h
    V_I = np.pi * r_i**2 * h_i

    m_E_mag = (Br * V_E) / mu0
    m_I_mag = (Br * V_I) / mu0

    # Magnet vectors
    m_E = np.array([0, m_E_mag, 0])
    m_I = np.array([m_I_mag, 0, 0])

    # Rotation of m_E by angle
    theta = np.deg2rad(angle_deg)
    m_E = m_E_mag * np.array([np.sin(theta), np.cos(theta), 0])

    m_E_hat = m_E / np.linalg.norm(m_E)
    m_I_hat = m_I / np.linalg.norm(m_I)

    p = r_vec
    p_norm = np.linalg.norm(p)
    if p_norm == 0:
        return np.zeros(3), np.zeros(3)

    p_hat = p / p_norm

    Z = np.eye(3) - 5 * np.outer(p_hat, p_hat)
    D = np.eye(3) - 3 * np.outer(p_hat, p_hat)

    lambda_ = (mu0 / (4 * np.pi)) * np.linalg.norm(m_E) * np.linalg.norm(m_I) / (p_norm**3)

    # --- Force ---
    term1 = np.outer(m_E_hat, m_I_hat)
    term2 = np.outer(m_I_hat, m_E_hat)
    term3 = (m_I_hat @ Z @ m_I_hat) * np.eye(3)
    F_m = (3 * lambda_ / p_norm) * (term1 + term2 + term3) @ p

    # --- Torque ---
    T_m = np.cross(lambda_ * m_I_hat, D @ m_E_hat)

    # --- Optional: Also computing B field if needed ---
    D_em = 2 * r       # Diameter
    L_em = h           # Height
    z = p_norm         # Distance along z-axis

    term1_B = (z + L_em) / np.sqrt((D_em/2)**2 + (z + L_em)**2)
    term2_B = z / np.sqrt((D_em/2)**2 + z**2)

    B_z = (Br / 2) * (term1_B - term2_B)
    B_vec_eq15 = np.array([0, B_z, 0])

    return F_m, T_m
# import numpy as np
# import matplotlib.pyplot as plt

# r_vec = np.array([0.0, 0.2, 0.0])  # fixed r_vec
# angles = np.linspace(0, 90, 100)   # sweep from 90 down to 0 degrees

# Tz_vals = []

# for angle in angles:
#     F_m, T_m = force_from_paper(r_vec, angle)
#     Tz_vals.append(T_m[2])  # Save Z-torque

# # Plot
# plt.figure(figsize=(8,5))
# plt.plot(angles, np.abs(Tz_vals), label='|Torque_z|')
# plt.xlabel('Magnet Rotation (degrees)')
# plt.ylabel('Torque_z (Nm)')
# plt.title('Torque_z vs Magnet Rotation')
# plt.grid(True)
# plt.legend()
# plt.show()
