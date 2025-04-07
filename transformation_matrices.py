import numpy as np

def build_transform(R, P):
    """Builds a 4x4 homogeneous transformation matrix."""
    P = np.array(P).reshape(3, 1)
    T = np.block([
        [R, P],
        [np.zeros((1, 3)), np.array([[1]])]
    ])
    return T

def rot_z(alpha):
    """2D rotation around z-axis (in the x-y plane)."""
    return np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha),  np.cos(alpha), 0],
        [0,              0,             1]
    ])

def get_T_camera_to_robot(R_cr, P_cr):
    P = np.array(P_cr).reshape(3, 1)
    T = np.block([
        [R_cr, P],
        [np.zeros((1, 3)), np.array([[1]])]
    ])
    return T


def get_T_advancer_to_robot(R_ar, P_ar):
    P = np.array(P_ar).reshape(3, 1)
    T = np.block([
        [R_ar, P],
        [np.zeros((1, 3)), np.array([[1]])]
    ])
    return T

def get_T_catheter_base_to_advancer(alpha, l_t, v_star):
    """From screenshot: catheter base in advancer frame"""
    Rz_alpha = rot_z(alpha)
    v_star = np.array(v_star).reshape(3, 1)
    displacement = l_t * v_star
    return build_transform(Rz_alpha, displacement)

def get_T_catheter_tip_to_camera(R_tc, P_tc):
    """Catheter tip in camera frame (tracked dynamically)"""
    return build_transform(R_tc, P_tc)

def get_T_catheter_tip_to_robot(T_cr, T_tc):
    """Chain: tip in camera -> robot frame"""
    return T_cr @ T_tc

def get_T_catheter_base_to_robot(T_ar, T_ba):
    """Chain: base in advancer -> robot frame"""
    return T_ar @ T_ba

def get_magnet_position_in_robot(catheter_base_robot_pos, initial_offset):
    """Compute the new magnet position to maintain fixed offset from catheter base"""
    catheter_base_robot_pos = np.array(catheter_base_robot_pos).reshape(3)
    initial_offset = np.array(initial_offset).reshape(3)
    return catheter_base_robot_pos + initial_offset


def compute_velocity(N_p_t, r_w, delta):
    """Computes velocity over time from pulse values."""
    N_pr = 360 / delta
    v_t = (np.pi * r_w / (30 * N_pr)) * N_p_t
    return v_t

def compute_displacement(time, v_t):
    """Computes displacement l(t) by numerical integration of velocity."""
    return np.trapz(v_t, x=time)  # Integrate using trapezoidal rule

# Time and pulse inputs (example)
time = np.linspace(0, 10, 1000)  # 10 seconds sampled at 1000 points
N_p_t = np.sin(time) * 50 + 100  # example varying pulse pattern

# Constants
r_w = 0.01       # 1 cm radius wheel
delta = 1.8      # step angle in degrees

# Compute
v_t = compute_velocity(N_p_t, r_w, delta)
l_t = compute_displacement(time, v_t)
v_star = np.array([0,0,1])
print(f"Propulsion displacement l(t) = {l_t:.4f} meters")

# === Example values ===
# Fixed (known) frames
# Camera rotation: 180 degrees about Z axis
R_cr = np.array([
    [-1, 0, 0],
    [ 0,-1, 0],
    [ 0, 0, 1]
])

# Camera position in robot base frame
P_cr = [1, 1, 1]


R_ar = np.eye(3)  # Advancer in robot base frame
P_ar = [0.2, 0.0, 0.9]

# From screenshot: catheter base in advancer frame
alpha = np.deg2rad(30)  # advancer rotation

# Tip in camera frame (dynamic)
R_tc = np.eye(3)        # identity for simplicity
P_tc = [0.1, 0.0, 0.3]  # detected by vision

# Initial magnet offset (say 10 cm in -x)
initial_offset = [-0.1, 0.0, 0.0]

# === Transformation matrices ===
T_cr = get_T_camera_to_robot(R_cr, P_cr)
T_ar = get_T_advancer_to_robot(R_ar, P_ar)
T_ba = get_T_catheter_base_to_advancer(alpha, l_t, v_star)
T_tc = get_T_catheter_tip_to_camera(R_tc, P_tc)

# Global poses
T_tr = get_T_catheter_tip_to_robot(T_cr, T_tc)
T_br = get_T_catheter_base_to_robot(T_ar, T_ba)

# Extract positions
P_tr = T_tr[:3, 3]
P_br = T_br[:3, 3]

# Compute magnet position in robot frame
magnet_position = get_magnet_position_in_robot(P_br, initial_offset)

import pandas as pd

df = pd.DataFrame({
    "Label": ["Catheter Tip", "Catheter Base", "Magnet"],
    "X": [P_tr[0], P_br[0], magnet_position[0]],
    "Y": [P_tr[1], P_br[1], magnet_position[1]],
    "Z": [P_tr[2], P_br[2], magnet_position[2]]
})

print("\nObject Positions in Robot Frame:")
print(df)

