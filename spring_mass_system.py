import numpy as np
import math
from scrap import R_epm, r_epm
# --- Existing helper: rotation vector -> rotation matrix (Rodrigues) ---
def rotvec_to_rotmat(r):
    r = np.array(r, dtype=float)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([
        [0,     -k[2],  k[1]],
        [k[2],  0,     -k[0]],
        [-k[1], k[0],  0    ]
    ])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R

# --- Your function: rotation matrix -> rotation vector (axis * angle) ---
def rot_to_axis_angle(R):
    eps = 1e-12
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = max(min(c, 1.0), -1.0)  # clamp for numerical stability
    theta = math.acos(c)
    if theta < 1e-12:
        return (0.0, 0.0, 0.0)
    rx = (R[2,1] - R[1,2]) / (2*math.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*math.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*math.sin(theta))
    return (theta*rx, theta*ry, theta*rz)  # rotation vector

def apply_z_rotation_deg_to_tcp(tcp_pose, delta_deg):
    """
    tcp_pose: [x, y, z, rx, ry, rz]  (UR-style axis-angle)
    delta_deg: rotation about *base* Z axis in degrees (right-hand rule)

    Returns a new TCP pose [x, y, z, rx', ry', rz'].
    """
    tcp_pose = list(tcp_pose)
    pos = np.array(tcp_pose[0:3], dtype=float)
    r   = np.array(tcp_pose[3:6], dtype=float)

    # current orientation as matrix
    R_current = rotvec_to_rotmat(r)

    # rotation about base Z axis
    theta = math.radians(delta_deg)
    c, s = math.cos(theta), math.sin(theta)
    Rz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

    # New orientation: first rotate around base Z, then apply old orientation
    R_new = Rz @ R_current

    # back to rotation vector
    r_new = np.array(rot_to_axis_angle(R_new))

    return np.concatenate([pos, r_new])

# -----------------------------------------------------------------
# Example with your pose:
tcp = [
0.8736501521917931, -0.3042779813358749, 0.45463250549805256, -0.3185948047669596, -3.104688675250433, -0.027665116580292432]
tcp[2] = tcp[2] + r_epm[2]
print(tcp)
tcp_new = apply_z_rotation_deg_to_tcp(tcp, 30.0)

print("Original TCP:", tcp)
print("New TCP (+30° about base Z):", tcp_new.tolist())
import numpy as np

theta = np.deg2rad(10.0)
c, s = np.cos(theta), np.sin(theta)
# 3D Rotation Matrix around the X-axis
R_x = np.array([
    [1.0, 0.0, 0.0],
    [0.0,   c,  -s],
    [0.0,   s,   c]
])

print(f"Rotation matrix R_x for {np.rad2deg(theta)} degrees:")
print(R_x)
R_epm2 = np.array([
    [c, -s, 0.0],
    [s,  c, 0.0],
    [0.0, 0.0, 1.0]
])
R_epm =np.array([[-0.17367001,  0.98475939, -0.00936341],
       [-0.9848039 , -0.17366216,  0.00165124],
       [-0.        ,  0.00950789,  0.9999548 ]])

R_y = np.array([
    [ c, 0.0,  s],
    [0.0, 1.0, 0.0],
    [-s, 0.0,  c]
])

print(f"Rotation matrix R_y for {np.rad2deg(theta)} degrees:")
print(R_y)
print(repr(R_epm))
tcp_pose = list(tcp)
pos = np.array(tcp_pose[0:3], dtype=float)
r   = np.array(tcp_pose[3:6], dtype=float)
R_current = rotvec_to_rotmat(r)
R_new = R_epm @ R_current
rx, ry, rz = rot_to_axis_angle(R_new)  # radians

x, y, z = tcp[:3]   # wherever you want the centre to be, e.g. current TCP

TCP_TARGET = [float(x), float(y), float(z),
              float(rx), float(ry), float(rz)]
# TCP_TARGET = np.array([0.48235856082065764, -0.3997407211794445, 0.6899791787635461, 2.50579693299435, -1.8239460762908732, 0.013319709198673793])
print(TCP_TARGET)

import numpy as np
import math

# --- helpers: rotvec <-> rotmat ---

def rotvec_to_rotmat(r):
    r = np.array(r, dtype=float)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([
        [0,     -k[2],  k[1]],
        [k[2],  0,     -k[0]],
        [-k[1], k[0],  0    ]
    ])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R

def rot_to_axis_angle(R):
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = max(min(c, 1.0), -1.0)  # clamp
    theta = math.acos(c)
    if theta < 1e-12:
        return (0.0, 0.0, 0.0)
    rx = (R[2,1] - R[1,2]) / (2*math.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*math.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*math.sin(theta))
    return (theta*rx, theta*ry, theta*rz)

# --- generic: rotate pose about pivot around world X/Y/Z ---

def rotate_pose_around_pivot_axis(pivot_pose, mag_pose, delta_deg, axis):
    """
    Rotate mag_pose [x,y,z,rx,ry,rz] by delta_deg about world axis 'x','y','z',
    around the point given by pivot_pose[:3].

    Both poses are in base/world frame (UR-style axis-angle).
    Right-hand rule about the chosen world axis.
    """
    pivot_pose = np.asarray(pivot_pose, float)
    mag_pose   = np.asarray(mag_pose, float)

    pivot = pivot_pose[:3]          # rotation centre
    pos   = mag_pose[:3]
    r     = mag_pose[3:6]

    # current orientation of magnet
    R_current = rotvec_to_rotmat(r)

    # rotation in world frame
    theta = math.radians(delta_deg)
    c, s = math.cos(theta), math.sin(theta)

    if axis.lower() == 'x':
        Raxis = np.array([
            [1.0, 0.0, 0.0],
            [0.0,   c, -s ],
            [0.0,   s,  c ]
        ])
    elif axis.lower() == 'y':
        Raxis = np.array([
            [  c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [ -s, 0.0,  c]
        ])
    elif axis.lower() == 'z':
        Raxis = np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    # new position: rigid rotation about pivot
    pos_rel = pos - pivot
    pos_new = pivot + Raxis @ pos_rel

    # new orientation: same world-axis rotation applied to the body frame
    R_new = Raxis @ R_current
    r_new = np.array(rot_to_axis_angle(R_new))

    return np.concatenate([pos_new, r_new])

# Convenience wrappers:

def rotate_pose_around_pivot_x(pivot_pose, mag_pose, delta_deg):
    return rotate_pose_around_pivot_axis(pivot_pose, mag_pose, delta_deg, 'x')

def rotate_pose_around_pivot_y(pivot_pose, mag_pose, delta_deg):
    return rotate_pose_around_pivot_axis(pivot_pose, mag_pose, delta_deg, 'y')

def rotate_pose_around_pivot_z(pivot_pose, mag_pose, delta_deg):
    return rotate_pose_around_pivot_axis(pivot_pose, mag_pose, delta_deg, 'z')

# ---------------------------------------------------------
# Your concrete numbers
pivot_pose = [0.7852216595743369, -0.47790432721335047, 0.42124824139667186, 0.3073610731210967, -3.1127062692872807, 0.03644269852131684]
mag_pose_start = [0.7742331615605501, -0.6021067715048376, 0.44016593152881767, 3.0871855897559737, -0.3264513836340089, 0.06731693031815834]

mag_pose_rotated_x = rotate_pose_around_pivot_z(pivot_pose, mag_pose_start, -90.0)
print("New magnet pose:", mag_pose_rotated_x.tolist())
import numpy as np
import math

# --- helpers: rotvec <-> rotmat ---

def rotvec_to_rotmat(r):
    r = np.array(r, dtype=float)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([
        [0,     -k[2],  k[1]],
        [k[2],  0,     -k[0]],
        [-k[1], k[0],  0    ]
    ])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R

def rot_to_axis_angle(R):
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = max(min(c, 1.0), -1.0)  # clamp
    theta = math.acos(c)
    if theta < 1e-12:
        return (0.0, 0.0, 0.0)
    rx = (R[2,1] - R[1,2]) / (2*math.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*math.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*math.sin(theta))
    return (theta*rx, theta*ry, theta*rz)

# --- combined Z then X rotation about a pivot in world frame ---

def rotate_pose_around_pivot_z_then_x(pivot_pose, mag_pose,
                                      delta_z_deg, delta_x_deg):
    """
    Rotate mag_pose [x,y,z,rx,ry,rz] first by delta_z_deg about world Z,
    then by delta_x_deg about world X, both around pivot_pose[:3].

    All poses are in the base/world frame (UR-style axis-angle).
    Right-hand rule about each world axis.
    """
    pivot_pose = np.asarray(pivot_pose, float)
    mag_pose   = np.asarray(mag_pose, float)

    pivot = pivot_pose[:3]          # rotation centre
    pos   = mag_pose[:3]
    r     = mag_pose[3:6]

    # current orientation of magnet
    R_current = rotvec_to_rotmat(r)

    # Z rotation (world)
    tz = math.radians(delta_z_deg)
    cz, sz = math.cos(tz), math.sin(tz)
    Rz = np.array([
        [ cz, -sz, 0.0],
        [ sz,  cz, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # X rotation (world)
    tx = math.radians(delta_x_deg)
    cx, sx = math.cos(tx), math.sin(tx)
    Rx = np.array([
        [1.0, 0.0,  0.0],
        [0.0,  cx, -sx],
        [0.0,  sx,  cx]
    ])

    # Total rotation: first Z, then X  →  R_total = Rx * Rz
    R_total = Rx @ Rz

    # new position: rigid rotation about pivot
    pos_rel = pos - pivot
    pos_new = pivot + R_total @ pos_rel

    # new orientation: same world rotations applied to the body frame
    R_new = R_total @ R_current
    r_new = np.array(rot_to_axis_angle(R_new))

    return np.concatenate([pos_new, r_new])



mag_pose_rotated = rotate_pose_around_pivot_z_then_x(
    pivot_pose, mag_pose_start,
    delta_z_deg=0,
    delta_x_deg=0.0
)

print("New pose after Z then X:", mag_pose_rotated.tolist())
import numpy as np

R_epm6 = np.array([[ 9.98871398e-01, -4.74962284e-02, -1.94000854e-04],
       [-4.74966246e-02, -9.98863066e-01, -4.07990896e-03],
       [ 0.00000000e+00,  4.08451875e-03, -9.99991658e-01]])
tz = math.radians(0)
cz, sz = math.cos(tz), math.sin(tz)
Rz = np.array([
    [ cz, -sz, 0.0],
    [ sz,  cz, 0.0],
    [0.0, 0.0, 1.0]
])
import math

def rotvec_to_rotmat(r):
    r = np.asarray(r, float)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([
        [0,     -k[2],  k[1]],
        [k[2],  0,     -k[0]],
        [-k[1], k[0],  0    ]
    ])
    return np.eye(3) + math.sin(theta)*K + (1 - math.cos(theta))*(K @ K)

def rot_to_axis_angle(R):
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = max(min(c, 1.0), -1.0)
    theta = math.acos(c)
    if theta < 1e-12:
        return (0.0, 0.0, 0.0)
    rx = (R[2,1] - R[1,2]) / (2*math.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*math.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*math.sin(theta))
    return (theta*rx, theta*ry, theta*rz)

def apply_R_around_pivot_to_pose(pose, pivot, R_inc):
    pose  = np.asarray(pose, float)
    pivot = np.asarray(pivot, float)
    pos   = pose[:3]
    rvec  = pose[3:6]

    R_current = rotvec_to_rotmat(rvec)

    # new orientation & position
    R_new = R_inc @ R_current
    pos_new = pivot + R_inc @ (pos - pivot)

    rvec_new = np.array(rot_to_axis_angle(R_new))
    return np.concatenate([pos_new, rvec_new])


p_rot = apply_R_around_pivot_to_pose(mag_pose_start, pivot_pose[:3], R_epm6)
print("Rotated position:", p_rot)
