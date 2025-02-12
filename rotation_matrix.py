import numpy as np
from bfgs_minimise import alpha_star
from constants import d,h


def rotation_matrix_z(alpha):
    return np.array([
        [np.cos(alpha), np.sin(alpha), 0],
        [-np.sin(alpha), np.cos(alpha), 0],
        [0,0,1]
        ])

def compute_transformation_matrix(alpha_star, d, h):
    r_DM_E_to_O = rotation_matrix_z(-alpha_star)
    t_DM_E_to_O = np.array([[d], [h], [0]])  # Column vector (3x1)
    T_e_to_O = np.block([
        [r_DM_E_to_O, t_DM_E_to_O], 
        [np.array([[0, 0, 0, 1]])]  # Last row
    ])
    return T_e_to_O
# source_points = np.array([
#     [0, 0],
#     [0.43, 0],
#     [0.4312, 0.4]
# ])

# # Target points (after transformation)
# target_points = np.array([
#     [-0.31561346376380156, 0.4392848297605487],
#     [0.11561346376380156, 0.4392848297605487],
#     [0.11561346376380156, 0.8392848297605487]
# ])

# # Construct the matrix A
# A = []
# b = []

# for i in range(len(source_points)):
#     x, y = source_points[i]
#     x_p, y_p = target_points[i]
    
#     A.append([x, y, 1, 0, 0, 0])  # Row for x'
#     A.append([0, 0, 0, x, y, 1])  # Row for y'
    
#     b.append(x_p)
#     b.append(y_p)

# A = np.array(A)
# b = np.array(b)

# # Solve for transformation parameters
# params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

# # Extract transformation matrix elements
# a, b, tx, c, d, ty = params

# # Form the affine transformation matrix
# T = np.array([
#     [a, b, tx],
#     [c, d, ty],
#     [0, 0, 1]
# ])

# print("Affine Transformation Matrix:")
# print(T)
T = np.array([
    [ 1.00285332e+00, -3.00855996e-03, -3.15613464e-01],
    [6.66133815e-16,  1.00000000,  4.39284830e-01],
    [0.00000000,  0.00000000,  1.00000000]
    ])

def transform_point(T, x, y):
    
    transformed = np.dot(T, np.array([x, y, 1]))  # Homogeneous multiplication
    return transformed[:2]  # Extract only x' and y'

# # Example points to transform
# source_points = [
#     (0, 0),
#     (0.085, -0.025),
#     (0.4312, 0.4)
# ]

# # Apply transformation
# transformed_points = [transform_point(T, p) for p in source_points]

# print("Transformed Points:")
# for original, transformed in zip(source_points, transformed_points):
#     print(f"{original} -> {transformed}")

transformed_points = transform_point(T, d, h)
x_robotic_arm = transformed_points[0]
y_robotic_arm = transformed_points[1]

print("Robotic x coordinate: ", x_robotic_arm)
print("Robotic y coordinate: ", y_robotic_arm)
# T_e_to_O = compute_transformation_matrix(alpha_star, x_robotic_arm, y_robotic_arm)

# print("Homogeneous Transformation Matrix T_e_to_O:")
# print(T_e_to_O)