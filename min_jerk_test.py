import numpy as np
import time
import matplotlib.pyplot as plt

class PathPlanTranslation(object):
    def __init__(self, pose_init, pose_desired, total_time):
        """
        Initialize trajectory planning for both position (X, Y, Z) and rotation (Rx, Ry, Rz).
        """
        self.position_init = pose_init[:3]
        self.position_des = pose_desired[:3]
        self.rotation_init = pose_init[3:]
        self.rotation_des = pose_desired[3:]

        self.tfinal = total_time

    def trajectory_planning(self, t):
        """
        Compute minimum jerk trajectory for position (X, Y, Z) and rotation (Rx, Ry, Rz).
        """
        # Extract initial and final positions
        X_init, Y_init, Z_init = self.position_init
        X_final, Y_final, Z_final = self.position_des

        # Extract initial and final rotations
        Rx_init, Ry_init, Rz_init = self.rotation_init
        Rx_final, Ry_final, Rz_final = self.rotation_des

        # Compute minimum jerk interpolation
        def min_jerk_trajectory(t, tf, init, final):
            return (final - init) / (tf ** 3) * (
                    6 * (t ** 5) / (tf ** 2) - 15 * (t ** 4) / tf + 10 * (t ** 3)) + init

        def min_jerk_velocity(t, tf, init, final):
            return (final - init) / (tf ** 3) * (
                    30 * (t ** 4) / (tf ** 2) - 60 * (t ** 3) / tf + 30 * (t ** 2))

        def min_jerk_acceleration(t, tf, init, final):
            return (final - init) / (tf ** 3) * (
                    120 * (t ** 3) / (tf ** 2) - 180 * (t ** 2) / tf + 60 * t)

        # Compute trajectory for position
        x_traj = min_jerk_trajectory(t, self.tfinal, X_init, X_final)
        y_traj = min_jerk_trajectory(t, self.tfinal, Y_init, Y_final)
        z_traj = min_jerk_trajectory(t, self.tfinal, Z_init, Z_final)
        position = np.array([x_traj, y_traj, z_traj])

        # Compute trajectory for rotation
        rx_traj = min_jerk_trajectory(t, self.tfinal, Rx_init, Rx_final)
        ry_traj = min_jerk_trajectory(t, self.tfinal, Ry_init, Ry_final)
        rz_traj = min_jerk_trajectory(t, self.tfinal, Rz_init, Rz_final)
        rotation = np.array([rx_traj, ry_traj, rz_traj])

        # Compute velocity for position
        vx = min_jerk_velocity(t, self.tfinal, X_init, X_final)
        vy = min_jerk_velocity(t, self.tfinal, Y_init, Y_final)
        vz = min_jerk_velocity(t, self.tfinal, Z_init, Z_final)
        velocity = np.array([vx, vy, vz])

        # Compute velocity for rotation
        vrx = min_jerk_velocity(t, self.tfinal, Rx_init, Rx_final)
        vry = min_jerk_velocity(t, self.tfinal, Ry_init, Ry_final)
        vrz = min_jerk_velocity(t, self.tfinal, Rz_init, Rz_final)
        rot_velocity = np.array([vrx, vry, vrz])

        # Compute acceleration for position
        ax = min_jerk_acceleration(t, self.tfinal, X_init, X_final)
        ay = min_jerk_acceleration(t, self.tfinal, Y_init, Y_final)
        az = min_jerk_acceleration(t, self.tfinal, Z_init, Z_final)
        acceleration = np.array([ax, ay, az])

        # Compute acceleration for rotation
        arx = min_jerk_acceleration(t, self.tfinal, Rx_init, Rx_final)
        ary = min_jerk_acceleration(t, self.tfinal, Ry_init, Ry_final)
        arz = min_jerk_acceleration(t, self.tfinal, Rz_init, Rz_final)
        rot_acceleration = np.array([arx, ary, arz])

        return [position, rotation, velocity, rot_velocity, acceleration, rot_acceleration]

