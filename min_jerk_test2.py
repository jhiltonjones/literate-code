import numpy as np

class PathPlanTranslation(object):
    def __init__(self, pose_init, pose_desired, total_time):
        
        self.position_init = pose_init[:3]
        self.position_des = pose_desired[:3]

        self.tfinal = total_time

    def trajectory_planning(self, t):
        X_init = self.position_init[0]
        Y_init = self.position_init[1]
        Z_init = self.position_init[2]

        X_final = self.position_des[0]
        Y_final = self.position_des[1]
        Z_final = self.position_des[2]

        # position
        x_traj = (X_final - X_init) / (self.tfinal ** 3) * (
                6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + X_init
        y_traj = (Y_final - Y_init) / (self.tfinal ** 3) * (
                6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + Y_init
        z_traj = (Z_final - Z_init) / (self.tfinal ** 3) * (
                6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + Z_init
        position = np.array([x_traj, y_traj, z_traj])

        # velocities
        vx = (X_final - X_init) / (self.tfinal ** 3) * (
                30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        vy = (Y_final - Y_init) / (self.tfinal ** 3) * (
                30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        vz = (Z_final - Z_init) / (self.tfinal ** 3) * (
                30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        velocity = np.array([vx, vy, vz])

        # acceleration
        ax = (X_final - X_init) / (self.tfinal ** 3) * (
                120 * (t ** 3) / (self.tfinal ** 2) - 180 * (t ** 2) / self.tfinal + 60 * t)
        ay = (Y_final - Y_init) / (self.tfinal ** 3) * (
                120 * (t ** 3) / (self.tfinal ** 2) - 180 * (t ** 2) / self.tfinal + 60 * t)
        az = (Z_final - Z_init) / (self.tfinal ** 3) * (
                120 * (t ** 3) / (self.tfinal ** 2) - 180 * (t ** 2) / self.tfinal + 60 * t)
        acceleration = np.array([ax, ay, az])

        return [position, velocity, acceleration]