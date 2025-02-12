import numpy as np

class PathPlanTranslation(object):
    def __init__(self, pose_init, pose_desired, total_time, arc_height=0.1):
        """
        Generates an arc trajectory between pose_init and pose_desired.
        
        :param pose_init: [x, y, z] initial position
        :param pose_desired: [x, y, z] target position
        :param total_time: Time duration for the trajectory
        :param arc_height: Height of the arc above the direct path (optional)
        """
        self.position_init = np.array(pose_init[:3])
        self.position_des = np.array(pose_desired[:3])
        self.tfinal = total_time
        
        # Compute a midpoint with added height for an arc
        self.midpoint = (self.position_init + self.position_des) / 2
        self.midpoint[1] += arc_height  # Increase height for an arced trajectory

    def bezier_curve(self, t):
        """
        Quadratic Bézier curve: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
        """
        P0, P1, P2 = self.position_init, self.midpoint, self.position_des
        position = (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2
        return position

    def bezier_velocity(self, t):
        """
        First derivative of the Bézier curve to get velocity.
        """
        P0, P1, P2 = self.position_init, self.midpoint, self.position_des
        velocity = 2 * (1 - t) * (P1 - P0) + 2 * t * (P2 - P1)
        return velocity

    def bezier_acceleration(self, t):
        """
        Second derivative of the Bézier curve to get acceleration.
        """
        P0, P1, P2 = self.position_init, self.midpoint, self.position_des
        acceleration = 2 * (P2 - 2 * P1 + P0)
        return acceleration

    def trajectory_planning(self, t):
        """
        Compute smooth arc trajectory at time t.
        """
        t_normalized = t / self.tfinal  # Normalize time to [0,1] range
        
        position = self.bezier_curve(t_normalized)
        velocity = self.bezier_velocity(t_normalized) / self.tfinal  # Scale for total time
        acceleration = self.bezier_acceleration(t_normalized) / (self.tfinal ** 2)  # Scale second derivative

        return [position, velocity, acceleration]
