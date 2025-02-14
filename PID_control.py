import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self.integral = 0.0
        self.previous_error = 0.0
    def update(self, error):
        P = self.Kp * error
        self.integral += error * self.dt
        I = self.Ki * self.integral
        D = self.Kd * (error - self.previous_error) / self.dt
        control_signal = P + I + D
        return control_signal
    