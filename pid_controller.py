"""用于蓝线循迹转向的增强 PID 控制器。"""

import time


class EnhancedPID:
    def __init__(self, p=1.1, i=0.1, d=0.05, out_limit=80, integral_limit=50):
        self.kp = p
        self.ki = i
        self.kd = d
        self.out_limit = out_limit
        self.integral_limit = integral_limit
        self.clear()

    def clear(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, target, current):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        if dt <= 0:
            dt = 0.01

        error = target - current
        p_term = self.kp * error

        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        i_term = self.ki * self.integral

        d_term = self.kd * (error - self.last_error) / dt
        self.last_error = error

        output = p_term + i_term + d_term
        return max(-self.out_limit, min(self.out_limit, output))
