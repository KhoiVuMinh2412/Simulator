import time

class PID:
    def __init__(self, Kp, Ki, Kd, Min_Output = -22, Max_output = 22):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.Max_Output = Max_output
        self.Min_Output = Min_Output

        self.previous_error = 0
        self.previous_time = time.time()
        self.integral = 0

    def Calculate_PID(self, error):
        current_time = time.time()
        dt = current_time - self.previous_time
        if dt <= 0: 
            dt = 1e-4

        P = self.Kp * error

        self.integral += error * dt
        I = self.Ki * self.integral

        Derivatives = (error - self.previous_error)/dt
        D = self.Kd * Derivatives

        Output = P + I + D

        self.previous_error = error
        self.previous_time = current_time

        return max(self.Min_Output, min(self.Max_Output, Output))

    def reset_PID(self):
        self.previous_error = 0
        self.integral = 0
        self.previous_time = time.time()




