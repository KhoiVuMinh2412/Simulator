import numpy as np
import math
import time
from .PID_controller import PID

class LaneController:
    def __init__(self, 
                 wheelbase=26.5,
                 max_steering_angle=20,
                 max_speed=25.0, 
                 min_speed=10.0,
                 lane_width=37.0):
        
        self.L = float(wheelbase)
        self.max_steer = np.deg2rad(max_steering_angle)
        self.max_speed = float(max_speed)
        self.min_speed = float(min_speed)
        self.lane_width = float(lane_width)

        # Tuning
        self.k_lookahead = 0.8    
        self.min_lookahead = 25.0 
        self.max_lookahead = 60.0 
        self.alpha = 0.7          
        
        self.prev_steer = 0.0
        self.no_lane_counter = 0 
        self.prev_time = time.time()
        self.PATIENCE_LIMIT = 15   

        self.steer_Ki = 0.4
        self.steer_Kd = 0.23
        self.steer_integral = 0
        self.prev_CrossTrackError = 0

    def get_control(self, left_poly, right_poly, current_speed):
        target_poly, offset_mode = self._select_target_path(left_poly, right_poly)

        # 1. Handle No Lane Found
        if target_poly is None:
            self.no_lane_counter += 1
            if self.no_lane_counter < self.PATIENCE_LIMIT:
                return np.rad2deg(self.prev_steer), self.min_speed, "F" #free
            else:
                return 0.0, 0.0, "N" #No_lane
        
        self.no_lane_counter = 0 

        # 2. Calculate Lookahead
        lookahead_radius = self.min_lookahead + (self.k_lookahead * current_speed)
        lookahead_radius = np.clip(lookahead_radius, self.min_lookahead, self.max_lookahead)

        try:
            # 3. Pure Pursuit Math -- P term -- Cross track error
            gx, gy = self._find_circle_intersection(target_poly, lookahead_radius, offset_mode)
            
            # Calculate Steer
            alpha_angle = math.atan2(gx, gy) 
            steer_rad = math.atan((2 * self.L * math.sin(alpha_angle)) / lookahead_radius)

            current_CrossTrackError = gx
            current_time = time.time()
            dt = current_time - self.prev_time
            if dt <= 0: dt = 1e-4
            steer_D = (current_CrossTrackError - self.prev_CrossTrackError)/dt

            scaling = 0.001

            self.steer_integral += current_CrossTrackError * dt
            self.steer_integral = np.clip(self.steer_integral, -1000, 1000)
            ID_PurePursuit = (self.steer_Ki * self.steer_integral) + (self.steer_Kd * steer_D)

            steer_rad += ID_PurePursuit * scaling
            steer_rad = np.clip(steer_rad, -self.max_steer, self.max_steer)
            
            # Smoothing
            steer_rad = (self.alpha * steer_rad) + ((1 - self.alpha) * self.prev_steer)
            self.prev_steer = steer_rad
            self.prev_CrossTrackError = current_CrossTrackError
            self.prev_time = current_time
            
            # Speed PID control
            target_speed = self.max_speed - (abs(steer_rad) * 20.0)
            target_speed = max(self.min_speed, min(self.max_speed, target_speed))

            state = "S" #Straight
            if np.rad2deg(steer_rad) > 5: state = "R" #right
            elif np.rad2deg(steer_rad) < -5: state = "L" #left

            return np.rad2deg(steer_rad), target_speed, state

        except Exception as e:
            print(f"\n[MATH ERROR] Details: {e}")
            if target_poly: print(f"Poly Coeffs: {target_poly.coeffs}")
            print(f"Radius: {lookahead_radius}")
            return np.rad2deg(self.prev_steer), self.min_speed, "MATH_ERR"

    def _select_target_path(self, left_poly, right_poly):
        if left_poly is not None and right_poly is not None:
            avg_poly = (left_poly + right_poly) / 2
            return avg_poly, "C" #center
        
        return None, "NONE"

    def _find_circle_intersection(self, poly, radius, mode):
        coeffs = poly.coeffs.copy().astype(float)
        radius = float(radius)

        squared_coeffs = np.polymul(coeffs, coeffs)
        
        y_sq_coeffs = np.array([1.0, 0.0, 0.0], dtype=float)
        final_coeffs = np.polyadd(squared_coeffs, y_sq_coeffs)
        
        final_coeffs[-1] -= (radius**2)
        
        roots = np.roots(final_coeffs)
        
        real_roots = roots[np.isreal(roots)].real
        forward_roots = real_roots[real_roots > 0]
        
        if len(forward_roots) == 0:
            y = radius
            x = np.polyval(coeffs, y)
            return (x, y)
        
        y = np.min(forward_roots)
        x = np.polyval(coeffs, y)
        
        return (x, y)
