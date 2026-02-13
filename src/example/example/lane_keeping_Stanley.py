import numpy as np
import math
import time

class LaneController:
    def __init__(self, 
                 wheelbase=26.5,
                 max_steering_angle=25,
                 max_speed=25.0, 
                 min_speed=10.0,
                 lane_width=37.0):
        
        self.L = float(wheelbase)
        self.max_steer = np.deg2rad(max_steering_angle)
        self.max_speed = float(max_speed)
        self.min_speed = float(min_speed)
        self.lane_width = float(lane_width)

        # Low (0.5 - 1.0) = Smooth
        # High (2.0 - 5.0) = Tight 
        self.k_gain =0.77   
        
        self.k_soft = 1.0   

        self.alpha = 0.7
        self.prev_steer = 0.0
        self.no_lane_counter = 0 
        self.PATIENCE_LIMIT = 15   

    def get_control(self, left_poly, right_poly, current_speed):
        target_poly, offset_mode = self._select_target_path(left_poly, right_poly)

        # 1. Handle No Lane Found
        if target_poly is None:
            self.no_lane_counter += 1
            if self.no_lane_counter < self.PATIENCE_LIMIT:
                return np.rad2deg(self.prev_steer), self.min_speed, "F" # Free
            else:
                return 0.0, 0.0, "N" # No Lane
        
        self.no_lane_counter = 0 

        try:

            CrossTrackError = target_poly(0)

            poly_deriv = target_poly.deriv()
            heading_error = math.atan(poly_deriv(0))

            # safe speed for division
            v = max(current_speed, 1.0) 

            # Calculate the cross-track correction term
            cross_track_steering = math.atan((self.k_gain * CrossTrackError) / (v + self.k_soft))
            
            # Combine terms
            steer_rad = heading_error + cross_track_steering
            
            # 5. Clip & Smooth
            steer_rad = np.clip(steer_rad, -self.max_steer, self.max_steer)
            steer_rad = (self.alpha * steer_rad) + ((1 - self.alpha) * self.prev_steer)
            self.prev_steer = steer_rad
            
            # 6. Smooth Speed 
            target_speed = self.max_speed - (abs(steer_rad) * 20.0)
            target_speed = max(self.min_speed, min(self.max_speed, target_speed))

            state = "S" 
            if abs(np.rad2deg(steer_rad)) > 5:
                state = "R" if steer_rad > 0 else "L"

            return np.rad2deg(steer_rad), target_speed, state

        except Exception as e:
            print(f"Stanley Error: {e}")
            return np.rad2deg(self.prev_steer), self.min_speed, "ERR"

    def _select_target_path(self, left_poly, right_poly):
        if left_poly is not None and right_poly is not None:
            avg_coeffs = (left_poly.coeffs + right_poly.coeffs) / 2
            return np.poly1d(avg_coeffs), "C"
        elif left_poly is not None:
            # Adjust to center: Add half lane width
            coeffs = left_poly.coeffs.copy()
            coeffs[-1] += (self.lane_width / 2.0)
            return np.poly1d(coeffs), "RO" 
        elif right_poly is not None:
            # Adjust to center: Subtract half lane width
            coeffs = right_poly.coeffs.copy()
            coeffs[-1] -= (self.lane_width / 2.0)
            return np.poly1d(coeffs), "LO" 
        return None, "NONE"
