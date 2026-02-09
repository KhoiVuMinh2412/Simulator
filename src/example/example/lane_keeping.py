import numpy as np
import math

class LaneController:
    def __init__(self, 
                 wheelbase=26.5,
                 max_steering_angle=18,
                 max_speed=25.0, 
                 min_speed=10.0,
                 lane_width=37.0):
        
        self.L = float(wheelbase)
        self.max_steer = np.deg2rad(max_steering_angle)
        self.max_speed = float(max_speed)
        self.min_speed = float(min_speed)
        self.lane_width = float(lane_width)

        # Tuning
        self.k_lookahead = 0.5    
        self.min_lookahead = 20.0 
        self.max_lookahead = 60.0 
        self.alpha = 0.7          
        
        self.prev_steer = 0.0
        self.no_lane_counter = 0 
        self.PATIENCE_LIMIT = 15   

    def get_control(self, left_poly, right_poly, current_speed=15.0):
        target_poly, offset_mode = self._select_target_path(left_poly, right_poly)

        # 1. Handle No Lane Found
        if target_poly is None:
            self.no_lane_counter += 1
            if self.no_lane_counter < self.PATIENCE_LIMIT:
                return np.rad2deg(self.prev_steer), self.min_speed, "FREE"
            else:
                return 0.0, 0.0, "NO_LANE"
        
        self.no_lane_counter = 0 

        # 2. Calculate Lookahead
        lookahead_radius = self.min_lookahead + (self.k_lookahead * current_speed)
        lookahead_radius = np.clip(lookahead_radius, self.min_lookahead, self.max_lookahead)

        try:
            # 3. Pure Pursuit Math
            gx, gy = self._find_circle_intersection(target_poly, lookahead_radius, offset_mode)
            
            # Calculate Steer
            alpha_angle = math.atan2(gx, gy) 
            steer_rad = math.atan((2 * self.L * math.sin(alpha_angle)) / lookahead_radius)
            steer_rad = np.clip(steer_rad, -self.max_steer, self.max_steer)
            
            # Smoothing
            steer_rad = (self.alpha * steer_rad) + ((1 - self.alpha) * self.prev_steer)
            self.prev_steer = steer_rad
            
            # Speed Control
            target_speed = self.max_speed - (abs(steer_rad) * 20.0)
            target_speed = max(self.min_speed, min(self.max_speed, target_speed))

            state = "STRAIGHT"
            if np.rad2deg(steer_rad) > 5: state = "RIGHT"
            elif np.rad2deg(steer_rad) < -5: state = "LEFT"

            return np.rad2deg(steer_rad), target_speed, state

        except Exception as e:
            print(f"\n[MATH ERROR] Details: {e}")
            if target_poly: print(f"Poly Coeffs: {target_poly.coeffs}")
            print(f"Radius: {lookahead_radius}")
            return np.rad2deg(self.prev_steer), self.min_speed, "MATH_ERR"

    def _select_target_path(self, left_poly, right_poly):
        if left_poly is not None and right_poly is not None:
            avg_coeffs = (left_poly.coeffs + right_poly.coeffs) / 2
            return np.poly1d(avg_coeffs), "CENTER"
        elif left_poly is not None:
            return left_poly, "RIGHT_OFFSET" 
        elif right_poly is not None:
            return right_poly, "LEFT_OFFSET"
        return None, "NONE"

    def _find_circle_intersection(self, poly, radius, mode):
        coeffs = poly.coeffs.copy().astype(float)
        radius = float(radius)

        if mode == "RIGHT_OFFSET":
            coeffs[-1] += (self.lane_width / 2.0)
        elif mode == "LEFT_OFFSET":
            coeffs[-1] -= (self.lane_width / 2.0)

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
