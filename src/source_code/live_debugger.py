import cv2
import numpy as np

class LaneVisualizer:
    def __init__(self, img_w=640, img_h=480):
        self.w = img_w
        self.h = img_h
        
        # MUST match the constants in lane_detection.py
        self.xm_per_pix = 0.135
        self.ym_per_pix = 0.165
        self.lane_width_cm = 37.0 

    def draw_debug_frame(self, warped, left_poly, right_poly, target_point, steer_deg, speed):
        # Create a color canvas
        if len(warped.shape) == 2:
            debug_img = cv2.cvtColor(warped * 255, cv2.COLOR_GRAY2BGR)
        else:
            debug_img = warped.copy()

        # 1. Prepare Y-values for plotting
        # We generate pixels 0..479, but we must convert them to WORLD Y (cm) 
        # to use the polynomial.
        ploty_pixels = np.linspace(0, self.h - 1, self.h)
        ploty_world = (self.h - ploty_pixels) * self.ym_per_pix

        # Helper function to draw any polynomial
        def draw_poly_curve(poly, color, offset_cm=0):
            if poly is None: return
            
            # Calculate X in World coordinates (cm)
            try:
                fitx_world = poly(ploty_world) + offset_cm
                
                # Convert (World X, World Y) -> (Pixel X, Pixel Y)
                fitx_pix = (fitx_world / self.xm_per_pix) + (self.w / 2.0)
                fity_pix = ploty_pixels # We already know these
                
                # Filter valid points
                valid = (fitx_pix >= 0) & (fitx_pix < self.w)
                pts = np.array([np.transpose(np.vstack([fitx_pix[valid], fity_pix[valid]]))])
                
                cv2.polylines(debug_img, np.int32([pts]), isClosed=False, color=color, thickness=4)
            except Exception as e:
                print(f"Draw error: {e}")

        # --- DRAW LANES ---
        # Draw Left (Blue in BGR is 255, 0, 0)
        draw_poly_curve(left_poly, (255, 0, 0)) 
        
        # Draw Right (Red in BGR is 0, 0, 255)
        draw_poly_curve(right_poly, (0, 0, 255))

        # --- DRAW CENTER ---
        center_poly = None
        if left_poly is not None and right_poly is not None:
            # Average the coefficients
            avg_coeffs = (left_poly.coeffs + right_poly.coeffs) / 2
            center_poly = np.poly1d(avg_coeffs)
            draw_poly_curve(center_poly, (0, 255, 255)) # Yellow
            
        elif left_poly is not None:
            # Offset left lane by +37cm
            draw_poly_curve(left_poly, (0, 255, 255), offset_cm=self.lane_width_cm/2)
            
        elif right_poly is not None:
            # Offset right lane by -37cm
            draw_poly_curve(right_poly, (0, 255, 255), offset_cm=-self.lane_width_cm/2)

        # --- DRAW PURE PURSUIT TARGET ---
        if target_point is not None:
            tx_world, ty_world = target_point # These are in CM
            
            # Convert to Pixels
            tx_pix = int((tx_world / self.xm_per_pix) + (self.w / 2.0))
            ty_pix = int(self.h - (ty_world / self.ym_per_pix))
            
            # Draw Target Dot (Purple)
            cv2.circle(debug_img, (tx_pix, ty_pix), 8, (255, 0, 255), -1)
            # Draw line from car hood (bottom center) to target
            cv2.line(debug_img, (int(self.w/2), self.h), (tx_pix, ty_pix), (255, 0, 255), 2)

        # --- DRAW TEXT ---
        # Moved Speed down to y=60 to avoid overlap
        cv2.putText(debug_img, f"Steer: {steer_deg:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Speed: {speed:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return debug_img