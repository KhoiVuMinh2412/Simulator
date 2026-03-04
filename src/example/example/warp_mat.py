import cv2
import numpy as np
import math
import time

# ==============================================================================
# CONFIGURATION (Replaces ROS Parameters & Constants)
# ==============================================================================
class Config:
    # Input Source: '0' for webcam, or path to video file e.g. 'test_video.mp4'
    VIDEO_SOURCE = 'output_video_1.avi'
    
    # Camera / IPM Settings
    RESOLUTION = 666.0       # pixels per meter (in BEV)
    FAR_M = 15.0             # How far to look ahead (meters)
    NEAR_M = 2.0             # How close to look (meters)
    WIDTH_M = 8.0            # Width of the road covered (meters)
    
    # Camera Intrinsic (fx, fy, cx, cy)
    # Adjust these to match your actual camera calibration
    CAM_PARAMS = [1230.0, 1230.0, 640.0, 360.0] 
    
    # Camera Extrinsic (x, y, z, roll, pitch, yaw)
    # Position of camera relative to rear axle/ground
    CAM_TF = [0.0, 0.0, 0.2, 0.0, np.deg2rad(35), 0.0]

    # Lane Detection Params
    LANE_WIDTH_M = 3.7       # Standard lane width
    LANE_WHITE_M = 0.15      # Width of the white line itself
    
    # Visualization
    SHOW_FPS = True
    SHOW_BEV = True          # Show Bird's Eye View window

# ==============================================================================
# IPM CAMERA CLASS
# ==============================================================================
class IPMCamera:
    def __init__(self):
        # Unpack Config
        self.resolution = Config.RESOLUTION
        self.far_m = Config.FAR_M
        self.near_m = Config.NEAR_M
        self.width_m = Config.WIDTH_M
        
        # 1. Build Intrinsic Matrix K
        fx, fy = Config.CAM_PARAMS[0], Config.CAM_PARAMS[1]
        cx, cy = Config.CAM_PARAMS[2], Config.CAM_PARAMS[3]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)

        # 2. Build Rotation Matrix R
        # Note: Code assumes standard ROS/Automotive coordinate systems (x=fwd, y=left, z=up)
        # converted to Camera Optical (z=fwd, x=right, y=down)
        tf = Config.CAM_TF
        roll, pitch, yaw = tf[3], tf[4], tf[5]

        # Rotation Matrices (Negative angles logic derived from original C++)
        c_y, s_y = np.cos(-yaw), np.sin(-yaw)
        Rz = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]])

        c_p, s_p = np.cos(-pitch), np.sin(-pitch)
        Ry = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])

        c_r, s_r = np.cos(-roll), np.sin(-roll)
        Rx = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])

        # Axis switch matrix (World -> Camera Optical)
        # x_cam = -y_world, y_cam = -z_world, z_cam = x_world
        Rs = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

        R = Rs @ (Rz @ (Ry @ Rx))

        # 3. Translation t = -R * C
        C = np.array([tf[0], tf[1], tf[2]]).reshape(3, 1)
        t = -R @ C

        # 4. Projection Matrix P = K * [R|t]
        Rt = np.hstack((R, t))
        P = K @ Rt

        # 5. Homography Matrix H = P * M
        # M maps Ground Plane (u_bev, v_bev) -> World (x, y, z=0) -> Camera
        px_per_m = self.resolution
        M = np.array([
            [1.0/px_per_m,          0.0,  self.near_m],
            [0.0,         -1.0/px_per_m,  self.width_m/2.0],
            [0.0,                   0.0,           0.0],
            [0.0,                   0.0,           1.0]
        ])
        
        H = P @ M
        
        # Invert for IPM (Image -> Ground)
        try:
            self.ipm_transform = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("Error: Singular Matrix in IPM!")
            self.ipm_transform = np.eye(3)

        # 6. Build Lookup Maps for cv2.remap
        self.out_h = int((self.far_m - self.near_m) * px_per_m)
        self.out_w = int(self.width_m * px_per_m)
        
        # Grid of destination pixels (u, v)
        grid_u, grid_v = np.meshgrid(np.arange(self.out_w), np.arange(self.out_h))
        
        # Homogeneous coords: [u, v, 1]
        # Shape: (3, N)
        dst_pts = np.vstack([grid_u.ravel(), grid_v.ravel(), np.ones_like(grid_u.ravel())])
        
        # Map backwards: H_inv * dst_pts
        src_pts = self.ipm_transform @ dst_pts 
        
        # Normalize w: x/w, y/w
        w_vec = src_pts[2, :]
        w_vec[w_vec == 0] = 1e-6 
        
        self.map_x = (src_pts[0, :] / w_vec).reshape(self.out_h, self.out_w).astype(np.float32)
        self.map_y = (src_pts[1, :] / w_vec).reshape(self.out_h, self.out_w).astype(np.float32)

    def get_ipm(self, img):
        if img is None: return None
        # Remap
        out = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
        # Rotate to make "Up" the direction of travel
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return out

# ==============================================================================
# LANE DETECTOR CLASS
# ==============================================================================
class LaneDetector:
    def __init__(self):
        self.ipm_camera = IPMCamera()
        
        # Scaling Constants
        self.METER_PER_PIXEL = 1.0 / Config.RESOLUTION
        self.LANE_WIDTH_PX = Config.LANE_WIDTH_M / self.METER_PER_PIXEL
        
        # Detection Constants
        self.WINDOW_MARGIN = int(Config.LANE_WHITE_M * Config.RESOLUTION)
        self.WINDOW_MIN_PIXELS = 50
        self.N_WINDOWS = 9
        
        # State Variables
        self.left_fit = None
        self.right_fit = None
        self.good_left = False
        self.good_right = False
        self.lane_to_fit = 0 # 0:NONE, 1:LEFT, 2:BOTH, 3:RIGHT
        self.stopline_dist = -1.0
        
    def preprocess(self, img):
        """ Grayscale -> Blur -> Adaptive Threshold """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 
            199, -20 # Block size, C
        )
        return binary

    def find_lane_starts(self, binary_bev):
        """ Histogram search to find starting X coordinates of lanes """
        h, w = binary_bev.shape
        # Analyze bottom 15% of image
        roi_h = int(h * 0.15)
        roi = binary_bev[h-roi_h:h, :]
        
        # Sum columns
        histogram = np.sum(roi, axis=0)
        
        # Find peaks (simple thresholding logic)
        midpoint = int(w // 2)
        left_half = histogram[:midpoint]
        right_half = histogram[midpoint:]
        
        # Argmax gives the index of the maximum value
        leftx_base = np.argmax(left_half)
        rightx_base = np.argmax(right_half) + midpoint
        
        # Basic validation: ensure there is actually some white pixels there
        if np.max(left_half) < 1000: leftx_base = None
        if np.max(right_half) < 1000: rightx_base = None
        
        return leftx_base, rightx_base

    def sliding_window_fit(self, binary_bev):
        """ Sliding Window and Polynomial Fit """
        h, w = binary_bev.shape
        leftx_base, rightx_base = self.find_lane_starts(binary_bev)
        
        # Identify non-zero pixels
        nonzero = cv2.findNonZero(binary_bev)
        if nonzero is None: return False
        nonzeroy = np.array(nonzero[:, 0, 1])
        nonzerox = np.array(nonzero[:, 0, 0])
        
        window_height = int(h // self.N_WINDOWS)
        
        left_lane_inds = []
        right_lane_inds = []
        
        # Current positions
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        margin = self.WINDOW_MARGIN
        minpix = self.WINDOW_MIN_PIXELS
        
        for window in range(self.N_WINDOWS):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height
            
            # Left Lane
            if leftx_current is not None:
                win_x_low = leftx_current - margin
                win_x_high = leftx_current + margin
                good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                left_lane_inds.append(good_left)
                if len(good_left) > minpix:
                    leftx_current = int(np.mean(nonzerox[good_left]))

            # Right Lane
            if rightx_current is not None:
                win_x_low = rightx_current - margin
                win_x_high = rightx_current + margin
                good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                right_lane_inds.append(good_right)
                if len(good_right) > minpix:
                    rightx_current = int(np.mean(nonzerox[good_right]))

        # Concatenate indices
        left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else []
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else []

        self.good_left = False
        self.good_right = False
        
        # Fit Left
        if len(left_lane_inds) > minpix * 2:
            lx = nonzerox[left_lane_inds]
            ly = nonzeroy[left_lane_inds]
            self.left_fit = np.polyfit(ly, lx, 3) # y -> x
            self.good_left = True
            
        # Fit Right
        if len(right_lane_inds) > minpix * 2:
            rx = nonzerox[right_lane_inds]
            ry = nonzeroy[right_lane_inds]
            self.right_fit = np.polyfit(ry, rx, 3)
            self.good_right = True
            
        # Failover logic (if one missing, infer from other)
        if self.good_left and not self.good_right:
            self.right_fit = np.copy(self.left_fit)
            self.right_fit[-1] += self.LANE_WIDTH_PX # Shift intercept
            self.good_right = True # Inferred
            
        if self.good_right and not self.good_left:
            self.left_fit = np.copy(self.right_fit)
            self.left_fit[-1] -= self.LANE_WIDTH_PX
            self.good_left = True # Inferred

        return self.good_left or self.good_right

    def detect_stopline(self, binary_bev):
        """ Basic horizontal line detection for Stop Lines """
        # Sum rows (horizontal projection)
        row_sum = np.sum(binary_bev, axis=1)
        h, w = binary_bev.shape
        
        # Look for a peak in row sums near the bottom, but not at the very bottom
        threshold = w * 255 * 0.4 # Line covers 40% of width
        
        # Scan bottom 50%
        search_zone = row_sum[int(h/2):]
        peaks = np.where(search_zone > threshold)[0]
        
        if len(peaks) > 0:
            # Found a line. In real code this uses RANSAC, here simplified to peak location
            # The peaks index is relative to h/2
            loc_y_rel = peaks[-1] # closest one
            loc_y_img = int(h/2) + loc_y_rel
            
            # Convert px to meters
            px_dist = h - loc_y_img
            dist_m = px_dist * self.METER_PER_PIXEL + Config.NEAR_M
            return dist_m, loc_y_img
        
        return -1.0, -1

    def generate_waypoints(self, h_px):
        """ Generate (x,y) points in Vehicle Frame for controller """
        if not (self.good_left or self.good_right):
            return []
            
        waypoints = []
        # Sample points from bottom (near) to top (far)
        for y_px in range(h_px, 0, -40): # Step size 40px
            # Evaluate polynomials
            lx = np.polyval(self.left_fit, y_px)
            rx = np.polyval(self.right_fit, y_px)
            cx = (lx + rx) / 2.0
            
            # Convert to Vehicle Coordinates
            # Image: (0,0) is Top-Left. 
            # Vehicle: X=Forward, Y=Left.
            
            # 1. Distance forward (X)
            dist_from_bottom_px = h_px - y_px
            x_vehicle = dist_from_bottom_px * self.METER_PER_PIXEL + Config.NEAR_M
            
            # 2. Lateral position (Y)
            # Center of image is 0 lateral
            center_img_x = self.ipm_camera.out_w / 2.0
            dist_from_center_px = center_img_x - cx # Positive if line is to left
            y_vehicle = dist_from_center_px * self.METER_PER_PIXEL
            
            waypoints.append((x_vehicle, y_vehicle))
            
        return waypoints

    def process_frame(self, frame):
        # 1. Preprocess
        binary = self.preprocess(frame)
        
        # 2. IPM
        bev = self.ipm_camera.get_ipm(binary)
        if bev is None: return frame, None
        
        # 3. Stopline
        sl_dist, sl_y = self.detect_stopline(bev)
        self.stopline_dist = sl_dist
        
        # 4. Lane Lines
        found_lanes = self.sliding_window_fit(bev)
        
        # 5. Waypoints
        waypoints = []
        if found_lanes:
            waypoints = self.generate_waypoints(bev.shape[0])
            
        # 6. Visualization
        # Create a color version of BEV for drawing
        viz_bev = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        
        if found_lanes:
            ploty = np.arange(bev.shape[0])
            # Draw Left (Blue)
            leftx = np.polyval(self.left_fit, ploty)
            pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))], np.int32)
            cv2.polylines(viz_bev, pts_left, False, (255, 0, 0), 3)
            
            # Draw Right (Red)
            rightx = np.polyval(self.right_fit, ploty)
            pts_right = np.array([np.transpose(np.vstack([rightx, ploty]))], np.int32)
            cv2.polylines(viz_bev, pts_right, False, (0, 0, 255), 3)
            
            # Draw Waypoints (Green Dots)
            for wp in waypoints:
                # convert back to px for display
                # x_veh = (h - y_px)*scale -> y_px = h - x_veh/scale
                x_veh, y_veh = wp
                y_px = int(bev.shape[0] - (x_veh - Config.NEAR_M) / self.METER_PER_PIXEL)
                x_px = int((self.ipm_camera.out_w / 2.0) - (y_veh / self.METER_PER_PIXEL))
                cv2.circle(viz_bev, (x_px, y_px), 4, (0, 255, 0), -1)

        # Draw Stopline
        if sl_dist > 0:
             cv2.line(viz_bev, (0, sl_y), (bev.shape[1], sl_y), (0, 255, 255), 2)
             cv2.putText(viz_bev, f"STOP: {sl_dist:.2f}m", (10, sl_y-10), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return frame, viz_bev, waypoints

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Initialize Detector
    detector = LaneDetector()
    
    # Open Video Source
    cap = cv2.VideoCapture(Config.VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {Config.VIDEO_SOURCE}")
        exit()
        
    print("Starting Lane Detection... Press 'q' to quit.")
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break
            
        # Resize for consistent processing if needed, e.g. 640x480
        frame = cv2.resize(frame, (640, 480))
        
        # Run Pipeline
        orig, bev_viz, waypoints = detector.process_frame(frame)
        
        # Visualization overlay on Original Image (basic)
        if len(waypoints) > 0:
            # Print offset on screen
            # Calculate offset from center (first waypoint y_veh)
            offset = waypoints[0][1]
            cv2.putText(frame, f"Offset: {offset:.2f}m", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # FPS Calc
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        if Config.SHOW_FPS:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 460), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display
        cv2.imshow("Front Camera", frame)
        if Config.SHOW_BEV and bev_viz is not None:
            cv2.imshow("Bird's Eye View", bev_viz)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()