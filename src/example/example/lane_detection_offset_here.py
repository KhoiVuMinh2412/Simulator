import cv2
import numpy as np

class LaneDetector:
    def __init__(self, 
                 img_w=640, img_h=480,
                 xm_per_pix=0.135, # xcm per pixel in X
                 ym_per_pix=0.165 # xcm per pixel in Y
                 ):
        
        self.img_w = img_w
        self.img_h = img_h
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix

        # BFMC Lane Width Standard
        REAL_LANE_WIDTH_CM = 37.0
        
        # Calculated expected width in pixels
        # NOTE: Verify this matches your IPM! If your IPM dst points are 
        # wider/narrower than reality, this might need manual tuning.
        self.LANE_WIDTH_PIX = int(REAL_LANE_WIDTH_CM / self.xm_per_pix)
        
        src = np.array([
            [img_w * 0.2, img_h * 0.30],  # Top Left (Narrower)
            [img_w * 0.8, img_h * 0.30],  # Top Right
            [img_w * 1, img_h * 0.63],    # Bot Right (Wider)
            [img_w * 0, img_h * 0.63]     # Bot Left
        ], dtype=np.float32)
        
        dst = np.array([
            [img_w * 0.2, 0],
            [img_w * 0.8, 0],
            [img_w * 0.8, img_h],
            [img_w * 0.2, img_h]
        ], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def preprocess(self, img):
        # Perspective Transform (IPM)
        warped = cv2.warpPerspective(img, self.M, (self.img_w, self.img_h))
        
        # 2. Color Threshold (HLS)
        hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]
        
        s_binary = np.zeros_like(s_channel)
        
        # Combined condition: (White) OR (Yellow)
        condition = (l_channel > 160) | ((s_channel > 80) & (l_channel > 80))
        s_binary[condition] = 1 
        
        # 3. Sobel X (Vertical line detection)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        
        if np.max(abs_sobelx) == 0:
            scaled_sobel = abs_sobelx
        else:
            scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel > 30) & (scaled_sobel < 200)] = 1
        
        # 4. Combine
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        
        return combined_binary, warped

    def find_lanes(self, binary_warped):
        """
        Sliding Window Approach with Ghost Lane Logic
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        
        # Find start points (peaks in histogram)
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Hyperparameters
        nwindows = 12
        window_height = int(binary_warped.shape[0] // nwindows)
        margin = 60       
        minpix = 50       

        # Identify x and y positions of all nonzero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []

        # --- SLIDING WINDOW LOOP ---
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate arrays
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Validate Detections
        has_left = len(leftx) > 100
        has_right = len(rightx) > 100

        # --- GHOST LANE LOGIC (CRITICAL FIX) ---
        # If we lose one lane, we reconstruct it using the known lane width.
        
        if has_left and not has_right:
            # Ghost Right: Shift Left by lane width
            rightx = leftx + self.LANE_WIDTH_PIX
            righty = lefty
            has_right = True
            
        elif has_right and not has_left:
            # Ghost Left: Shift Right by lane width
            leftx = rightx - self.LANE_WIDTH_PIX
            lefty = righty
            has_left = True

        # FIT POLYNOMIALS
        left_poly = None
        right_poly = None
        
        center_offset = self.img_w / 2.0
        
        # x = f(y)
        # Using 3rd Degree (Cubic) fit for better S-curve handling
        if has_left:
            real_y = (self.img_h - lefty) * self.ym_per_pix
            real_x = (leftx - center_offset) * self.xm_per_pix
            
            left_fit = np.polyfit(real_y, real_x, 3) # Changed from 2 to 3
            left_poly = np.poly1d(left_fit)
            
        if has_right:
            real_y = (self.img_h - righty) * self.ym_per_pix
            real_x = (rightx - center_offset) * self.xm_per_pix
            
            right_fit = np.polyfit(real_y, real_x, 3) # Changed from 2 to 3
            right_poly = np.poly1d(right_fit)
            
        return left_poly, right_poly