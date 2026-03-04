#!/usr/bin/env python3

# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE


import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from .lane_detection_short import LaneDetector
# from .lane_detection_offset_here import LaneDetector
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node


class CameraHandler(Node):
    # ===================================== INIT==========================================
    def __init__(self):
        """
        Creates a bridge for converting the image from Gazebo image intro OpenCv image
        """
        super().__init__('camera_node')
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        self.detector = LaneDetector(img_w=640, img_h=480)

        self.left_poly_pub = self.create_publisher(Float64MultiArray, 'lane/left_poly', 10)
        self.right_poly_pub = self.create_publisher(Float64MultiArray, 'lane/right_poly', 10)
        
        # TODO: Changed from "/automobile/image_raw", I don't understand why.
        self.image_sub = self.create_subscription(Image, "/camera1/image_raw", self.callback, 10)

    def callback(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazsbo format
        :return: nothing but sets [cv_image] to the usefull image that can be use in opencv (numpy array)
        """
        self.get_logger().info("Image received from Gazebo") # Debug print
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        binary_warped, warped_color = self.detector.preprocess(self.cv_image)
        left_poly, right_poly = self.detector.find_lanes(binary_warped=binary_warped)

        if left_poly is not None and right_poly is not None:
            self.get_logger().info("Lanes detected! Publishing...") # Debug print
            left_poly_msg = Float64MultiArray()
            # If left_poly is a numpy.poly1d object or numpy array wrapping it
            if hasattr(left_poly, 'c'): # Check if it's a poly1d object which has coefficients 'c'
                left_poly_msg.data = left_poly.c.tolist()
            else:
                 left_poly_msg.data = list(left_poly)

            right_poly_msg = Float64MultiArray()
            if hasattr(right_poly, 'c'):
                 right_poly_msg.data = right_poly.c.tolist()
            else:
                right_poly_msg.data = list(right_poly)

            self.left_poly_pub.publish(left_poly_msg)
            self.right_poly_pub.publish(right_poly_msg)
        else:
            self.get_logger().warn("No lanes found.") # Debug print

        # Stack the raw image and the binary debug view for visualization
        # Convert binary (0/1) to (0/255) for display and make it 3-channel
        debug_view = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        debug_view = debug_view.astype(np.uint8)
        debug_view = cv2.resize(debug_view, (640, 480)) # Ensure size matches if needed
        
        # --- Draw Detected Lanes ---
        # Constants from detector
        img_h = self.detector.img_h
        img_w = self.detector.img_w
        ym_per_pix = self.detector.ym_per_pix
        xm_per_pix = self.detector.xm_per_pix
        center_offset = img_w / 2.0
        
        # Generate Y points (0 to Height)
        plot_y = np.linspace(0, img_h - 1, img_h)
        # Convert Pixel Y to Real Y (meters) for polynomial evaluation
        real_y = (img_h - plot_y) * ym_per_pix
        
        if left_poly is not None:
            # 1. P(y_real) -> x_real
            left_real_x = left_poly(real_y)
            # 2. x_real -> x_pixel
            left_plot_x = (left_real_x / xm_per_pix) + center_offset
            
            # Format points for cv2.polylines
            pts_left = np.transpose(np.vstack([left_plot_x, plot_y]))
            pts_left = pts_left.astype(np.int32)
            
            # Draw Left Lane (Red)
            cv2.polylines(debug_view, [pts_left], isClosed=False, color=(0, 0, 255), thickness=5)

        if right_poly is not None:
             # 1. P(y_real) -> x_real
            right_real_x = right_poly(real_y)
            # 2. x_real -> x_pixel
            right_plot_x = (right_real_x / xm_per_pix) + center_offset
            
            # Format points
            pts_right = np.transpose(np.vstack([right_plot_x, plot_y]))
            pts_right = pts_right.astype(np.int32)
            
            # Draw Right Lane (Blue)
            cv2.polylines(debug_view, [pts_right], isClosed=False, color=(255, 0, 0), thickness=5)
        # ---------------------------

        cv2.imshow("Main Camera", self.cv_image)
        cv2.imshow("Lane Drawing", debug_view)
        key = cv2.waitKey(1)
    

def main(args=None):
    rclpy.init(args=args)
    nod = CameraHandler()
    rclpy.spin(nod)
    nod.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
