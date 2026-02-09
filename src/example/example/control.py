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

from std_msgs.msg import Float64MultiArray, String
from .lane_keeping import LaneController
import numpy as np
import json

import rclpy
from rclpy.node import Node

class RemoteControlTransmitterProcess(Node):
    # ===================================== INIT==========================================
    def __init__(self):
        """
        Process the lane polynomials and publishes the command for the car.
        """
        super().__init__('control_node')
        
        self.publisher = self.create_publisher(String, '/automobile/command', 1)
        
        self.left_poly_sub = self.create_subscription(Float64MultiArray, 'lane/left_poly', self.left_poly_callback, 10)
        self.right_poly_sub = self.create_subscription(Float64MultiArray, 'lane/right_poly', self.right_poly_callback, 10)
        
        self.current_left_poly = None
        self.current_right_poly = None
        
        self.controller = LaneController()

    def left_poly_callback(self, msg):
        self.current_left_poly = msg
        self.check_and_compute()

    def right_poly_callback(self, msg):
        self.current_right_poly = msg
        self.check_and_compute()

    def check_and_compute(self):
        if self.current_left_poly is not None and self.current_right_poly is not None:
            self.lane_data_callback(self.current_left_poly, self.current_right_poly)
            # Reset after processing to wait for new pair
            self.current_left_poly = None
            self.current_right_poly = None

    def lane_data_callback(self, left_poly_msg, right_poly_msg):
        # 1. Extract the data
        left_coeffs = np.array(left_poly_msg.data)
        right_coeffs = np.array(right_poly_msg.data)

        left_poly = np.poly1d(left_coeffs)
        right_poly = np.poly1d(right_coeffs)

        # 2. Call your lane keeping algorithm
        steer, speed, state = self.controller.get_control(left_poly, right_poly)

        # 3. Create and publish the command
        # Send Speed
        # Convert cm/s (controller) to m/s (gazebo)
        speed_cmd = {
            "action": "1",
            "speed": float(speed) / 100.0
        }
        self.publisher.publish(String(data=json.dumps(speed_cmd)))

        # Send Steer
        steer_cmd = {
            "action": "2",
            "steerAngle": float(steer)
        }
        self.publisher.publish(String(data=json.dumps(steer_cmd)))


def main(args=None):
    rclpy.init(args=args)
    nod = RemoteControlTransmitterProcess()
    rclpy.spin(nod)
    nod.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
