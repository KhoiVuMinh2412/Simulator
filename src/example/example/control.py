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
# from .lane_keeping import LaneController
from .lane_keeping_PID import LaneController
# from .lane_keeping_Stanley import LaneController
from .RcBrainThread import RcBrainThread
from .modeChanger import StateChanger
from .systemMode import CarMode, CarSpeed
import numpy as np
import json
import sys, select, termios, tty
import threading
import time

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

        # ── State Machine (sign-detection driven) ──────────────────────
        self.state_changer = StateChanger()
        self.prev_car_mode = CarMode.STRAIGHT
        self.prev_car_speed = CarSpeed.NORMAL
        self.last_state_time = time.time()

        # Subscribe to sign detections published by sign_detector node
        self.det_sub = self.create_subscription(
            Float64MultiArray, 'sign/detections',
            self.detection_callback, 10)

        # Periodic timer to tick state-changer elapsed time (10 Hz)
        self.state_timer = self.create_timer(0.1, self.state_timer_callback)

        # MANUAL CONTROL SETUP
        self.manual_mode = False
        self.settings = termios.tcgetattr(sys.stdin)
        
        # Initialize RcBrainThread for advanced control logic
        self.rcBrain = RcBrainThread()
        self.dirKeys   = ['w', 'a', 's', 'd']
        self.paramKeys = ['t','g','y','h','u','j','i','k', 'r', 'p']
        self.pidKeys = ['z','x','v','b','n','m']
        self.allKeys = self.dirKeys + self.paramKeys + self.pidKeys
        
        # Key tracking for press/release simulation
        self.key_timestamps = {}
        self.key_timeout = 0.2 # Seconds before considering a key released

        # Dashboard state
        self._dash_steer = 0.0
        self._dash_speed = 0.0
        self._dash_lane = ''
        self._dash_detections = []

        self.print_instructions()
        
        # Start keyboard thread
        self.key_thread = threading.Thread(target=self.keyboard_loop)
        self.key_thread.daemon = True
        self.key_thread.start()

    def print_instructions(self):
        print("\n" + "=" * 50)
        print("           CONTROL NODE STARTED")
        print("=" * 50)
        print("  Default Mode : AUTONOMOUS")
        print("  TAB          : Toggle MANUAL / AUTONOMOUS")
        print("-" * 50)
        print("  MANUAL CONTROLS (RcBrain):")
        print("  w/s   Speed +/-    a/d   Steer L/R")
        print("  Space Brake        t/g   Max Speed +/-")
        print("  y/h   Max Steer    u/j   Speed Step +/-")
        print("  i/k   Steer Step   r     Reset Params")
        print("  p     PID Toggle")
        print("=" * 50 + "\n")

    # ── Dashboard ───────────────────────────────────────────────────
    def _print_dashboard(self):
        """Print a single-line status bar using carriage return to overwrite."""
        mode_str = "MANUAL" if self.manual_mode else "AUTO"
        car_mode = self.state_changer._get_mode().name
        car_speed = self.state_changer._get_speed().name

        det_str = ','.join(self._dash_detections) if self._dash_detections else '-'

        line = (
            f"[{mode_str}] "
            f"mode={car_mode:<12s} "
            f"spd_mode={car_speed:<7s} "
            f"steer={self._dash_steer:>6.1f}  "
            f"speed={self._dash_speed:>5.1f}cm/s  "
            f"lane={self._dash_lane:<10s} "
            f"det={det_str}"
        )

        # Pad to clear any leftover chars from a longer previous line
        sys.stdout.write('\r' + line.ljust(160) + '\r')
        sys.stdout.flush()

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05) # Fast poll
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def keyboard_loop(self):
        while rclpy.ok():
            key = self.getKey()
            current_time = time.time()
            
            # Mode Toggle (TAB = \t)
            if key == '\t':
                self.manual_mode = not self.manual_mode
                mode_str = "MANUAL" if self.manual_mode else "AUTONOMOUS"
                print(f"\n[SWITCH] Switched to {mode_str} mode")
                # When switching out of manual, maybe stop the car?
                if not self.manual_mode:
                    # Clear any held keys
                    for k in list(self.key_timestamps.keys()):
                        self._process_rc_command('r.' + k)
                    self.key_timestamps.clear()
                continue
            
            if not self.manual_mode:
                # Consume keys but do nothing if not manual
                # (Or maybe allow param tuning in auto mode? User probably wants full manual takeover)
                continue

            # Handle Press
            if key:
                if key == '\x03': # Ctrl-C
                    break
                
                # Special handling for Space (Brake)
                if key == ' ':
                    key_char = 'space'
                else:
                    key_char = key.lower()

                # If key valid for RcBrain
                if key_char in self.allKeys or key_char == 'space':
                    # If new press or repeat
                    # With RcBrain, 'p.w' starts acceleration. Repeated 'p.w' is harmless or ignored.
                    # We just need to ensure we register it as active.
                    
                    is_new_press = key_char not in self.key_timestamps
                    
                    # For toggles (like 'p' for PID, 'r' for Reset), only send on new press
                    # For continuous controls (w,a,s,d) and increments (u,j,i,k), allow repeats
                    should_send = True
                    if key_char in ['p', 'r'] and not is_new_press:
                        should_send = False

                    if should_send:
                        # print(f"Press {key_char}")
                        self._process_rc_command('p.' + key_char)
                    
                    self.key_timestamps[key_char] = current_time

            # Handle Release (Timeout)
            # Check all active keys
            for k in list(self.key_timestamps.keys()):
                if current_time - self.key_timestamps[k] > self.key_timeout:
                    # Timed out, consider released
                    # print(f"Release {k}")
                    self._process_rc_command('r.' + k)
                    del self.key_timestamps[k]

    def _process_rc_command(self, key_msg):
        # key_msg is like 'p.w' or 'r.w'
        command = self.rcBrain.getMessage(key_msg)
        if command is not None:
            # Publish
            # print(f"Cmd: {command}")
            command_str = json.dumps(command)
            self.publisher.publish(String(data=command_str))

    # Removed old publish_manual_command logic as RcBrain handles it

    # ── Sign Detection / State Machine ──────────────────────────────
    def detection_callback(self, msg: Float64MultiArray):
        """Receive sign detections from SignDetector and feed them into StateChanger."""
        data = msg.data
        if len(data) == 0:
            # No detections this frame – still let the recorder decay counters
            self.state_changer.record_detection([], [])
            return

        # Unpack flat list: every 5 elements = [cls_idx, cx, cy, w, h]
        n = len(data) // 5
        idxes = []
        boxes = []
        for i in range(n):
            base = i * 5
            idxes.append(int(data[base]))
            boxes.append([data[base + 1], data[base + 2],
                          data[base + 3], data[base + 4]])

        self.state_changer.record_detection(idxes, boxes)

        # Update dashboard detection names
        sign_names = {
            0: 'pedestrian', 1: 'cyclist', 2: 'car', 3: 'bus', 4: 'truck',
            5: 'red_light', 6: 'yellow_light', 7: 'green_light',
            8: 'crosswalk', 9: 'enter_hwy', 10: 'leave_hwy',
            11: 'oneway', 12: 'parking', 13: 'priority',
            14: 'noentry', 15: 'roundabout', 16: 'stop'
        }
        self._dash_detections = [sign_names.get(i, str(i)) for i in idxes]

    def state_timer_callback(self):
        """Tick the state-changer timer and evaluate mode/speed changes."""
        now = time.time()
        dt = now - self.last_state_time
        self.last_state_time = now

        self.state_changer.update_timer(dt)
        self.state_changer.change_state()

        new_mode = self.state_changer._get_mode()
        new_speed = self.state_changer._get_speed()

        # Log state transitions
        if new_mode != self.prev_car_mode:
            self.get_logger().warn(
                f'[SM] Mode: {self.prev_car_mode.name} -> {new_mode.name}')
            self.prev_car_mode = new_mode
        if new_speed != self.prev_car_speed:
            self.get_logger().warn(
                f'[SM] Speed: {self.prev_car_speed.name} -> {new_speed.name}')
            self.prev_car_speed = new_speed

        # If the state-machine says STOP and we are in autonomous mode,
        # immediately publish a zero-speed command so the car halts.
        if not self.manual_mode and new_speed == CarSpeed.STOP:
            stop_cmd = {"action": "1", "speed": 0.0}
            self.publisher.publish(String(data=json.dumps(stop_cmd)))

    def left_poly_callback(self, msg):
        self.current_left_poly = msg
        self.check_and_compute()

    def right_poly_callback(self, msg):
        self.current_right_poly = msg
        self.check_and_compute()

    def check_and_compute(self):
        if self.current_left_poly is not None and self.current_right_poly is not None:
            if not self.manual_mode:
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
        current_speed = 0

        # 2. Call your lane keeping algorithm
        steer, speed, state = self.controller.get_control(
            left_poly, right_poly, current_speed=current_speed)
        current_speed = speed

        # ── 3. Modulate speed & steer using the state-machine ──────────
        car_speed_enum = self.state_changer._get_speed()
        car_mode_enum = self.state_changer._get_mode()

        if car_speed_enum == CarSpeed.STOP:
            speed = 0.0
        elif car_speed_enum == CarSpeed.SLOW:
            # Cap speed to SLOW value (cm/s)
            speed = min(speed, float(CarSpeed.SLOW.value))
        elif car_speed_enum == CarSpeed.FAST:
            # Allow higher speed up to FAST value (cm/s)
            speed = max(speed, float(CarSpeed.FAST.value))
        # else NORMAL – keep lane-keeping speed as-is

        # Mode-based steering adjustments (placeholder hooks)
        if car_mode_enum == CarMode.PARKING:
            # Parking mode: reduce speed further; steering handled by lane-keeping for now
            speed = min(speed, float(CarSpeed.SLOW.value))
        elif car_mode_enum == CarMode.OVERTAKING:
            # Future: bias steer to avoid vehicle ahead
            pass
        elif car_mode_enum == CarMode.TAILING:
            speed = min(speed, float(CarSpeed.SLOW.value))

        # 4. Create and publish the command
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

        # Update dashboard state
        self._dash_steer = float(steer)
        self._dash_speed = float(speed)
        self._dash_lane = str(state)
        self._print_dashboard()


def main(args=None):
    rclpy.init(args=args)
    nod = RemoteControlTransmitterProcess()
    rclpy.spin(nod)
    nod.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
