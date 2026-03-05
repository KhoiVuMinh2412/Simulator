#!/usr/bin/env python3

"""
Car Monitor Node
================
Subscribes to /automobile/localisation and /automobile/IMU topics to display
the car's position, heading and orientation in real time.

Published data is logged at a throttled rate to avoid log spam.
"""

import math

import rclpy
from rclpy.node import Node

from utils.msg import Localisation, IMU


class CarMonitor(Node):
    """Single-responsibility node: monitors and logs car pose in real time."""

    def __init__(self):
        super().__init__('car_monitor')

        # ── Parameters (overridable via config YAML) ──────────────────
        self.declare_parameter('log_rate_sec', 1.0)
        self._log_rate: float = self.get_parameter('log_rate_sec').value or 1.0

        # ── Latest state ──────────────────────────────────────────────
        self._pos_x: float = 0.0
        self._pos_y: float = 0.0
        self._gps_yaw: float = 0.0
        self._gps_ts: float = 0.0

        self._imu_roll: float = 0.0
        self._imu_pitch: float = 0.0
        self._imu_yaw: float = 0.0

        # ── Subscriptions ─────────────────────────────────────────────
        self._gps_sub = self.create_subscription(
            Localisation, '/automobile/localisation', self._gps_callback, 10
        )
        self._imu_sub = self.create_subscription(
            IMU, '/automobile/IMU', self._imu_callback, 10
        )

        # ── Throttled logging timer ──────────────────────────────────
        self._log_timer = self.create_timer(self._log_rate, self._log_state)

        self.get_logger().info(
            'Car monitor started – subscribing to /automobile/localisation '
            'and /automobile/IMU'
        )

    # ── Callbacks ─────────────────────────────────────────────────────

    def _gps_callback(self, msg: Localisation):
        self._pos_x = msg.pos_a
        self._pos_y = msg.pos_b
        self._gps_yaw = msg.rot_a
        self._gps_ts = msg.timestamp

    def _imu_callback(self, msg: IMU):
        self._imu_roll = msg.roll
        self._imu_pitch = msg.pitch
        self._imu_yaw = msg.yaw

    # ── Periodic log ──────────────────────────────────────────────────

    def _log_state(self):
        yaw_deg = math.degrees(self._gps_yaw)
        imu_yaw_deg = math.degrees(self._imu_yaw)

        self.get_logger().info(
            f'[GPS] x={self._pos_x:+.3f} m  y={self._pos_y:+.3f} m  '
            f'yaw={yaw_deg:+.1f}°  t={self._gps_ts:.2f}  |  '
            f'[IMU] roll={math.degrees(self._imu_roll):+.1f}°  '
            f'pitch={math.degrees(self._imu_pitch):+.1f}°  '
            f'yaw={imu_yaw_deg:+.1f}°'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CarMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
