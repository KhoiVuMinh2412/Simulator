#!/usr/bin/env python3

"""
Sign Detector Node
-------------------
Subscribes to the camera image, runs YOLO inference for traffic sign / object
detection, and publishes the results as a Float64MultiArray.

Published message format (on 'sign/detections'):
    Flat array where every 5 consecutive elements represent one detection:
        [class_idx, center_x_norm, center_y_norm, width_norm, height_norm, ...]
    Normalised coordinates (0-1) relative to image dimensions (xywhn).
    An empty array means no detections.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

# Attempt to import ultralytics; gracefully degrade if unavailable
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO: type | None = None
    YOLO_AVAILABLE = False


class SignDetector(Node):
    def __init__(self):
        super().__init__('sign_detector_node')

        # ── ROS Parameters ──────────────────────────────────────────────
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence', 0.50)
        self.declare_parameter('imgsz', 416)
        self.declare_parameter('device', '')          # '' = auto (cpu / cuda)
        self.declare_parameter('half', False)          # FP16 inference
        self.declare_parameter('vid_stride', 2)        # process every N-th frame

        model_path = self.get_parameter('model_path').value
        self.conf = self.get_parameter('confidence').value
        self.imgsz = self.get_parameter('imgsz').value
        self.device = self.get_parameter('device').value or None
        self.half = self.get_parameter('half').value
        self.vid_stride = int(self.get_parameter('vid_stride').value or 2)

        # ── YOLO Model ──────────────────────────────────────────────────
        self.model = None
        if not YOLO_AVAILABLE:
            self.get_logger().error(
                'ultralytics is not installed. '
                'Run: pip install ultralytics')
        elif not model_path or not os.path.isfile(model_path):
            self.get_logger().warn(
                f'No valid YOLO model at "{model_path}".  '
                'Sign detection disabled until a valid model_path parameter is set.  '
                'Place your .pt weights in example/models/ and pass the path as a ROS param.')
        else:
            self.get_logger().info(f'Loading YOLO model from {model_path} …')
            try:
                assert YOLO is not None
                self.model = YOLO(model_path)
                self.get_logger().info('YOLO model loaded successfully.')
            except Exception as e:
                self.get_logger().error(f'Failed to load YOLO model: {e}')

        # ── Publishers / Subscribers ────────────────────────────────────
        self.det_pub = self.create_publisher(Float64MultiArray, 'sign/detections', 10)

        self.bridge = CvBridge()
        self.frame_counter = 0
        self.image_sub = self.create_subscription(
            Image, '/camera1/image_raw', self.image_callback, 10)

    # ────────────────────────────────────────────────────────────────────
    def image_callback(self, msg: Image):
        """Process incoming camera frames."""
        self.frame_counter += 1

        # Skip frames according to vid_stride
        if self.frame_counter % self.vid_stride != 0:
            return

        if self.model is None:
            # Publish empty detection so downstream knows the node is alive
            self.det_pub.publish(Float64MultiArray(data=[]))
            return

        # Convert ROS Image → OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Run YOLO inference
        results = self.model.predict(
            source=cv_image,
            imgsz=self.imgsz,
            conf=self.conf,
            half=self.half,
            device=self.device,
            verbose=False
        )

        # Pack results into Float64MultiArray
        det_msg = Float64MultiArray()
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                cls_ids = r.boxes.cls.cpu().numpy().astype(float)
                xywhn = r.boxes.xywhn.cpu().numpy().astype(float)  # normalised

                flat = []
                for cls_id, box in zip(cls_ids, xywhn):
                    flat.extend([float(cls_id), *box.tolist()])
                det_msg.data = flat

                n = len(cls_ids)
                names = [self.model.names[int(c)] for c in cls_ids]
                self.get_logger().info(
                    f'Detected {n} object(s): {names}',
                    throttle_duration_sec=2.0)

        self.det_pub.publish(det_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SignDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
