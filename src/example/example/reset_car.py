#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
import math

# Car initial position from launch file
# x 0.82 -y -14.91 -z 0.032939 -Y 1.570796

def euler_to_quaternion(roll, pitch, yaw):
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]

class CarResetter(Node):
    def __init__(self):
        super().__init__('car_resetter')
        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service /gazebo/set_entity_state not available, waiting again...')
        self.req = SetEntityState.Request()

    def send_request(self):
        self.req.state.name = 'automobile'
        self.req.state.pose.position.x = 0.82
        self.req.state.pose.position.y = -14.91
        self.req.state.pose.position.z = 0.032939
        
        q = euler_to_quaternion(0, 0, 1.570796)
        self.req.state.pose.orientation.x = q[0]
        self.req.state.pose.orientation.y = q[1]
        self.req.state.pose.orientation.z = q[2]
        self.req.state.pose.orientation.w = q[3]
        
        # Stop the car
        self.req.state.twist.linear.x = 0.0
        self.req.state.twist.linear.y = 0.0
        self.req.state.twist.linear.z = 0.0
        self.req.state.twist.angular.x = 0.0
        self.req.state.twist.angular.y = 0.0
        self.req.state.twist.angular.z = 0.0
        # self.req.state.reference_frame = 'world' # Default is world
        
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    car_resetter = CarResetter()
    
    try:
        response = car_resetter.send_request()
        if response and response.success:
            car_resetter.get_logger().info('Car RESET successfully.')
        else:
            car_resetter.get_logger().error('Failed to reset car.')
    except Exception as e:
        car_resetter.get_logger().error(f'Service call failed: {e}')
        
    car_resetter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
