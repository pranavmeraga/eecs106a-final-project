"""
head_teleop_mapper_node.py

Maps facial motions to robot commands.

Subscribes:
- /head_pose (Vector3): x=yaw, y=pitch, z=roll
- /blink_event (Int8): 0=none, 1=long_blink
- /mouth_open (Bool): True if mouth open

Publishes:
- /base_joint_cmd (Twist): Base rotation commands
- /limb_joint_cmd (Twist): Limb movement commands
- /grasp_cmd (Bool): Gripper command
- /stop_cmd (Bool): Emergency stop
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Twist
from std_msgs.msg import Int8, Bool


# Blink events
BLINK_NONE = 0
BLINK_LONG = 1


class HeadTeleopMapperNode(Node):
    def __init__(self):
        super().__init__('head_teleop_mapper_node')

        # Subscribers
        self.head_pose_sub = self.create_subscription(
            Vector3, 'head_pose', self.head_pose_callback, 10
        )
        self.blink_sub = self.create_subscription(
            Int8, 'blink_event', self.blink_callback, 10
        )
        self.mouth_sub = self.create_subscription(
            Bool, 'mouth_open', self.mouth_callback, 10
        )

        # Publishers
        self.base_twist_pub = self.create_publisher(Twist, 'base_joint_cmd', 10)
        self.limb_twist_pub = self.create_publisher(Twist, 'limb_joint_cmd', 10)
        self.grasp_pub = self.create_publisher(Bool, 'grasp_cmd', 10)
        self.stop_pub = self.create_publisher(Bool, 'stop_cmd', 10)

        # Thresholds (radians)
        self.pitch_threshold = 0.15
        self.roll_threshold = 0.15
        self.yaw_threshold = 0.15
        self.deadzone = 0.05

        # Control gains
        self.base_yaw_gain = 0.5
        self.limb_pitch_gain = 0.5
        self.limb_roll_gain = 0.5

        # State
        self.last_grasp_time = 0.0
        self.grasp_cooldown = 2.0  # seconds

        self.get_logger().info("Head Teleop Mapper Node Started")
        self.get_logger().info("Control Mapping:")
        self.get_logger().info("  Turn Left/Right → Base rotation")
        self.get_logger().info("  Nod Up/Down → Limb vertical")
        self.get_logger().info("  Tilt Left/Right → Limb forward/back")
        self.get_logger().info("  Long Blink → Grasp")
        self.get_logger().info("  Open Mouth → Emergency Stop")

    def head_pose_callback(self, msg: Vector3):
        """Process head pose and generate motion commands."""
        dyaw = msg.x     # turn left/right
        dpitch = msg.y   # nod up/down
        droll = msg.z    # tilt left/right

        # BASE JOINT (Yaw)
        base_twist = Twist()
        if abs(dyaw) > self.deadzone:
            if dyaw < -self.yaw_threshold:
                base_twist.angular.z = -self.base_yaw_gain
                self.get_logger().info(f"⬅️  Turn LEFT: {np.degrees(dyaw):.1f}°", 
                                      throttle_duration_sec=1.0)
            elif dyaw > self.yaw_threshold:
                base_twist.angular.z = self.base_yaw_gain
                self.get_logger().info(f"➡️  Turn RIGHT: {np.degrees(dyaw):.1f}°",
                                      throttle_duration_sec=1.0)
            else:
                base_twist.angular.z = (dyaw / self.yaw_threshold) * self.base_yaw_gain
        
        self.base_twist_pub.publish(base_twist)

        # LIMB JOINT (Pitch and Roll)
        limb_twist = Twist()
        
        # Pitch
        if abs(dpitch) > self.deadzone:
            if dpitch < -self.pitch_threshold:
                limb_twist.angular.y = -self.limb_pitch_gain
                self.get_logger().info(f"⬇️  Nod DOWN: {np.degrees(dpitch):.1f}°",
                                      throttle_duration_sec=1.0)
            elif dpitch > self.pitch_threshold:
                limb_twist.angular.y = self.limb_pitch_gain
                self.get_logger().info(f"⬆️  Nod UP: {np.degrees(dpitch):.1f}°",
                                      throttle_duration_sec=1.0)
            else:
                limb_twist.angular.y = (dpitch / self.pitch_threshold) * self.limb_pitch_gain
        
        # Roll
        if abs(droll) > self.deadzone:
            if droll < -self.roll_threshold:
                limb_twist.angular.x = -self.limb_roll_gain
                self.get_logger().info(f"↶  Rotate LEFT: {np.degrees(droll):.1f}°",
                                      throttle_duration_sec=1.0)
            elif droll > self.roll_threshold:
                limb_twist.angular.x = self.limb_roll_gain
                self.get_logger().info(f"↷  Rotate RIGHT: {np.degrees(droll):.1f}°",
                                      throttle_duration_sec=1.0)
            else:
                limb_twist.angular.x = (droll / self.roll_threshold) * self.limb_roll_gain
        
        self.limb_twist_pub.publish(limb_twist)

    def blink_callback(self, msg: Int8):
        """Process blink events."""
        if msg.data == BLINK_LONG:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self.last_grasp_time > self.grasp_cooldown:
                self.get_logger().info("✊ GRASP - Long Blink")
                grasp_msg = Bool()
                grasp_msg.data = True
                self.grasp_pub.publish(grasp_msg)
                self.last_grasp_time = current_time

    def mouth_callback(self, msg: Bool):
        """Process mouth open for emergency stop."""
        if msg.data:
            self.get_logger().warn("🛑 EMERGENCY STOP - Mouth Open")
            
            # Publish stop
            stop_msg = Bool()
            stop_msg.data = True
            self.stop_pub.publish(stop_msg)
            
            # Publish zero velocities
            zero_twist = Twist()
            self.base_twist_pub.publish(zero_twist)
            self.limb_twist_pub.publish(zero_twist)


def main(args=None):
    rclpy.init(args=args)
    node = HeadTeleopMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()