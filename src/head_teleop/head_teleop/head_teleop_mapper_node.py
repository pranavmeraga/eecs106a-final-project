import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Twist
from std_msgs.msg import Int8
import numpy as np

BLINK_NONE = 0
BLINK_SINGLE = 1
BLINK_DOUBLE = 2
BLINK_LONG = 3


class HeadTeleopMapper(Node):
    def __init__(self):
        super().__init__('head_teleop_mapper')

        self.create_subscription(Vector3, 'head_pose', self.head_cb, 10)
        self.create_subscription(Int8, 'blink_event', self.blink_cb, 10)

        self.vel_pub = self.create_publisher(Twist, 'ee_velocity_cmd', 10)
        self.gripper_pub = self.create_publisher(Int8, 'gripper_cmd', 10)

        self.current_head = Vector3()
        self.coarse_mode = True
        self.gripper_closed = False

        self.kx_coarse, self.ky_coarse, self.kz_coarse = 0.4, 0.4, 0.4
        self.kx_fine, self.ky_fine, self.kz_fine = 0.1, 0.1, 0.1
        self.vmax = 0.2

        self.create_timer(1.0 / 30.0, self.control_loop)

    def head_cb(self, msg):
        self.current_head = msg

    def blink_cb(self, msg):
        code = msg.data
        if code == BLINK_SINGLE:
            self.gripper_closed = not self.gripper_closed
            self.gripper_pub.publish(Int8(data=1 if self.gripper_closed else 0))
            self.get_logger().info(f"Gripper {'closed' if self.gripper_closed else 'opened'}")
        elif code == BLINK_DOUBLE:
            self.current_head = Vector3()
            self.get_logger().warn("EMERGENCY STOP (double blink)")
        elif code == BLINK_LONG:
            self.coarse_mode = not self.coarse_mode
            self.get_logger().info(f"Mode: {'COARSE' if self.coarse_mode else 'FINE'}")

    def control_loop(self):
        dyaw, dpitch, dscale = self.current_head.x, self.current_head.y, self.current_head.z
        kx, ky, kz = (self.kx_coarse, self.ky_coarse, self.kz_coarse) if self.coarse_mode else (self.kx_fine, self.ky_fine, self.kz_fine)

        vx = np.clip(kx * dyaw, -self.vmax, self.vmax)
        vy = np.clip(ky * dscale, -self.vmax, self.vmax)
        vz = np.clip(kz * (-dpitch), -self.vmax, self.vmax)

        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = float(vx), float(vy), float(vz)
        self.vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = HeadTeleopMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()