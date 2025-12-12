#!/usr/bin/env python3
"""
ik.py

Inverse Kinematics planner for UR7e robot using MoveIt services.

Provides IK computation and motion planning capabilities.
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import PositionIKRequest, Constraints, JointConstraint
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from moveit_msgs.msg import RobotState
import sys


class IKPlanner(Node):
    def __init__(self):
        super().__init__('ik_planner')

        # ---- Clients ----
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')

        for srv, name in [(self.ik_client, 'compute_ik'),
                          (self.plan_client, 'plan_kinematic_path')]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for /{name} service...')

    # -----------------------------------------------------------
    # Compute IK for a given (x, y, z) + quat and current robot joint state
    # -----------------------------------------------------------
    def compute_ik(self, current_joint_state, x, y, z,
                   qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """
        Compute inverse kinematics for a target pose.
        
        Args:
            current_joint_state: Current joint state of the robot
            x, y, z: Target position (in base_link frame)
            qx, qy, qz, qw: Target orientation quaternion (default: gripper down)
        
        Returns:
            JointState with IK solution, or None if IK failed
        """
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        
        ik_req = GetPositionIK.Request()
        ik_req.ik_request.ik_link_name = 'wrist_3_link'
        ik_req.ik_request.pose_stamped = pose
        rs = RobotState()
        rs.joint_state = current_joint_state
        ik_req.ik_request.robot_state = rs
        ik_req.ik_request.avoid_collisions = True
        ik_req.ik_request.timeout = Duration(sec=2)
        ik_req.ik_request.group_name = 'ur_manipulator'
        
        future = self.ik_client.call_async(ik_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('IK service failed.')
            return None

        result = future.result()
        if result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(f'IK failed, code: {result.error_code.val}')
            return None

        self.get_logger().info('IK solution found.')
        return result.solution.joint_state

    # -----------------------------------------------------------
    # Plan motion given a desired joint configuration
    # -----------------------------------------------------------
    def plan_to_joints(self, target_joint_state):
        """
        Plan motion to a target joint configuration.
        
        Args:
            target_joint_state: Target joint state
        
        Returns:
            RobotTrajectory if planning succeeds, None otherwise
        """
        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = 'ur_manipulator'
        req.motion_plan_request.allowed_planning_time = 5.0
        req.motion_plan_request.planner_id = "RRTConnectkConfigDefault"

        goal_constraints = Constraints()
        for name, pos in zip(target_joint_state.name, target_joint_state.position):
            goal_constraints.joint_constraints.append(
                JointConstraint(
                    joint_name=name,
                    position=pos,
                    tolerance_above=0.01,
                    tolerance_below=0.01,
                    weight=1.0
                )
            )

        req.motion_plan_request.goal_constraints.append(goal_constraints)
        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('Planning service failed.')
            return None

        result = future.result()
        if result.motion_plan_response.error_code.val != 1:
            self.get_logger().error('Planning failed.')
            return None

        self.get_logger().info('Motion plan computed successfully.')
        return result.motion_plan_response.trajectory

