from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='head_teleop',
            executable='head_pose_blink',
            name='head_pose_blink_node',
            output='screen'
        ),
        Node(
            package='head_teleop',
            executable='head_mapper',
            name='head_teleop_mapper',
            output='screen'
        ),
    ])