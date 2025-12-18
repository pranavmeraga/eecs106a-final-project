from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    camera_arg = DeclareLaunchArgument(
        'camera', default_value='0'
    )

    no_mirror_arg = DeclareLaunchArgument(
        'no_mirror', default_value='false'
    )

    # 🔹 MoveIt launch (LAB REQUIRED)
    ur_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur_moveit_config'),
                'launch',
                'ur_moveit.launch.py'
            )
        ),
        launch_arguments={
            'ur_type': 'ur7e',
            'launch_rviz': 'true'
        }.items()
    )

    facemesh_node = Node(
        package='head_teleop',
        executable='facemesh_ur7e_control',
        name='facemesh_ur7e_control_node',
        output='screen',
        arguments=[
            '--camera', LaunchConfiguration('camera'),
        ]
    )

    return LaunchDescription([
        camera_arg,
        no_mirror_arg,
        ur_moveit_launch,      # ✅ MoveIt FIRST
        facemesh_node          # ✅ Your node LAST
    ])
