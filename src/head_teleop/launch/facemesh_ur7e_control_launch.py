from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for Facemesh UR7e Control Node.
    
    This launches the facemesh control node which displays a live camera feed
    with face detection overlay in an OpenCV GUI window.
    
    The GUI shows:
    - Live camera feed with face mesh overlay
    - Real-time head movement deltas (dNod, dTurn, dTilt)
    - Status indicators (OK, Emergency Stop, Long Blink)
    - Eye aspect ratio (EAR) and mouth status
    - Control instructions
    
    Usage:
        ros2 launch head_teleop facemesh_ur7e_control_launch.py
        ros2 launch head_teleop facemesh_ur7e_control_launch.py camera:=1
        ros2 launch head_teleop facemesh_ur7e_control_launch.py camera:=1 no_mirror:=true
    """
    
    # Declare launch arguments
    camera_arg = DeclareLaunchArgument(
        'camera',
        default_value='0',
        description='Camera index (0 for default, 1 for external camera like Logitech)'
    )
    
    no_mirror_arg = DeclareLaunchArgument(
        'no_mirror',
        default_value='false',
        description='Disable camera mirroring (true/false)'
    )
    
    # Get launch configurations
    camera_index = LaunchConfiguration('camera')
    no_mirror = LaunchConfiguration('no_mirror')
    
    def create_node(context):
        # Build arguments list
        node_args = ['--camera', context.launch_configurations['camera']]
        
        # Add --no-mirror flag if requested
        if context.launch_configurations.get('no_mirror', 'false').lower() == 'true':
            node_args.append('--no-mirror')
        
        # Facemesh UR7e Control Node
        # This node automatically displays the live camera feed in an OpenCV window
        facemesh_control_node = Node(
            package='head_teleop',
            executable='facemesh_ur7e_control',
            name='facemesh_ur7e_control_node',
            output='screen',
            arguments=node_args,
            parameters=[],
            remappings=[
                # Ensure proper topic remappings if needed
                ('/joint_states', '/joint_states'),
                ('/scaled_joint_trajectory_controller/joint_trajectory', 
                 '/scaled_joint_trajectory_controller/joint_trajectory'),
            ]
        )
        return [facemesh_control_node]
    
    return LaunchDescription([
        camera_arg,
        no_mirror_arg,
        OpaqueFunction(function=create_node),
    ])

