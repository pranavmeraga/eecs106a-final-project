# Head-Gesture Controlled UR7e Robotic Arm

EE106A Fall 2024 Final Project

## Team


## Overview
Control a UR7e robotic arm using only head movements and eye blinks - no physical controllers needed.

## Control Mapping

| Human Action | Robot Response |
|--------------|----------------|
| Turn head left/right | Move end-effector ±X (lateral) |
| Nod up/down | Move end-effector ±Z (vertical) |
| Move closer/farther | Move end-effector ±Y (forward/back) |
| Single blink | Toggle gripper open/close |
| Double blink | Emergency stop |
| Long blink (0.7s) | Toggle coarse/fine mode |

## Installation

### Local Development (Mac)
```bash
# Clone repo
git clone <your-repo-url>
cd eecs106a-final-project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe numpy
```

### At School (Ubuntu with ROS2)
```bash
# Clone repo
git clone <your-repo-url>
cd eecs106a-final-project

# Build ROS2 package
colcon build --packages-select head_teleop
source install/setup.bash
```

## Usage

### 1. Launch Head Tracking
```bash
ros2 launch head_teleop head_teleop_launch.py
```

### 2. Launch UR7e Driver (separate terminal)
```bash
# Use lab's provided UR driver
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur7e
```

### 3. Check Topics
```bash
# Verify head pose is publishing
ros2 topic echo /head_pose

# Verify velocity commands
ros2 topic echo /ee_velocity_cmd

# List all topics
ros2 topic list
```

## Project Structure
```
eecs106a-final-project/
├── src/
│   └── head_teleop/
│       ├── head_teleop/              # Python package
│       │   ├── __init__.py
│       │   ├── head_pose_blink_node.py
│       │   └── head_teleop_mapper_node.py
│       ├── launch/
│       │   └── head_teleop_launch.py
│       ├── test/
│       │   ├── test_flake8.py
│       │   ├── test_pep257.py
│       │   └── test_copyright.py
│       ├── resource/
│       ├── package.xml
│       ├── setup.py
│       └── setup.cfg
├── .venv/                            # Python virtual env (local only)
└── README.md
```

## Tuning Parameters

### Movement Gains
Edit `head_teleop/head_teleop_mapper_node.py`:
```python
# Coarse mode (fast movement)
self.kx_coarse, self.ky_coarse, self.kz_coarse = 0.4, 0.4, 0.4

# Fine mode (precision)
self.kx_fine, self.ky_fine, self.kz_fine = 0.1, 0.1, 0.1

# Max velocity
self.vmax = 0.2  # m/s
```

### Blink Detection
Edit `head_teleop/head_pose_blink_node.py`:
```python
self.EAR_THRESH = 0.21        # Eye aspect ratio threshold
self.LONG_BLINK_TIME = 0.7    # Long blink duration (seconds)
self.eps_yaw = 0.02           # Yaw deadzone
self.eps_pitch = 0.02         # Pitch deadzone
self.eps_scale = 0.005        # Scale deadzone
```

## Troubleshooting

### Camera not found
```bash
# List available cameras
ls /dev/video*

# Try different camera index (0, 1, 2...)
# Edit head_pose_blink_node.py line 31:
self.cap = cv2.VideoCapture(0)  # Change 0 to 1, 2, etc.
```

### Topics not connecting
```bash
# Check if nodes are running
ros2 node list

# Check topic connections
ros2 topic info /ee_velocity_cmd

# Relay to different topic if needed
ros2 run topic_tools relay /ee_velocity_cmd /servo_node/delta_twist_cmd
```

## Safety
- ⚠️ Always have emergency stop ready (double blink)
- ⚠️ Start with LOW gains, increase gradually
- ⚠️ Keep workspace clear
- ⚠️ Test in simulation first if available

## Acknowledgments
- UC Berkeley EE106A Fall 2025
- MediaPipe for face tracking
- ROS2 community
