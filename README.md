# Head-Gesture Controlled UR7e Robotic Arm

**EE106A Fall 2025 Final Project**

## Team

## Overview
Control a UR7e robotic arm using only head movements and facial gestures - no physical controllers needed. Uses computer vision (MediaPipe) to track facial landmarks and translate head poses into robot commands in real-time.

## Control Mapping

| Human Action | Robot Response | Notes |
|--------------|----------------|-------|
| **Turn head left/right** | Base joint rotation (1st DOF) | Yaw angle ~±9° threshold |
| **Nod up/down** | Limb joint vertical movement (2nd DOF) | Face position ~±20px threshold |
| **Tilt head left/right** | Limb joint forward/backward (3rd DOF) | Roll angle + eye-mouth distance |
| **Long blink (>1s)** | Toggle gripper open/close | Hold eyes closed |
| **Open mouth wide** | **Emergency stop** | Immediate zero velocity |

### Control Behavior
- **Velocity Control**: Robot moves while you hold the gesture. Return to neutral to stop.
- **Large Deadzones**: Stable neutral zone (~3-9° depending on axis) prevents unintended commands
- **Deliberate Movements**: Thresholds require clear, intentional gestures (not micro-movements)
- **Recenter Anytime**: Press 'r' to set current head position as new neutral reference

## Installation

### Prerequisites
```bash
# Python 3.11 or 3.12 (MediaPipe does not support 3.13+)
python3 --version

# ROS2
ros2 --version
```

### Development Setup
```bash
# Create Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe numpy

# Build ROS2 package
cd eecs106a-final-project
colcon build
source install/setup.bash
```

## Usage

### Quick Start: Testing Detection (No Robot)
```bash
# Activate virtual environment
source .venv/bin/activate

# Run standalone preview
python facemesh_preview.py

# Test with external webcam
python facemesh_preview.py --camera 1

# Press 'r' to recenter at comfortable position
# Press 'm' to toggle camera mirroring
# Press 'q' to quit
```

### Full System: Robot Control

#### 1. Launch Head Tracking
```bash
source install/setup.bash
ros2 launch head_teleop head_teleop_launch.py
```

#### 2. Launch UR7e Driver (separate terminal)
```bash
# Use lab's provided UR driver

#### 3. Verify System
```bash
# Check if nodes are running
ros2 node list
# Expected output:
#   /head_pose_blink_node
#   /head_teleop_mapper_node

# Verify head pose publishing
ros2 topic echo /head_pose

# Verify robot commands
ros2 topic echo /base_joint_cmd
ros2 topic echo /limb_joint_cmd

# List all active topics
ros2 topic list
```

#### 4. Topic Remapping (if needed)
```bash
# If robot uses different topic names, remap:
ros2 run topic_tools relay /base_joint_cmd /ur_driver/base_cmd
ros2 run topic_tools relay /limb_joint_cmd /ur_driver/limb_cmd
```

## Project Structure
```
eecs106a-final-project/
├── src/
│   └── head_teleop/                  # ROS2 package
│       ├── head_teleop/              # Python modules
│       │   ├── __init__.py
│       │   ├── head_pose_blink_node.py      # Vision processing
│       │   └── head_teleop_mapper_node.py   # Command mapping
│       ├── launch/
│       │   └── head_teleop_launch.py        # Launch file
│       ├── test/
│       │   ├── test_flake8.py
│       │   ├── test_pep257.py
│       │   └── test_copyright.py
│       ├── resource/
│       ├── package.xml              # ROS2 dependencies
│       ├── setup.py                 # Python package setup
│       └── setup.cfg
├── facemesh_preview.py              # Standalone testing tool
├── .venv/                           # Python virtual env (gitignored)
└── README.md
```

## Tuning Parameters

### Detection Thresholds
Edit `facemesh_preview.py` (lines 112-124) or `head_pose_blink_node.py`:
```python
# Movement detection thresholds
YAW_THRESHOLD = 0.15      # ~8.6° - turn left/right
PITCH_THRESHOLD = 20.0    # 20 pixels - nod up/down
ROLL_THRESHOLD = 0.15     # ~8.6° - tilt left/right
TILT_THRESHOLD = 10.0     # 10 pixels - tilt detection

# Deadzones (stable neutral zone)
YAW_DEADZONE = 0.06       # ~3.4° - turn stability
PITCH_DEADZONE = 8.0      # 8 pixels - nod stability
ROLL_DEADZONE = 0.06      # ~3.4° - tilt stability
TILT_DEADZONE = 5.0       # 5 pixels - tilt stability
```

### Robot Control Gains
Edit `head_teleop_mapper_node.py` (lines 48-54):
```python
# Detection thresholds (radians)
self.yaw_threshold = 0.08
self.pitch_threshold = 0.08
self.roll_threshold = 0.08
self.deadzone = 0.05

# Velocity control gains
self.base_yaw_gain = 0.5     # Base rotation speed
self.limb_pitch_gain = 0.5   # Limb vertical speed
self.limb_roll_gain = 0.5    # Limb forward/back speed
```

### Blink & Mouth Detection
Edit `head_pose_blink_node.py`:
```python
# Eye detection
EAR_THRESH = 0.21         # Eye aspect ratio threshold
GRASP_BLINK_TIME = 1.0    # Long blink duration (seconds)

# Mouth detection
MOUTH_OPEN_THRESH = 0.03  # Mouth open threshold (relative)
```

### Smoothing
Edit filter alpha values (lines 107-111 in `facemesh_preview.py`):
```python
# Lower alpha = more smoothing, less responsive
yaw_filter = SmoothingFilter(alpha=0.4)     # Default: 0.4
pitch_filter = SmoothingFilter(alpha=0.4)
roll_filter = SmoothingFilter(alpha=0.4)
```

## Troubleshooting

### Camera Issues
```bash
# Test different camera indices (Local)
python facemesh_preview.py --camera 0  # Built-in
python facemesh_preview.py --camera 1  # External
```

### Face Not Detected
- Ensure good lighting (not backlit)
- Position face 1-3 feet from camera
- Check camera is not blocked
- Try toggling mirror mode with 'm'

### Commands Always Active (No Neutral Zone)
- **Press 'r' to recenter** at comfortable position
- Increase deadzones in code if needed
- Check absolute angle values on screen - neutral should be stable

### Topics Not Connecting
```bash
# Verify nodes are running
ros2 node list

# Check topic producers/consumers
ros2 topic info /base_joint_cmd

# Debug topic data
ros2 topic echo /head_pose --no-arr

# Check for topic name mismatches
ros2 topic list | grep cmd
```

### MediaPipe Python Version Error
```bash
# MediaPipe only supports Python 3.8-3.12
python3 --version

# Create venv with correct Python version
python3.12 -m venv .venv
source .venv/bin/activate
pip install opencv-python mediapipe numpy
```

### Robot Not Responding
```bash
# Verify UR driver is running
ros2 node list | grep ur

# Test with manual command
ros2 topic pub /base_joint_cmd geometry_msgs/msg/Twist \
    "{angular: {z: 0.1}}" --once

# Check emergency stop is not active
ros2 topic echo /stop_cmd
```

## Development Notes

### Algorithm Details
- **Head Pose**: Uses OpenCV `solvePnP` with 6 stable facial landmarks
- **Nod Detection**: Tracks vertical position of face center (nose, chin, forehead average)
- **Tilt Detection**: Combines roll angle + eye-mouth distance ratio
- **Turn Detection**: Uses yaw angle from rotation matrix
- **Smoothing**: Exponential moving average filter (α=0.4)

### ROS2 Topics
**Published by `head_pose_blink_node`:**
- `/head_pose` (Vector3): x=yaw, y=pitch, z=roll deltas
- `/blink_event` (Int8): 0=none, 1=long_blink
- `/mouth_open` (Bool): True if mouth open

**Published by `head_teleop_mapper_node`:**
- `/base_joint_cmd` (Twist): Base rotation commands
- `/limb_joint_cmd` (Twist): Limb movement commands
- `/grasp_cmd` (Bool): Gripper toggle
- `/stop_cmd` (Bool): Emergency stop

## Safety

⚠️ **Critical Safety Rules:**
- Always have **emergency stop ready** (mouth open also stop)
- **Start with LOW gains**, increase gradually after testing
- **Keep workspace clear** of obstacles and people
- **Never leave robot unattended** during head-controlled operation
- Use **teach pendant emergency stop** as backup
- Establish **safe home position** before starting
- Define **workspace limits** to prevent collisions

## Performance Tips

1. **Lighting**: Use consistent, front-facing light (avoid backlighting)
2. **Distance**: Stay 2-3 feet from camera for best tracking
3. **Deliberate Movements**: Make clear, intentional gestures (not micro-movements)
4. **Recenter Often**: Press 'r' when changing posture or position
5. **Practice**: Test with preview tool before controlling robot
6. **Gains**: Start at 0.1-0.2, increase to 0.5 after comfortable

## Known Limitations

- Requires good lighting conditions
- Single user only (cannot track multiple faces)
- Requires frontal face view (±45° max rotation)
- Glasses/masks may affect detection
- Latency: ~50-100ms (camera fps + processing)
t
## Acknowledgments

- **UC Berkeley EE106A Fall 2025** - Course framework
- **MediaPipe** (Google) - Face mesh tracking library
- **ROS2** - Robot operating system
- **OpenCV** - Computer vision library

## License

## Contact
