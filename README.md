# Head-Gesture Controlled UR7e Robotic Arm

**EE106A Fall 2025 Final Project**

## Team
- Pranav Meraga
- Loveveer Singh
- Yutong Bian
- Alan Li

## Overview
Control a UR7e robotic arm using only head movements and facial gestures - no physical controllers needed. Uses computer vision (MediaPipe) to track facial landmarks and translate head poses into robot commands in real-time. Features include dual control modes, intelligent waypoint saving with IK-based trajectory planning via MoveIt, and comprehensive safety controls.

## Control Mapping

The system supports **two control modes** that can be toggled during operation:

### MODE 1: Shoulder Control (Default)
| Human Action | Robot Response | Detection Method |
|--------------|----------------|------------------|
| **Turn head left/right** | `shoulder_pan_joint` rotation | Mouth horizontal position |
| **Nod up/down** | `shoulder_lift_joint` movement | Face vertical position |
| **Open mouth (hold >1s) - 1st time** | Save current pose as waypoint | Mouth opening duration |
| **Open mouth (hold >1s) - subsequent** | Return to saved waypoint via IK/MoveIt | Mouth opening duration |

### MODE 2: Wrist Control
| Human Action | Robot Response | Detection Method |
|--------------|----------------|------------------|
| **Turn head left/right** | `wrist_2_joint` (rotate gripper) | Mouth horizontal position |
| **Nod up/down** | `shoulder_lift_joint` + `elbow_joint` (extend/retract arm) | Face vertical position |
| **Automatic compensation** | `wrist_1_joint` maintains 90° gripper angle | Calculated automatically |
| **Open mouth (hold >1s)** | Gripper toggle (open/close) | Mouth opening duration |

### Control Actions (Both Modes)
| Action | Function | Detection Method |
|--------|----------|------------------|
| **Both eyes blink (hold >1s)** | Toggle between MODE 1 and MODE 2 | Both eyes closed simultaneously |
| **Left eye blink (hold >1s)** | Emergency stop (toggle on/off) | Left eye closed only |
| **Right eye blink (hold >1s)** | Recenter neutral pose | Right eye closed only |

### Control Behavior
- **Incremental Control**: Robot joints adjust continuously based on head movements from neutral pose
- **Responsive Deadzones**: Small deadzones (2px) for responsive control while preventing drift
- **Independent Detection**: Each motion type uses distinct measurement (mouth X, face Y)
- **Mode Toggle**: Switch between shoulder and wrist control modes with both-eyes blink
- **Recenter Anytime**: Right eye blink (hold >1s) to set current head position as new neutral reference
- **Visual Feedback**: OpenCV GUI window shows live camera feed with control status and mode indicator

### IK-Based Waypoint Saving (MODE 1 Feature)
In MODE 1, the system includes an intelligent waypoint saving and returning feature powered by MoveIt:
- **First mouth open (hold >1s)**: Saves the current end-effector pose as a waypoint
- **Subsequent mouth opens (hold >1s)**: Automatically returns to the saved waypoint using:
  - **Inverse Kinematics (IK)**: Computes joint angles needed to reach the saved pose
  - **Motion Planning**: Uses MoveIt with RRTConnect planner to generate collision-free trajectory
  - **Smooth Execution**: Returns to waypoint with time-parameterized joint trajectory

This feature allows users to quickly return to important positions during manipulation tasks, reducing redundant movements and improving task efficiency. The IK solution is anchored to the current joint state and validated against joint limits before execution.

### How Each Motion Works
- **TURN**: Tracks horizontal position of mouth center - mouth moves left/right in frame
- **NOD**: Tracks vertical position of entire face (nose, chin, forehead) - whole face moves up/down in frame
- **MODE TOGGLE**: Both eyes closed simultaneously for >1s switches between shoulder and wrist control
- **EMERGENCY STOP**: Left eye closed for >1s toggles emergency stop (holds current position)
- **RECENTER**: Right eye closed for >1s resets neutral pose to current head position

## Installation

### Prerequisites
```bash
# Python 3.11 or 3.12 (MediaPipe does not support 3.13+)
python3 --version

# ROS2
ros2 --version

# MoveIt (required for IK planning and waypoint return)
# Should be installed with your ROS2 workspace
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

**Visual Feedback in Preview:**
- **GREEN values** = In deadzone (stable neutral)
- **YELLOW values** = Moving but below threshold
- **RED values** = Command active
- **Blue circles** = Eyes (for tilt detection)
- **Green circles** = Mouth corners (for turn detection)
- **Yellow circles** = Face center (for nod detection)

### Full System: Robot Control

**Note**: The facemesh control launch file automatically starts MoveIt for IK planning. If using the older head_teleop_launch.py, you'll need to launch MoveIt separately.

#### 1. Launch Head Tracking (Legacy Method)
```bash
source install/setup.bash
ros2 launch head_teleop head_teleop_launch.py
```

#### 2. Launch UR7e Driver (separate terminal)
```bash
# Use lab's provided UR driver
```

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

### Direct UR7e Joint Control with Facemesh

This method directly controls UR7e joints using facemesh detection, similar to the keyboard controller but with head movements.

#### Prerequisites
1. **Enable UR7e Communication** (in a separate terminal):
   ```bash
   ros2 run ur7e_utils enable_comms
   ```
   Keep this terminal running to maintain connection with UR7e.

2. **Tuck the Robot** (optional, for safe starting position):
   ```bash
   ros2 run ur7e_utils tuck
   ```

#### Running the Facemesh Control Node

1. **Build the package** (if not already built):
   ```bash
   cd /path/to/eecs106a-final-project
   colcon build
   source install/setup.bash
   ```

2. **Run the facemesh control node** (choose one method):

   **Method A: Using launch file (recommended)**
   ```bash
   # Using default camera - shows live GUI with mode indicator
   ros2 launch head_teleop facemesh_ur7e_control_launch.py
   
   # Using a different camera (e.g., Logitech camera at index 1)
   ros2 launch head_teleop facemesh_ur7e_control_launch.py camera:=1
   
   # Disable camera mirroring
   ros2 launch head_teleop facemesh_ur7e_control_launch.py camera:=1 no_mirror:=true
   ```

   **Method B: Direct node execution**
   ```bash
   # Using default camera
   ros2 run head_teleop facemesh_ur7e_control
   
   # Using a different camera (e.g., Logitech camera at index 1)
   ros2 run head_teleop facemesh_ur7e_control --camera 1
   
   # Disable camera mirroring
   ros2 run head_teleop facemesh_ur7e_control --no-mirror
   ```

3. **Control the robot**:
   - The node will automatically open an **OpenCV GUI window** showing:
     - **Live camera feed** with face mesh overlay
     - **Current control mode** (MODE 1: SHOULDER or MODE 2: WRIST) with color indicator
     - **Real-time head movement deltas** (dNod, dTurn) with color coding
     - **Eye status** (Left, Right, Both) with blink hold timers
     - **Mouth status** and grasp toggle indicator
     - **Control instructions** and keyboard shortcuts
   
   **MODE 1 (Shoulder Control) - Default:**
   - **Turn head left/right** → Controls `shoulder_pan_joint` (base rotation)
   - **Nod up/down** → Controls `shoulder_lift_joint` (vertical movement)
   - **Open mouth (hold >1s) - 1st time** → Save current pose as waypoint
   - **Open mouth (hold >1s) - subsequent** → Return to saved waypoint via IK/MoveIt
   
   **MODE 2 (Wrist Control):**
   - **Turn head left/right** → Controls `wrist_2_joint` (rotate gripper)
   - **Nod up/down** → Controls `shoulder_lift_joint` + `elbow_joint` (extend/retract arm)
   - **Automatic wrist_1 compensation** → Maintains 90° gripper angle
   - **Open mouth (hold >1s)** → Gripper toggle (open/close)
   
   **Control Actions:**
   - **Both eyes blink (hold >1s)** → Toggle between MODE 1 and MODE 2 (beep sound confirms)
   - **Left eye blink (hold >1s)** → Emergency stop toggle (holds current position)
   - **Right eye blink (hold >1s)** → Recenter neutral pose
   - **Press 'q'** in the OpenCV window → Quit the node

#### Control Mapping Details

**MODE 1 (Shoulder Control):**
| Head Movement | Joint Controlled | Detection Method |
|---------------|------------------|------------------|
| Turn left/right | `shoulder_pan_joint` | Mouth horizontal position |
| Nod up/down | `shoulder_lift_joint` | Face vertical position |
| Open mouth (hold >1s) - 1st | Save waypoint | Mouth opening duration |
| Open mouth (hold >1s) - subsequent | Return to waypoint via IK | Mouth opening duration |

**MODE 2 (Wrist Control):**
| Head Movement | Joint Controlled | Detection Method |
|---------------|------------------|------------------|
| Turn left/right | `wrist_2_joint` | Mouth horizontal position |
| Nod up/down | `shoulder_lift_joint` + `elbow_joint` | Face vertical position |
| Automatic | `wrist_1_joint` compensation | Calculated to maintain 90° angle |
| Open mouth (hold >1s) | Gripper toggle (open/close) | Mouth opening duration |

**Control Actions (Both Modes):**
| Action | Function | Detection Method |
|--------|----------|------------------|
| Both eyes blink (hold >1s) | Toggle MODE 1 ↔ MODE 2 | Both eyes closed simultaneously |
| Left eye blink (hold >1s) | Emergency stop toggle | Left eye closed only |
| Right eye blink (hold >1s) | Recenter neutral pose | Right eye closed only |

#### Safety Notes

⚠️ **IMPORTANT SAFETY WARNINGS:**
- **DO NOT input random angles** - This will cause the robot to emergency stop
- Always start with the robot in a **safe tuck position**
- The node uses **incremental control** - it adjusts joint positions based on head movements from a neutral pose
- **Left eye blink (hold >1s)** to toggle emergency stop at any time
- Keep the **e-stop button** accessible at all times
- The node publishes trajectories with `time_from_start.sec = 5` for smooth motion
- **Mode switching**: Use both-eyes blink carefully - beep sound confirms mode change
- **Recenter often**: Use right-eye blink to recenter when changing posture

#### Testing with Safe Joint Angles

After running `ros2 run ur7e_utils tuck`, the robot will be in this safe test position:
```bash
# Safe tucked position (for reference):
# shoulder_pan: 4.1768, shoulder_lift: -2.2087, elbow: -1.2924
# wrist_1: -1.1133, wrist_2: 1.4865, wrist_3: -2.8460
# Note: The facemesh node controls joints incrementally from this position
```

#### Differences from Keyboard Controller

- **Incremental control**: Facemesh node adjusts joints incrementally based on head movements, while keyboard controller moves joints by fixed steps
- **Continuous control**: Head movements provide smooth, continuous control vs. discrete key presses
- **Dual modes**: Two control modes (shoulder vs wrist) that can be toggled during operation
- **IK planning**: MODE 1 includes waypoint saving/returning using MoveIt for automated trajectory planning
- **Safety features**: Built-in emergency stop toggle via left-eye blink
- **Visual feedback**: Live GUI window with mode indicator and status information

## Project Structure
```
eecs106a-final-project/
├── src/
│   └── head_teleop/                  # ROS2 package
│       ├── head_teleop/              # Python modules
│       │   ├── __init__.py
│       │   ├── head_pose_blink_node.py      # Vision processing
│       │   ├── head_teleop_mapper_node.py   # Command mapping
│       │   └── facemesh_ur7e_control_node.py # Direct UR7e joint control
│       ├── launch/
│       │   ├── head_teleop_launch.py        # Launch file for head teleop
│       │   └── facemesh_ur7e_control_launch.py # Launch file for UR7e control with GUI
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
Edit `facemesh_preview.py` (lines 125-135):
```python
# CLEAR THRESHOLDS - each motion is independent
TURN_THRESHOLD = 25.0     # pixels - mouth horizontal movement (TURN)
NOD_THRESHOLD = 25.0      # pixels - face vertical movement (NOD)
TILT_THRESHOLD = 0.12     # radians ~6.9° - eye line angle (TILT)

# LARGE DEADZONES for stable neutral
TURN_DEADZONE = 10.0      # pixels - stable center for turn
NOD_DEADZONE = 10.0       # pixels - stable center for nod
TILT_DEADZONE = 0.05      # radians ~2.9° - stable center for tilt
```

### Robot Control Gains
Edit `facemesh_ur7e_control_node.py`:
```python
# Detection thresholds (pixels)
self.TURN_THRESHOLD = 18.0     # pixels
self.NOD_THRESHOLD = 18.0      # pixels

# Deadzones (pixels)
self.TURN_DEADZONE = 2.0       # pixels - responsive for turning
self.NOD_DEADZONE = 2.0        # pixels - responsive for nodding

# Control gains for MODE 1 (shoulder control)
self.yaw_to_pan_gain = 0.15    # radians per pixel (turn → shoulder_pan)
self.pitch_to_lift_gain = 0.15  # radians per pixel (nod → shoulder_lift)

# Control gains for MODE 2 (wrist control)
self.yaw_to_wrist2_gain = 1.0   # radians per pixel (turn → wrist_2_joint)
self.pitch_to_wrist1_gain = 1.0 # radians per pixel (nod → wrist_1_joint)
```

### Blink & Mouth Detection
Edit `facemesh_ur7e_control_node.py`:
```python
# Eye detection
self.EAR_THRESH = 0.21              # Eye aspect ratio threshold
self.BLINK_HOLD_TIME = 1.0          # Blink hold duration (seconds)
self.MODE_TOGGLE_HOLD_TIME = 1.0    # Mode toggle hold time (seconds)

# Mouth detection
self.MOUTH_OPEN_THRESH = 0.03       # Mouth open threshold (relative)
self.MOUTH_HOLD_TIME = 1.0          # Mouth hold time for grasp toggle (seconds)
```

### Smoothing
Edit filter alpha values (lines 107-109 in `facemesh_preview.py`):
```python
# Lower alpha = more smoothing, less responsive
face_y_filter = SmoothingFilter(alpha=0.3)    # Nod detection
mouth_x_filter = SmoothingFilter(alpha=0.3)   # Turn detection
tilt_filter = SmoothingFilter(alpha=0.3)      # Tilt detection
```

## Troubleshooting

### Camera Issues
```bash
# Test different camera indices
python facemesh_preview.py --camera 0  # Built-in
python facemesh_preview.py --camera 1  # External
```

### Face Not Detected
- Ensure good lighting (not backlit)
- Position face 1-3 feet from camera
- Check camera is not blocked
- Try toggling mirror mode with 'm'

### Commands Always Active (No Neutral Zone)
- **Right-eye blink (hold >1s) to recenter** at comfortable position
- Check all delta values show **0.00** and **GREEN** after recentering
- The deadzones are set to 2px for responsive control
- Watch the GUI overlay to verify tracking and mode status

### Mode Toggle Not Working
- **Both eyes must be closed simultaneously** for >1s to toggle modes
- Listen for beep sound to confirm mode change
- Check GUI window shows correct mode (MODE 1: SHOULDER or MODE 2: WRIST)
- Make sure you're not accidentally triggering single-eye blinks

### Control Not Responsive
- The system uses small deadzones (2px) for responsive control
- Make deliberate head movements - small micro-movements may not register
- Recenter with right-eye blink if control feels off
- Check that emergency stop is not active (left-eye blink to toggle)

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
python3.11 -m venv .venv
source .venv/bin/activate
pip install opencv-python mediapipe numpy
```

### Robot Not Responding
```bash
# Verify UR driver is running
ros2 node list | grep ur

# Check if robot is in remote control mode
# (Check teach pendant)

# Test with manual command
ros2 topic pub /base_joint_cmd geometry_msgs/msg/Twist \
    "{angular: {z: 0.1}}" --once

# Check emergency stop is not active
ros2 topic echo /stop_cmd
```

### Camera Not Found
- Check camera index: `ls /dev/video*` (Linux) or use system camera settings
- Try different indices: `--camera 0`, `--camera 1`, etc.
- For Logitech cameras, typically index 0 or 1

### Robot Not Moving
- Verify `enable_comms` is running
- Check joint states: `ros2 topic echo /joint_states`
- Verify trajectory publishing: `ros2 topic echo /scaled_joint_trajectory_controller/joint_trajectory`
- Make sure you've recentered with right-eye blink after starting
- Check if emergency stop is active (left-eye blink to toggle)
- Verify you're in the correct control mode (both-eyes blink to toggle)

### Waypoint Return Not Working (MODE 1)
```bash
# Verify MoveIt services are available
ros2 service list | grep compute_ik
ros2 service list | grep plan_kinematic_path

# Check if MoveIt is running
ros2 node list | grep move_group

# Test IK service manually
ros2 service call /compute_ik moveit_msgs/srv/GetPositionIK "{...}"
```

**Common issues:**
- MoveIt not launched before control node
- Saved waypoint is unreachable (out of workspace or in collision)
- Joint limits prevent IK solution
- Try saving waypoint in a different position within the robot's workspace

## Development Notes

### Algorithm Details
- **Turn Detection**: Tracks horizontal position (X coordinate) of mouth center
- **Nod Detection**: Tracks vertical position (Y coordinate) of face center (nose, chin, forehead average)
- **Eye Blink Detection**: Separate tracking for left eye, right eye, and both eyes using Eye Aspect Ratio (EAR)
- **Mouth Detection**: Tracks mouth opening distance for waypoint control (MODE 1) or gripper toggle (MODE 2)
- **Mode Toggle**: Both eyes closed simultaneously for >1s triggers mode switch with audio feedback
- **Smoothing**: Exponential moving average filter for stable readings
- **Control Modes**: Two distinct control modes with different joint mappings and automatic wrist compensation in MODE 2
- **IK Planning (MODE 1)**: 
  - Constructs PoseStamped target in base_link frame
  - Anchors IK solution using current joint state
  - Requests collision-aware IK solution via `/compute_ik`
  - Validates result against joint limits
  - Uses MoveIt RRTConnect planner for trajectory generation
  - Returns time-parameterized joint trajectory for smooth waypoint return

### Key Landmarks Used
- **Eyes (landmarks 33, 263)**: For tilt angle calculation
- **Mouth corners (landmarks 61, 291)**: For turn detection
- **Nose tip (landmark 1)**: For face center (nod detection)
- **Chin (landmark 152)**: For face center (nod detection)
- **Forehead (landmark 10)**: For face center (nod detection)

### ROS2 Topics & Services
**Published by `head_pose_blink_node`:**
- `/head_pose` (Vector3): x=yaw, y=pitch, z=roll deltas
- `/blink_event` (Int8): 0=none, 1=long_blink
- `/mouth_open` (Bool): True if mouth open

**Published by `head_teleop_mapper_node`:**
- `/base_joint_cmd` (Twist): Base rotation commands
- `/limb_joint_cmd` (Twist): Limb movement commands
- `/grasp_cmd` (Bool): Gripper toggle
- `/stop_cmd` (Bool): Emergency stop

**Published by `facemesh_ur7e_control_node`:**
- `/scaled_joint_trajectory_controller/joint_trajectory` (JointTrajectory): Direct joint position commands for UR7e

**Subscribed by `facemesh_ur7e_control_node`:**
- `/joint_states` (JointState): Current joint positions from UR7e

**MoveIt Services Used (for IK Planning in MODE 1):**
- `/compute_ik` (ComputeIK): Computes inverse kinematics for saved waypoint
- `/plan_kinematic_path` (GetMotionPlan): Plans collision-free trajectory to waypoint

## Safety

⚠️ **Critical Safety Rules:**
- Always have **emergency stop ready** (left-eye blink to toggle)
- **Start with LOW gains**, increase gradually after testing
- **Keep workspace clear** of obstacles and people
- **Never leave robot unattended** during head-controlled operation
- Use **teach pendant emergency stop** as backup
- Establish **safe home position** before starting
- Define **workspace limits** to prevent collisions
- **Mode switching**: Be aware of current control mode (check GUI indicator)
- **Practice mode switching** in safe position before using during operation
- **Waypoint safety**: Only save waypoints in collision-free, reachable positions within the robot's workspace

## Performance Tips

1. **Lighting**: Use consistent, front-facing light (avoid backlighting)
2. **Distance**: Stay 2-3 feet from camera for best tracking
3. **Deliberate Movements**: Make clear, intentional gestures (not micro-movements)
4. **Recenter Often**: Right-eye blink when changing posture or position
5. **Practice**: Test with preview tool before controlling robot
6. **Mode Awareness**: Always check GUI to know which control mode is active
7. **Blink Technique**: For mode toggle, close both eyes simultaneously and hold for >1s
8. **Gains**: Control gains are optimized for each mode (0.15 for shoulder, 1.0 for wrist)
9. **Waypoint Strategy**: In MODE 1, save waypoints at frequently used positions to reduce repetitive manual control and improve efficiency

## Known Limitations

- Requires good lighting conditions
- Single user only (cannot track multiple faces)
- Requires frontal face view (±45° max rotation)
- Glasses/masks may affect detection
- Latency: ~50-100ms (camera fps + processing)

## Acknowledgments

- **UC Berkeley EE106A Fall 2025** - Course framework
- **MediaPipe** (Google) - Face mesh tracking library
- **ROS2** - Robot operating system
- **MoveIt** - Motion planning framework for IK and trajectory generation
- **OpenCV** - Computer vision library

## Contact

For questions or issues, please contact the team members or create an issue in the project repository.