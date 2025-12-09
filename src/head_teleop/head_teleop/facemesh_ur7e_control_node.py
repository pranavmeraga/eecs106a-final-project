#!/usr/bin/env python3
"""facemesh_ur7e_control_node.py

Controls UR7e robot joints using facemesh head pose detection.
Combines facemesh detection with joint trajectory publishing.

Subscribes:
- /joint_states (JointState): Current joint positions

Publishes:
- /scaled_joint_trajectory_controller/joint_trajectory (JointTrajectory): Joint commands

Control Mapping:
MODE 1 (Shoulder Control):
- Turn head left/right (yaw) → shoulder_pan_joint
- Nod up/down (pitch) → shoulder_lift_joint
- Open mouth (hold >1s) → Grasp toggle (alternate grasp/open)

MODE 2 (Wrist Control):
- Turn head left/right (yaw) → wrist_2_joint (rotate gripper left/right)
- Nod up/down (pitch) → shoulder_lift_joint + elbow_joint (extend/retract arm)
- Automatic wrist_1_joint compensation to keep gripper at 90° to plane
- Open mouth (hold >1s) → Grasp toggle (alternate grasp/open)

Control Actions:
- Both eyes blink (hold >1s) → Toggle between MODE 1 and MODE 2
- Left eye blink (hold >1s) → Emergency stop (toggle)
- Right eye blink (hold >1s) → Recenter neutral pose
"""

import cv2
import time
import numpy as np
import mediapipe as mp
import argparse
import wave
import os
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Trigger


# Facemesh landmarks
POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14

# Additional landmarks for better detection
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291


class SmoothingFilter:
    """Exponential moving average filter."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        self.value = None


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """Calculate eye aspect ratio (EAR)."""
    p1, p2, p3, p4, p5, p6 = eye_pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    return (v1 + v2) / (2.0 * h + 1e-6)


def calculate_face_center_y(lm, h):
    """Calculate vertical center of face for NOD detection."""
    nose_y = lm[NOSE_TIP].y * h
    chin_y = lm[CHIN].y * h
    forehead_y = lm[FOREHEAD].y * h
    center_y = (nose_y + chin_y + forehead_y) / 3.0
    return center_y


def calculate_mouth_horizontal_position(lm, w):
    """Calculate horizontal center of mouth for TURN detection."""
    left_mouth_x = lm[LEFT_MOUTH_CORNER].x * w
    right_mouth_x = lm[RIGHT_MOUTH_CORNER].x * w
    mouth_center_x = (left_mouth_x + right_mouth_x) / 2.0
    return mouth_center_x


def rotation_matrix_to_euler_angles(R: np.ndarray) -> tuple:
    """Convert rotation matrix to Euler angles."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


class FacemeshUR7eControlNode(Node):
    def __init__(self, camera_index: int = 0, mirror: bool = True):
        super().__init__('facemesh_ur7e_control_node')
        
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.joint_positions = [0.0] * 6
        self.got_joint_states = False  # Track if we've received joint states
        self.joint_states_timeout = time.time() + 5.0  # Wait max 5 seconds for joint states
        
        # Subscribe to joint states
        self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            10
        )
        
        self.get_logger().info("Subscribed to /joint_states topic")
        self.get_logger().info("Publishing to /scaled_joint_trajectory_controller/joint_trajectory")
        
        # Publisher for joint trajectory
        self.pub = self.create_publisher(
            JointTrajectory, 
            '/scaled_joint_trajectory_controller/joint_trajectory', 
            10
        )
        
        # Gripper service client
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f"Unable to open camera index {camera_index}")
            raise RuntimeError(f"Camera failed to open: {camera_index}")
        
        self.mirror = mirror
        
        # Neutral pose tracking
        self.neutral_set = False
        self.face_center_y0 = 0.0
        self.mouth_center_x0 = 0.0
        
        # Smoothing filters
        self.face_y_filter = SmoothingFilter(alpha=0.3)
        self.mouth_x_filter = SmoothingFilter(alpha=0.3)
        
        # Blink tracking - left eye, right eye, and both eyes for mode toggle
        self.left_eye_closed = False
        self.left_eye_close_start = 0.0
        self.right_eye_closed = False
        self.right_eye_close_start = 0.0
        self.both_eyes_closed = False
        self.both_eyes_close_start = 0.0
        self.EAR_THRESH = 0.21  # Eye aspect ratio threshold
        self.BLINK_HOLD_TIME = 1.0  # Hold blink for 1 second to trigger action
        self.MODE_TOGGLE_HOLD_TIME = 1.0  # Hold both eyes closed for 1s to toggle mode
        self.left_blink_triggered = False  # Track if we've already triggered left blink
        self.right_blink_triggered = False  # Track if we've already triggered right blink
        self.mode_toggle_triggered = False  # Track if we've already triggered mode toggle
        
        # Mouth tracking for grasp
        self.MOUTH_OPEN_THRESH = 0.03
        self.mouth_was_open = False  # Track previous mouth state for toggle
        self.grasp_toggled = False  # Track if we've already triggered grasp toggle
        self.grasp_open_start = 0.0  # Track when mouth opens
        
        # Thresholds - same as facemesh_preview.py
        self.TURN_THRESHOLD = 18.0     # pixels
        self.NOD_THRESHOLD = 18.0      # pixels
        
        # Deadzones - reduced for more responsive control
        self.TURN_DEADZONE = 2.0       # pixels - very responsive for turning
        self.NOD_DEADZONE = 2.0        # pixels - very responsive for nodding
        
        # Control gains for MODE 1 (shoulder control)
        self.yaw_to_pan_gain = 0.15    # radians per pixel (turn → shoulder_pan)
        self.pitch_to_lift_gain = 0.15  # radians per pixel (nod → shoulder_lift)
        self.pitch_to_wrist_gain = 1.0  # radians per pixel (nod → wrist_1 counteraction)
        
        # Control gains for MODE 2 (wrist control)
        self.yaw_to_wrist2_gain = 1.5   # radians per pixel (turn → wrist_2_joint) - faster
        self.pitch_to_wrist1_gain = 1.5 # radians per pixel (nod → wrist_1_joint) - faster
        
        # Max joint velocity limits (radians)
        self.max_joint_velocity = 0.5
        
        # Emergency stop flag (toggle state)
        self.emergency_stop = False
        
        # Control mode flag: False = MODE 1 (shoulder), True = MODE 2 (wrist)
        self.control_mode_2 = False
        
        # Quit flag
        self.should_quit = False
        
        # Display flag to track if window has been created
        self.window_created = False
        
        # Timer for camera loop
        self.timer = self.create_timer(1.0 / 60.0, self.camera_loop)  # 60 Hz for faster response
        
        self.get_logger().info("Facemesh UR7e Control Node Started")
        self.get_logger().info(f"Camera index: {camera_index}, Mirror: {mirror}")
        self.get_logger().info("Control Mapping:")
        self.get_logger().info("  MODE 1 (Shoulder Control):")
        self.get_logger().info("    Turn head LEFT/RIGHT → shoulder_pan_joint")
        self.get_logger().info("    Nod UP/DOWN → shoulder_lift_joint")
        self.get_logger().info("  MODE 2 (Wrist Control):")
        self.get_logger().info("    Turn head LEFT/RIGHT → wrist_2_joint (rotate gripper)")
        self.get_logger().info("    Nod UP/DOWN → shoulder_lift + elbow (extend/retract arm)")
        self.get_logger().info("    Auto wrist_1 compensation → maintain 90° gripper angle")
        self.get_logger().info("  Both eyes blink (hold >1s) → Toggle between MODE 1 and MODE 2")
        self.get_logger().info("  Open mouth (hold >1s) → Grasp toggle")
        self.get_logger().info("  Left eye blink (hold >1s) → Emergency stop")
        self.get_logger().info("  Right eye blink (hold >1s) → Recenter position")
    
    def joint_state_callback(self, msg: JointState):
        """Update current joint positions from joint_states topic."""
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.joint_positions[i] = msg.position[idx]
        
        if not self.got_joint_states:
            self.got_joint_states = True
            self.get_logger().info("Received joint states! Robot controller is active.")
            self.get_logger().info(f"   Initial positions: {[f'{p:.3f}' for p in self.joint_positions]}")
    
    def camera_loop(self):
        """Main camera processing loop."""
        if self.should_quit:
            rclpy.shutdown()
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame grab failed")
            return
        
        if self.mirror:
            frame = cv2.flip(frame, 1)
        
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        
        if not result.multi_face_landmarks:
            # No face detected - publish zero velocities if emergency stop
            if self.emergency_stop:
                self.publish_zero_trajectory()
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Show display even without face
            try:
                if not self.window_created:
                    cv2.namedWindow("Facemesh UR7e Control", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Facemesh UR7e Control", 1280, 960)
                    self.window_created = True
                    self.get_logger().info("OpenCV GUI window created and displaying camera feed!")
                
                cv2.imshow("Facemesh UR7e Control", frame)
                cv2.setWindowProperty("Facemesh UR7e Control", cv2.WND_PROP_TOPMOST, 1)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("Quit requested")
                    self.should_quit = True
            except Exception as e:
                self.get_logger().warn(f"Display error: {e}", throttle_duration_sec=5.0)
            return
        
        lm = result.multi_face_landmarks[0].landmark
        pts = np.array([[l.x * w, l.y * h] for l in lm])
        
        # Calculate face metrics
        face_center_y = calculate_face_center_y(lm, h)
        mouth_center_x = calculate_mouth_horizontal_position(lm, w)
        
        # Set neutral pose on first frame
        if not self.neutral_set:
            self.face_center_y0 = face_center_y
            self.mouth_center_x0 = mouth_center_x
            self.neutral_set = True
            self.get_logger().info(
                f"Neutral pose set: Face Y={self.face_center_y0:.1f}px, "
                f"Mouth X={self.mouth_center_x0:.1f}px"
            )
        
        # Calculate deltas
        dnod = face_center_y - self.face_center_y0
        dturn = mouth_center_x - self.mouth_center_x0
        
        # Apply smoothing
        dnod = self.face_y_filter.update(dnod)
        dturn = self.mouth_x_filter.update(dturn)
        
        # Apply deadzones
        if abs(dnod) < self.NOD_DEADZONE:
            dnod = 0.0
        if abs(dturn) < self.TURN_DEADZONE:
            dturn = 0.0
        
        # Detect left and right eye blinks separately
        left_eye = pts[LEFT_EYE_IDX]
        right_eye = pts[RIGHT_EYE_IDX]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        now = time.time()
        left_blink_triggered = False
        right_blink_triggered = False
        mode_toggle_triggered = False
        left_blink_hold_time = 0.0
        right_blink_hold_time = 0.0
        both_blink_hold_time = 0.0
        
        # Check if both eyes are closed
        both_eyes_closed_now = left_ear < self.EAR_THRESH and right_ear < self.EAR_THRESH
        
        # Both eyes blink detection - Mode toggle (hold >1s)
        if both_eyes_closed_now:
            if not self.both_eyes_closed:
                self.both_eyes_closed = True
                self.both_eyes_close_start = now
                self.mode_toggle_triggered = False
            else:
                # Both eyes still closed, check if we should trigger mode toggle
                both_blink_hold_time = now - self.both_eyes_close_start
                if both_blink_hold_time >= self.MODE_TOGGLE_HOLD_TIME and not self.mode_toggle_triggered:
                    mode_toggle_triggered = True
                    self.mode_toggle_triggered = True
                    self.control_mode_2 = not self.control_mode_2
                    mode_name = "MODE 2 (Wrist Control)" if self.control_mode_2 else "MODE 1 (Shoulder Control)"
                    self.get_logger().info(f"CONTROL MODE CHANGED to {mode_name} - Both eyes held closed for {both_blink_hold_time:.2f}s!")
                    self.play_beep(frequency=1000, duration=0.3)  # Play beep on mode change
        else:
            if self.both_eyes_closed:
                self.both_eyes_closed = False
                self.mode_toggle_triggered = False
        
        # Left eye blink detection - Emergency stop
        if left_ear < self.EAR_THRESH:
            if not self.left_eye_closed:
                self.left_eye_closed = True
                self.left_eye_close_start = now
                self.left_blink_triggered = False
            else:
                # Eye is still closed, check if we should trigger emergency stop
                left_blink_hold_time = now - self.left_eye_close_start
                if left_blink_hold_time >= self.BLINK_HOLD_TIME and not self.left_blink_triggered:
                    left_blink_triggered = True
                    self.left_blink_triggered = True
                    self.emergency_stop = not self.emergency_stop
                    if self.emergency_stop:
                        self.get_logger().warn(f"EMERGENCY STOP - Left eye blink held for {left_blink_hold_time:.2f}s")
                    else:
                        self.get_logger().info(f"Emergency stop released - Left eye blink held for {left_blink_hold_time:.2f}s")
        else:
            if self.left_eye_closed:
                self.left_eye_closed = False
                self.left_blink_triggered = False
        
        # Right eye blink detection - Recenter
        if right_ear < self.EAR_THRESH:
            if not self.right_eye_closed:
                self.right_eye_closed = True
                self.right_eye_close_start = now
                self.right_blink_triggered = False
            else:
                # Eye is still closed, check if we should trigger recenter
                right_blink_hold_time = now - self.right_eye_close_start
                if right_blink_hold_time >= self.BLINK_HOLD_TIME and not self.right_blink_triggered:
                    right_blink_triggered = True
                    self.right_blink_triggered = True
                    self.recenter_neutral()
                    self.get_logger().info(f"Recentering - Right eye blink held for {right_blink_hold_time:.2f}s")
        else:
            if self.right_eye_closed:
                self.right_eye_closed = False
                self.right_blink_triggered = False
        
        # Detect mouth open for grasp toggle
        mouth_top = pts[MOUTH_TOP_IDX]
        mouth_bottom = pts[MOUTH_BOTTOM_IDX]
        mouth_dist = np.linalg.norm(mouth_top - mouth_bottom)
        relative_dist = mouth_dist / h
        mouth_open = relative_dist > self.MOUTH_OPEN_THRESH
        
        now_time = time.time()
        grasp_triggered = False
        if mouth_open and not self.mouth_was_open:
            self.mouth_was_open = True
            self.grasp_open_start = now_time
            self.grasp_toggled = False
        elif mouth_open and self.mouth_was_open:
            mouth_open_duration = now_time - self.grasp_open_start
            if mouth_open_duration >= self.BLINK_HOLD_TIME and not self.grasp_toggled:
                grasp_triggered = True
                self.grasp_toggled = True
                self.get_logger().info(f"GRASP TOGGLE - Mouth held open for {mouth_open_duration:.2f}s")
        elif not mouth_open and self.mouth_was_open:
            self.mouth_was_open = False
            self.grasp_toggled = False
        
        if self.emergency_stop:
            self.publish_zero_trajectory()
            # Still show the display
            self.draw_overlay(frame, dnod, dturn, left_ear, right_ear, mouth_open, self.control_mode_2,
                             left_blink_hold_time, right_blink_hold_time, both_blink_hold_time)
            self.draw_mesh(frame, result, lm, pts, w, h)
            
            try:
                if not self.window_created:
                    cv2.namedWindow("Facemesh UR7e Control", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Facemesh UR7e Control", 1280, 960)
                    self.window_created = True
                    self.get_logger().info("OpenCV GUI window created and displaying camera feed!")
                
                cv2.imshow("Facemesh UR7e Control", frame)
                cv2.setWindowProperty("Facemesh UR7e Control", cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warn(f"Display error: {e}", throttle_duration_sec=5.0)
            return
        
        # Detect mouth open (emergency stop TOGGLE)
        if not self.got_joint_states:
            if time.time() > self.joint_states_timeout:
                self.got_joint_states = True
                self.get_logger().warn(
                    "WARNING: Joint states not received after 5 seconds. Proceeding anyway with default positions."
                )
                self.get_logger().warn("    Make sure the robot controller is running and publishing /joint_states")
            else:
                self.get_logger().warn("Waiting for joint states...", throttle_duration_sec=2.0)
                # Still show display while waiting
                self.draw_overlay(frame, dnod, dturn, left_ear, right_ear, mouth_open, self.control_mode_2,
                                 left_blink_hold_time, right_blink_hold_time, both_blink_hold_time)
                self.draw_mesh(frame, result, lm, pts, w, h)
                
                try:
                    if not self.window_created:
                        cv2.namedWindow("Facemesh UR7e Control", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Facemesh UR7e Control", 1280, 960)
                        self.window_created = True
                        self.get_logger().info("OpenCV GUI window created and displaying camera feed!")
                    
                    cv2.imshow("Facemesh UR7e Control", frame)
                    cv2.setWindowProperty("Facemesh UR7e Control", cv2.WND_PROP_TOPMOST, 1)
                    cv2.waitKey(1)
                except Exception as e:
                    self.get_logger().warn(f"Display error: {e}", throttle_duration_sec=5.0)
                return
        
        # Calculate new joint positions based on head movements and control mode
        new_positions = self.joint_positions.copy()
        
        if self.control_mode_2:
            # MODE 2: Wrist control
            # Turn (yaw) → wrist_2_joint (rotate gripper left/right)
            # Only move if beyond TURN_THRESHOLD (not yellow zone)
            if abs(dturn) > self.TURN_THRESHOLD:
                delta_wrist2 = dturn * self.yaw_to_wrist2_gain
                new_positions[4] += delta_wrist2  # wrist_2_joint for rotation
            
            # Nod (pitch) → Extend/Retract arm (move shoulder_lift + elbow simultaneously)
            # Only move if beyond NOD_THRESHOLD (not yellow zone)
            # Nod up (negative dnod) = extend arm, Nod down (positive dnod) = retract arm
            if abs(dnod) > self.NOD_THRESHOLD:
                # Move shoulder_lift and elbow in opposite directions to extend/retract
                # When shoulder_lift goes up, elbow goes down (and vice versa) to extend the arm
                arm_movement = dnod * self.pitch_to_lift_gain  # Nod up (negative dnod) = extend
                new_positions[1] += arm_movement  # shoulder_lift_joint moves in one direction
                new_positions[2] -= arm_movement  # elbow_joint moves in opposite direction to extend
            
            # Wrist must always stay perpendicular (90 degrees) to plane
            # wrist_1_joint = -(shoulder_lift_joint + elbow_joint) to maintain 90 degrees
            # This is set always, not just during extend/retract
            new_positions[3] = -(new_positions[1] + new_positions[2])
        else:
            # MODE 1: Shoulder control (default)
            # Turn (yaw) → shoulder_pan_joint
            # Only move if beyond TURN_THRESHOLD (not yellow zone)
            if abs(dturn) > self.TURN_THRESHOLD:
                delta_pan = dturn * self.yaw_to_pan_gain
                new_positions[0] += delta_pan
            
            # Nod (pitch) → shoulder_lift_joint
            # Only move if beyond NOD_THRESHOLD (not yellow zone)
            if abs(dnod) > self.NOD_THRESHOLD:
                delta_lift = -dnod * self.pitch_to_lift_gain  # Negative because nod up should lift
                new_positions[1] += delta_lift
                
                # Wrist counteraction: move wrist_1_joint opposite to shoulder_lift to keep gripper at 90deg
                delta_wrist = -delta_lift * self.pitch_to_wrist_gain  # Opposite to shoulder_lift movement
                new_positions[3] += delta_wrist  # wrist_1_joint counteracts pitch
        
        # Mouth open → GRASP toggle (use service call instead of joint trajectory)
        if grasp_triggered:
            self.toggle_gripper_service()
        
        # Publish trajectory
        self.publish_trajectory(new_positions)
        
        # Update display with overlay and show window
        self.draw_overlay(frame, dnod, dturn, left_ear, right_ear, mouth_open, self.control_mode_2,
                         left_blink_hold_time, right_blink_hold_time, both_blink_hold_time)
        self.draw_mesh(frame, result, lm, pts, w, h)
        
        try:
            # Create window if it doesn't exist and show the frame
            if not self.window_created:
                cv2.namedWindow("Facemesh UR7e Control", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Facemesh UR7e Control", 1280, 960)
                self.window_created = True
                self.get_logger().info("✅ OpenCV GUI window created and displaying camera feed!")
            
            cv2.imshow("Facemesh UR7e Control", frame)
            cv2.setWindowProperty("Facemesh UR7e Control", cv2.WND_PROP_TOPMOST, 1)
            
            # Wait for key press (1ms timeout)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.recenter_neutral()
            elif key == ord('q'):
                self.get_logger().info("Quit requested")
                self.should_quit = True
            elif key == ord('m'):
                self.mirror = not self.mirror
                self.get_logger().info(f"Mirror mode: {'ON' if self.mirror else 'OFF'}")
        except Exception as e:
            self.get_logger().warn(f"Display error: {e}", throttle_duration_sec=5.0)
    
    def draw_overlay(self, frame, dnod, dturn, left_ear, right_ear, mouth_open, control_mode_2, 
                     left_blink_hold_time=0.0, right_blink_hold_time=0.0, both_blink_hold_time=0.0):
        """Draw overlay information on frame with detailed status."""
        h, w = frame.shape[:2]
        
        # Top-left column: Control Mode and Status
        y_offset = 30
        mode_text = "MODE 2: WRIST" if control_mode_2 else "MODE 1: SHOULDER"
        mode_color = (100, 200, 255) if control_mode_2 else (100, 255, 150)
        cv2.putText(frame, mode_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 3)
        y_offset += 45
        
        status_color = (0, 255, 0) if not self.emergency_stop else (0, 0, 255)
        status_text = "RUNNING" if not self.emergency_stop else "STOPPED"
        cv2.putText(frame, status_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)
        
        # Deadzone indicator
        y_offset += 40
        in_deadzone = abs(dnod) < self.NOD_DEADZONE and abs(dturn) < self.TURN_DEADZONE
        deadzone_color = (0, 255, 0) if in_deadzone else (0, 200, 255)
        deadzone_text = "DEADZONE OK" if in_deadzone else "DEADZONE"
        cv2.putText(frame, deadzone_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, deadzone_color, 2)
        
        # Top-left column continued: Deltas with color coding
        y_offset += 35
        cv2.putText(frame, "MOTION:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
        
        # Nod status
        if dnod == 0.0:
            nod_color = (0, 255, 0)  # Green - in deadzone
            nod_text = "Nod:  NEUTRAL"
        elif abs(dnod) < self.NOD_THRESHOLD:
            nod_color = (0, 255, 255)  # Yellow - in yellow zone, no movement
            nod_text = f"Nod:  {dnod:+6.1f}px (YELLOW)"
        else:
            nod_color = (0, 0, 255)  # Red - active
            nod_text = f"Nod:  {dnod:+6.1f}px {'UP' if dnod < 0 else 'DOWN'}"
        
        cv2.putText(frame, nod_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, nod_color, 2)
        y_offset += 30
        
        # Turn status
        if dturn == 0.0:
            turn_color = (0, 255, 0)  # Green - in deadzone
            turn_text = "Turn: NEUTRAL"
        elif abs(dturn) < self.TURN_THRESHOLD:
            turn_color = (0, 255, 255)  # Yellow - in yellow zone, no movement
            turn_text = f"Turn: {dturn:+6.1f}px (YELLOW)"
        else:
            turn_color = (0, 0, 255)  # Red - active
            turn_text = f"Turn: {dturn:+6.1f}px {'LEFT' if dturn < 0 else 'RIGHT'}"
        
        cv2.putText(frame, turn_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, turn_color, 2)
        
        # Eye status with blink timers (top-left, below motion)
        y_offset += 40
        cv2.putText(frame, "EYE BLINKS:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
        
        # Left eye indicator
        left_eye_color = (255, 0, 0) if left_ear < self.EAR_THRESH else (0, 255, 0)
        left_eye_status = "CLOSED" if left_ear < self.EAR_THRESH else "OPEN"
        left_eye_text = f"L-Eye: {left_eye_status} {left_blink_hold_time:.2f}s"
        cv2.putText(frame, left_eye_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_eye_color, 2)
        y_offset += 30
        
        # Right eye indicator
        right_eye_color = (255, 0, 0) if right_ear < self.EAR_THRESH else (0, 255, 0)
        right_eye_status = "CLOSED" if right_ear < self.EAR_THRESH else "OPEN"
        right_eye_text = f"R-Eye: {right_eye_status} {right_blink_hold_time:.2f}s"
        cv2.putText(frame, right_eye_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_eye_color, 2)
        y_offset += 30
        
        # Both eyes indicator for mode toggle
        both_eyes_color = (255, 0, 0) if left_ear < self.EAR_THRESH and right_ear < self.EAR_THRESH else (0, 255, 0)
        both_blink_bar = "=" * min(10, int(both_blink_hold_time * 10))
        both_eyes_text = f"Mode: {both_blink_bar} {both_blink_hold_time:.2f}s"
        cv2.putText(frame, both_eyes_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, both_eyes_color, 2)
        
        # Top-center column: Active Command
        y_offset = 30
        cv2.putText(frame, "COMMAND:", (350, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        y_offset += 40
        
        command_text = "(neutral)"
        command_color = (0, 255, 0)
        
        if control_mode_2:
            # MODE 2: Wrist control commands
            if dturn < -self.TURN_THRESHOLD:
                command_text = "ROTATE LEFT"
                command_color = (0, 0, 255)
            elif dturn > self.TURN_THRESHOLD:
                command_text = "ROTATE RIGHT"
                command_color = (0, 0, 255)
            elif dnod < -self.NOD_THRESHOLD:
                command_text = "EXTEND ARM"
                command_color = (0, 0, 255)
            elif dnod > self.NOD_THRESHOLD:
                command_text = "RETRACT ARM"
                command_color = (0, 0, 255)
        else:
            # MODE 1: Shoulder control commands
            if dturn < -self.TURN_THRESHOLD:
                command_text = "TURN LEFT"
                command_color = (0, 0, 255)
            elif dturn > self.TURN_THRESHOLD:
                command_text = "TURN RIGHT"
                command_color = (0, 0, 255)
            elif dnod < -self.NOD_THRESHOLD:
                command_text = "NOD UP"
                command_color = (0, 0, 255)
            elif dnod > self.NOD_THRESHOLD:
                command_text = "NOD DOWN"
                command_color = (0, 0, 255)
        
        cv2.putText(frame, command_text, (350, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, command_color, 3)
        
        # Top-right column: Mouth and Grasp status
        y_offset = 30
        mouth_color = (0, 165, 255) if mouth_open else (0, 255, 0)
        mouth_status = "OPEN" if mouth_open else "CLOSED"
        mouth_text = f"MOUTH: {mouth_status}"
        cv2.putText(frame, mouth_text, (650, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, mouth_color, 3)
        y_offset += 45
        
        # Thresholds reference
        cv2.putText(frame, "Thresholds:", (650, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_offset += 30
        cv2.putText(frame, f"Turn: +/-{self.TURN_THRESHOLD:.0f}px", (650, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        y_offset += 28
        cv2.putText(frame, f"Nod: +/-{self.NOD_THRESHOLD:.0f}px", (650, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        y_offset += 28
        cv2.putText(frame, f"Deadzone: {self.TURN_DEADZONE:.0f}px", (650, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Bottom instruction bar
        instructions = "BOTH EYES BLINK=MODE  |  L-BLINK=STOP  |  R-BLINK=CENTER  |  MOUTH=GRASP"
        cv2.putText(frame, instructions,
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    def draw_mesh(self, frame, result, lm, pts, w, h):
        """Draw face mesh and key landmarks on frame."""
        if not result.multi_face_landmarks:
            return
        
        # Draw full mesh
        for lm_point in result.multi_face_landmarks[0].landmark:
            x, y = int(lm_point.x * w), int(lm_point.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        
        # Highlight key landmarks
        # Eyes - Blue
        cv2.circle(frame, (int(lm[33].x * w), int(lm[33].y * h)), 5, (255, 0, 0), -1)  # Left eye
        cv2.circle(frame, (int(lm[263].x * w), int(lm[263].y * h)), 5, (255, 0, 0), -1)  # Right eye
        
        # Mouth corners (for turn) - Green
        cv2.circle(frame, (int(lm[LEFT_MOUTH_CORNER].x * w), int(lm[LEFT_MOUTH_CORNER].y * h)), 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(lm[RIGHT_MOUTH_CORNER].x * w), int(lm[RIGHT_MOUTH_CORNER].y * h)), 5, (0, 255, 0), -1)
        
        # Face center points (for nod) - Yellow
        cv2.circle(frame, (int(lm[NOSE_TIP].x * w), int(lm[NOSE_TIP].y * h)), 5, (255, 255, 0), -1)
        cv2.circle(frame, (int(lm[CHIN].x * w), int(lm[CHIN].y * h)), 5, (255, 255, 0), -1)
    
    def publish_trajectory(self, positions):
        """Publish joint trajectory to UR7e."""
        from builtin_interfaces.msg import Duration
        
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.velocities = [0.0] * 6
        # Fast trajectory execution: 50ms per command for real-time response (faster)
        point.time_from_start = Duration(sec=0, nanosec=50000000)  # 50ms in nanoseconds
        traj.points.append(point)
        
        self.pub.publish(traj)
        self.joint_positions = positions.copy()
        self.get_logger().debug(f"📨 Published trajectory: {[f'{p:.3f}' for p in positions]}")
    
    def publish_zero_trajectory(self):
        """Publish zero velocity trajectory for emergency stop."""
        from builtin_interfaces.msg import Duration
        
        if not self.got_joint_states:
            return
        
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = self.joint_positions.copy()  # Hold current position
        point.velocities = [0.0] * 6
        point.time_from_start = Duration(sec=0, nanosec=50000000)  # 50ms
        traj.points.append(point)
        
        self.pub.publish(traj)
    
    def toggle_gripper_service(self):
        """Call gripper toggle service to fully open/close gripper."""
        if not self.gripper_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Gripper service not available", throttle_duration_sec=2.0)
            return
        
        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        
        # Wait for the service call to complete (blocking)
        try:
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            response = future.result()
            if response.success:
                self.get_logger().info("Gripper toggled (will fully open/close)!")
            else:
                self.get_logger().warn(f"Gripper toggle failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Gripper service error: {e}")
    
    def recenter_neutral(self):
        """Recenter the neutral pose (called from right eye blink)."""
        # Reset filters and neutral tracking so next frame sets new neutral
        self.neutral_set = False
        self.face_y_filter.reset()
        self.mouth_x_filter.reset()
        self.get_logger().info("Neutral pose reset - will recenter on next frame")
    
    def play_beep(self, frequency=1000, duration=0.2, sample_rate=44100):
        """Play a beep sound at specified frequency."""
        try:
            # Generate sine wave
            num_samples = int(sample_rate * duration)
            frames = []
            for i in range(num_samples):
                sample = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
                frames.append((sample & 0xFF).to_bytes(1, 'little'))
                frames.append(((sample >> 8) & 0xFF).to_bytes(1, 'little'))
            
            # Write to temporary wave file and play
            temp_file = "/tmp/mode_beep.wav"
            with wave.open(temp_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(b''.join(frames))
            
            # Play the sound using system command
            os.system(f"aplay {temp_file} &> /dev/null &")
        except Exception as e:
            self.get_logger().warn(f"Could not play beep sound: {e}", throttle_duration_sec=5.0)
    
    def shutdown(self):
        """Cleanup resources."""
        self.cap.release()
        self.face_mesh.close()
        cv2.destroyAllWindows()


def main(args=None):
    parser = argparse.ArgumentParser(description='Facemesh UR7e Control Node')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--no-mirror', action='store_true', help='Disable camera mirroring')
    args_parsed, ros_args = parser.parse_known_args(args)
    
    rclpy.init(args=ros_args)
    node = FacemeshUR7eControlNode(
        camera_index=args_parsed.camera,
        mirror=not args_parsed.no_mirror
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
