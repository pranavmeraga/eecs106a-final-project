#!/usr/bin/env python3
"""
facemesh_ur7e_control_node.py

Controls UR7e robot joints using facemesh head pose detection.
Combines facemesh detection with joint trajectory publishing.

Subscribes:
- /joint_states (JointState): Current joint positions

Publishes:
- /scaled_joint_trajectory_controller/joint_trajectory (JointTrajectory): Joint commands

Control Mapping:
- Turn head left/right (yaw) → shoulder_pan_joint
- Nod up/down (pitch) → shoulder_lift_joint
- Tilt head left/right (roll) → elbow_joint
- Long blink → wrist_1_joint adjustment
- Open mouth → Emergency stop (zero velocities)
"""

import cv2
import time
import numpy as np
import mediapipe as mp
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


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


def calculate_head_tilt_angle(lm, w, h):
    """Calculate TILT based on eye horizontal alignment."""
    # Get eye positions
    left_eye_x = lm[33].x * w
    left_eye_y = lm[33].y * h
    right_eye_x = lm[263].x * w
    right_eye_y = lm[263].y * h
    
    # Calculate angle of line connecting eyes
    dx = right_eye_x - left_eye_x
    dy = right_eye_y - left_eye_y
    
    # Angle in radians (0 = horizontal, + = right eye higher, - = left eye higher)
    tilt_angle = np.arctan2(dy, dx)
    
    return tilt_angle


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
        
        self.get_logger().info("📡 Subscribed to /joint_states topic")
        self.get_logger().info("📤 Publishing to /scaled_joint_trajectory_controller/joint_trajectory")
        
        # Publisher for joint trajectory
        self.pub = self.create_publisher(
            JointTrajectory, 
            '/scaled_joint_trajectory_controller/joint_trajectory', 
            10
        )
        
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
        self.tilt_angle0 = 0.0
        
        # Smoothing filters
        self.face_y_filter = SmoothingFilter(alpha=0.3)
        self.mouth_x_filter = SmoothingFilter(alpha=0.3)
        self.tilt_filter = SmoothingFilter(alpha=0.3)
        
        # Blink tracking - SENSITIVE for immediate grasp detection
        self.eye_closed = False
        self.eye_close_start = 0.0
        self.EAR_THRESH = 0.15  # LOWERED from 0.21 - more sensitive blink detection
        self.GRASP_BLINK_TIME = 0.3  # REDUCED from 1.0 - triggers grasp immediately after 300ms blink
        self.blink_frames_count = 0  # Track consecutive blink frames
        
        # Mouth tracking
        self.MOUTH_OPEN_THRESH = 0.03
        
        # Thresholds - same as facemesh_preview.py
        self.TURN_THRESHOLD = 25.0     # pixels
        self.NOD_THRESHOLD = 25.0      # pixels
        self.TILT_THRESHOLD = 0.12     # radians ~6.9°
        
        # Deadzones
        self.TURN_DEADZONE = 10.0      # pixels
        self.NOD_DEADZONE = 10.0       # pixels
        self.TILT_DEADZONE = 0.05      # radians ~2.9°
        
        # Control gains - how much joint angle changes per pixel/radian
        self.yaw_to_pan_gain = 0.05    # radians per pixel (turn → shoulder_pan) - INCREASED for faster response
        self.pitch_to_lift_gain = 0.05  # radians per pixel (nod → shoulder_lift) - INCREASED for faster response
        self.roll_to_elbow_gain = 1.0   # radians per radian (tilt → elbow) - INCREASED for faster response
        
        # Max joint velocity limits (radians)
        self.max_joint_velocity = 0.5
        
        # Emergency stop flag
        self.emergency_stop = False
        
        # Quit flag
        self.should_quit = False
        
        # Display flag to track if window has been created
        self.window_created = False
        
        # Timer for camera loop
        self.timer = self.create_timer(1.0 / 60.0, self.camera_loop)  # 60 Hz for faster response
        
        self.get_logger().info("Facemesh UR7e Control Node Started")
        self.get_logger().info(f"Camera index: {camera_index}, Mirror: {mirror}")
        self.get_logger().info("Control Mapping:")
        self.get_logger().info("  Turn head LEFT/RIGHT → shoulder_pan_joint")
        self.get_logger().info("  Nod UP/DOWN → shoulder_lift_joint")
        self.get_logger().info("  Tilt LEFT/RIGHT → elbow_joint")
        self.get_logger().info("  Long blink → wrist_1_joint adjustment")
        self.get_logger().info("  Open mouth → Emergency stop")
        self.get_logger().info("Press 'r' in OpenCV window to recenter neutral pose")
        self.get_logger().info("🎬 OpenCV GUI window will appear shortly...")
        self.get_logger().info("⚡ Performance Settings:")
        self.get_logger().info("   • Update rate: 60 Hz (was 30 Hz)")
        self.get_logger().info("   • Trajectory time: 100ms (was 5 seconds)")
        self.get_logger().info("   • Control gains: 5x faster response")
    
    def joint_state_callback(self, msg: JointState):
        """Update current joint positions from joint_states topic."""
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.joint_positions[i] = msg.position[idx]
        
        if not self.got_joint_states:
            self.got_joint_states = True
            self.get_logger().info("✅ Received joint states! Robot controller is active.")
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
                    cv2.namedWindow("Facemesh UR7e Control", cv2.WINDOW_AUTOSIZE)
                    self.window_created = True
                    self.get_logger().info("✅ OpenCV GUI window created and displaying camera feed!")
                
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
        tilt_angle = calculate_head_tilt_angle(lm, w, h)
        
        # Set neutral pose on first frame
        if not self.neutral_set:
            self.face_center_y0 = face_center_y
            self.mouth_center_x0 = mouth_center_x
            self.tilt_angle0 = tilt_angle
            self.neutral_set = True
            self.get_logger().info(
                f"Neutral pose set: Face Y={self.face_center_y0:.1f}px, "
                f"Mouth X={self.mouth_center_x0:.1f}px, "
                f"Tilt={np.degrees(self.tilt_angle0):.1f}°"
            )
        
        # Calculate deltas
        dnod = face_center_y - self.face_center_y0
        dturn = mouth_center_x - self.mouth_center_x0
        dtilt = tilt_angle - self.tilt_angle0
        
        # Apply smoothing
        dnod = self.face_y_filter.update(dnod)
        dturn = self.mouth_x_filter.update(dturn)
        dtilt = self.tilt_filter.update(dtilt)
        
        # Apply deadzones
        if abs(dnod) < self.NOD_DEADZONE:
            dnod = 0.0
        if abs(dturn) < self.TURN_DEADZONE:
            dturn = 0.0
        if abs(dtilt) < self.TILT_DEADZONE:
            dtilt = 0.0
        
        # Detect blink - SENSITIVE for grasp
        left_eye = pts[LEFT_EYE_IDX]
        right_eye = pts[RIGHT_EYE_IDX]
        ear = 0.5 * (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye))
        
        now = time.time()
        long_blink = False
        
        if ear < self.EAR_THRESH:
            if not self.eye_closed:
                self.eye_closed = True
                self.eye_close_start = now
                self.blink_frames_count = 0
            else:
                self.blink_frames_count += 1
                duration = now - self.eye_close_start
                # Trigger grasp if held closed long enough (300ms at 60Hz = ~18 frames)
                if duration >= self.GRASP_BLINK_TIME and self.blink_frames_count >= 15:
                    long_blink = True
                    self.get_logger().info(f"🤏 GRASP TRIGGERED! (EAR={ear:.3f}, held for {duration:.2f}s)")
        else:
            if self.eye_closed:
                self.eye_closed = False
                self.blink_frames_count = 0
        
        # Detect mouth open (emergency stop)
        mouth_top = pts[MOUTH_TOP_IDX]
        mouth_bottom = pts[MOUTH_BOTTOM_IDX]
        mouth_dist = np.linalg.norm(mouth_top - mouth_bottom)
        relative_dist = mouth_dist / h
        mouth_open = relative_dist > self.MOUTH_OPEN_THRESH
        
        if mouth_open:
            self.emergency_stop = True
            self.get_logger().warn("🛑 EMERGENCY STOP - Mouth Open")
            self.publish_zero_trajectory()
            # Still show the display
            self.draw_overlay(frame, dnod, dturn, dtilt, ear, mouth_open, long_blink)
            self.draw_mesh(frame, result, lm, pts, w, h)
            
            try:
                if not self.window_created:
                    cv2.namedWindow("Facemesh UR7e Control", cv2.WINDOW_AUTOSIZE)
                    self.window_created = True
                    self.get_logger().info("✅ OpenCV GUI window created and displaying camera feed!")
                
                cv2.imshow("Facemesh UR7e Control", frame)
                cv2.setWindowProperty("Facemesh UR7e Control", cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warn(f"Display error: {e}", throttle_duration_sec=5.0)
            return
        
        # Reset emergency stop if mouth closes
        if self.emergency_stop and not mouth_open:
            self.emergency_stop = False
            self.get_logger().info("✅ Emergency stop released")
        
        # Check if we should start publishing (either got joint states or timeout)
        if not self.got_joint_states:
            if time.time() > self.joint_states_timeout:
                self.got_joint_states = True
                self.get_logger().warn(
                    "⚠️  Joint states not received after 5 seconds. Proceeding anyway with default positions."
                )
                self.get_logger().warn("    Make sure the robot controller is running and publishing /joint_states")
            else:
                self.get_logger().warn("Waiting for joint states...", throttle_duration_sec=2.0)
                # Still show display while waiting
                self.draw_overlay(frame, dnod, dturn, dtilt, ear, mouth_open, long_blink)
                self.draw_mesh(frame, result, lm, pts, w, h)
                
                try:
                    if not self.window_created:
                        cv2.namedWindow("Facemesh UR7e Control", cv2.WINDOW_AUTOSIZE)
                        self.window_created = True
                        self.get_logger().info("✅ OpenCV GUI window created and displaying camera feed!")
                    
                    cv2.imshow("Facemesh UR7e Control", frame)
                    cv2.setWindowProperty("Facemesh UR7e Control", cv2.WND_PROP_TOPMOST, 1)
                    cv2.waitKey(1)
                except Exception as e:
                    self.get_logger().warn(f"Display error: {e}", throttle_duration_sec=5.0)
                return
        
        # Calculate new joint positions based on head movements
        new_positions = self.joint_positions.copy()
        
        # Map head movements to joints
        # Turn (yaw) → shoulder_pan_joint
        if abs(dturn) > self.TURN_DEADZONE:
            delta_pan = dturn * self.yaw_to_pan_gain
            new_positions[0] += delta_pan
        
        # Nod (pitch) → shoulder_lift_joint
        if abs(dnod) > self.NOD_DEADZONE:
            delta_lift = -dnod * self.pitch_to_lift_gain  # Negative because nod up should lift
            new_positions[1] += delta_lift
        
        # Tilt (roll) → elbow_joint
        if abs(dtilt) > self.TILT_DEADZONE:
            delta_elbow = dtilt * self.roll_to_elbow_gain
            new_positions[2] += delta_elbow
        
        # Long blink → GRASP! Execute strong grasp command
        if long_blink:
            # Strong grasp: move wrist joints significantly to close gripper
            new_positions[3] += 1.5  # LARGE adjustment for aggressive grasp (was 0.1)
            new_positions[4] += 0.3  # Also adjust wrist_2_joint for better grasp
            new_positions[5] -= 0.3  # Adjust wrist_3_joint
            self.get_logger().info("✊✊✊ GRASP COMMAND EXECUTED! Closing gripper with force...")
        
        # Publish trajectory
        self.publish_trajectory(new_positions)
        
        # Update display with overlay and show window
        self.draw_overlay(frame, dnod, dturn, dtilt, ear, mouth_open, long_blink)
        self.draw_mesh(frame, result, lm, pts, w, h)
        
        try:
            # Create window if it doesn't exist and show the frame
            if not self.window_created:
                cv2.namedWindow("Facemesh UR7e Control", cv2.WINDOW_AUTOSIZE)
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
    
    def draw_overlay(self, frame, dnod, dturn, dtilt, ear, mouth_open, long_blink):
        """Draw comprehensive overlay information on frame (from facemesh_preview)."""
        h, w = frame.shape[:2]
        y_offset = 30
        
        # === RAW POSITIONS ===
        cv2.putText(frame, "=== RAW POSITIONS ===", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        # We'll calculate and display raw positions
        # For now show the deltas, will add raw in camera_loop
        cv2.putText(frame, "Facemesh UR7e Control", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # === DELTA (smoothed) ===
        cv2.putText(frame, "=== DELTA (smoothed) ===", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        # Color code: GREEN=deadzone, CYAN=below threshold, RED=command active
        nod_color = (0, 255, 0) if dnod == 0.0 else ((0, 0, 255) if abs(dnod) > self.NOD_THRESHOLD else (0, 255, 255))
        turn_color = (0, 255, 0) if dturn == 0.0 else ((0, 0, 255) if abs(dturn) > self.TURN_THRESHOLD else (0, 255, 255))
        tilt_color = (0, 255, 0) if dtilt == 0.0 else ((0, 0, 255) if abs(dtilt) > self.TILT_THRESHOLD else (0, 255, 255))
        
        cv2.putText(frame, f"dNod:   {dnod:7.2f} px", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, nod_color, 2)
        y_offset += 25
        cv2.putText(frame, f"dTurn:  {dturn:7.2f} px", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, turn_color, 2)
        y_offset += 25
        cv2.putText(frame, f"dTilt:  {np.degrees(dtilt):7.2f} deg", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, tilt_color, 2)

        # === COMMANDS ===
        y_offset += 35
        cv2.putText(frame, "=== COMMANDS ===", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

        command_shown = False

        # TURN command - based on MOUTH horizontal position
        if dturn < -self.TURN_THRESHOLD:
            cv2.putText(frame, "Turn LEFT (mouth moves left)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            command_shown = True
            y_offset += 30
        elif dturn > self.TURN_THRESHOLD:
            cv2.putText(frame, "Turn RIGHT (mouth moves right)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            command_shown = True
            y_offset += 30

        # NOD command - based on FACE vertical position
        if dnod < -self.NOD_THRESHOLD:
            cv2.putText(frame, "Nod UP (face moves up)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            command_shown = True
            y_offset += 30
        elif dnod > self.NOD_THRESHOLD:
            cv2.putText(frame, "Nod DOWN (face moves down)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            command_shown = True
            y_offset += 30

        # TILT command - based on EYE LINE angle
        if dtilt < -self.TILT_THRESHOLD:
            cv2.putText(frame, "Tilt LEFT (left eye higher)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            command_shown = True
            y_offset += 30
        elif dtilt > self.TILT_THRESHOLD:
            cv2.putText(frame, "Tilt RIGHT (right eye higher)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            command_shown = True
            y_offset += 30
        
        if not command_shown:
            cv2.putText(frame, "(neutral - no command)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30

        # === BLINK & MOUTH STATUS ===
        y_offset += 10
        blink_status = ""
        blink_color = (255, 255, 255)
        
        if ear < self.EAR_THRESH:
            if long_blink:
                blink_status = "✊✊✊ GRASPING! ✊✊✊"
                blink_color = (0, 0, 255)  # Red - active grasp!
            else:
                blink_status = f"Eyes closed - Hold for grasp! (EAR={ear:.3f})"
                blink_color = (0, 165, 255)  # Orange - blink detected
        else:
            blink_status = f"Eyes open (EAR={ear:.3f})"
            blink_color = (0, 255, 0)  # Green - normal

        cv2.putText(frame, blink_status, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, blink_color, 3)  # Larger text for grasp status
        y_offset += 25

        # Mouth detection
        if mouth_open:
            cv2.putText(frame, "STOP (mouth open)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Mouth closed", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Instructions at bottom
        cv2.putText(frame, "Press 'r' to recenter | 'q' to quit | Ctrl+C to stop", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    def draw_mesh(self, frame, result, lm, pts, w, h):
        """Draw face mesh and key landmarks on frame."""
        if not result.multi_face_landmarks:
            return
        
        # Draw full mesh
        for lm_point in result.multi_face_landmarks[0].landmark:
            x, y = int(lm_point.x * w), int(lm_point.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        
        # Highlight key landmarks
        # Eyes (for tilt) - Blue
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
        # Fast trajectory execution: 100ms per command for real-time response
        point.time_from_start = Duration(sec=0, nanosec=100000000)  # 100ms in nanoseconds
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
    
    def recenter_neutral(self):
        """Recenter the neutral pose (called from keyboard handler)."""
        # This will be called when 'r' is pressed
        # Reset filters and wait for next frame to set new neutral
        self.neutral_set = False
        self.face_y_filter.reset()
        self.mouth_x_filter.reset()
        self.tilt_filter.reset()
        self.get_logger().info("Recenter requested - will set new neutral on next frame")
    
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

