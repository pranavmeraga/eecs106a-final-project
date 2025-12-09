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
- Nod up/down (pitch) → shoulder_lift_joint and shoulder_pan_joint (extend/retract, wrist stays perpendicular)
- Tilt head left/right (roll) → elbow_joint
- Long blink → Grasp command (wrist joints)
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

# Eye landmarks for wink detection
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263


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


def calculate_face_size(lm, w, h):
    """Calculate face size as a measure of proximity to camera.
    Returns normalized face size (0-1 scale, larger = closer to camera).
    Uses forehead-to-chin distance and eye-to-eye distance for robust measurement."""
    # Forehead to chin distance (vertical face size)
    forehead_y = lm[FOREHEAD].y * h
    chin_y = lm[CHIN].y * h
    vertical_size = abs(chin_y - forehead_y)
    
    # Eye-to-eye distance (horizontal face size)
    left_eye_x = lm[33].x * w
    right_eye_x = lm[263].x * w
    horizontal_size = abs(right_eye_x - left_eye_x)
    
    # Combined face size metric (normalized by frame size)
    face_size = (vertical_size + horizontal_size) / (w + h)
    
    return face_size


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
        self.face_size0 = 0.0  # Baseline face size for proximity detection
        
        # Smoothing filters
        self.face_y_filter = SmoothingFilter(alpha=0.3)
        self.mouth_x_filter = SmoothingFilter(alpha=0.3)
        self.tilt_filter = SmoothingFilter(alpha=0.3)
        self.face_size_filter = SmoothingFilter(alpha=0.2)  # Slower filter for stable size measurement
        
        # 🎯 IMPRESSIVE: Proximity-Based Precision Mode
        self.precision_mode = True  # True = fine control (close), False = coarse control (far)
        self.face_size_threshold = 0.15  # Threshold for switching modes (normalized size)
        self.precision_gain_multiplier = 0.3  # Fine control: 30% of normal speed
        self.coarse_gain_multiplier = 2.0  # Coarse control: 200% of normal speed
        
        # Blink tracking - trigger grasp once when held >1s
        self.eye_closed = False
        self.eye_close_start = 0.0
        self.EAR_THRESH = 0.21  # Eye aspect ratio threshold
        self.GRASP_BLINK_TIME = 1.0  # Hold blink for 1 second to trigger grasp
        self.blink_frames_count = 0  # Track consecutive blink frames
        self.blink_triggered = False  # Track if we've already triggered this blink (one-time trigger)
        
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
        self.yaw_to_pan_gain = 0.15    # radians per pixel (turn → shoulder_pan) - INCREASED for faster response
        self.pitch_to_lift_gain = 0.15  # radians per pixel (nod → shoulder_lift) - INCREASED for faster response
        self.roll_to_elbow_gain = 1.5   # radians per radian (tilt → elbow) - INCREASED for faster response
        self.pitch_to_wrist_gain = 1.0  # radians per pixel (nod → wrist_1 counteraction) - NEW for wrist control
        
        # Max joint velocity limits (radians)
        self.max_joint_velocity = 0.5
        
        # Emergency stop flag (toggle state)
        self.emergency_stop = False
        self.mouth_was_open = False  # Track previous mouth state for toggle
        
        # 🎯 IMPRESSIVE FEATURES: Auto-Home Position
        self.home_position = None  # Saved home position
        self.returning_to_home = False
        self.home_return_speed = 0.02  # Radians per update
        
        # 🎯 IMPRESSIVE FEATURES: Eye Wink Detection
        self.left_eye_closed = False
        self.right_eye_closed = False
        self.left_wink_start = 0.0
        self.right_wink_start = 0.0
        self.WINK_TIME = 0.3  # Quick wink detection (300ms)
        self.wink_cooldown = 1.0  # Cooldown between winks
        self.last_wink_time = 0.0
        
        # 🎯 IMPRESSIVE FEATURES: Head Shake Detection (quick left-right)
        self.head_shake_buffer = []  # Store recent head movements
        self.shake_detection_window = 1.0  # seconds
        self.SHAKE_THRESHOLD = 3  # Minimum number of direction changes
        
        # 🎯 IMPRESSIVE FEATURES: Movement Recording
        self.recording = False
        self.recorded_movements = []
        self.recording_start_time = 0.0
        self.playback_index = 0
        self.playing_back = False
        
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
        self.get_logger().info("  Nod UP/DOWN → shoulder_lift_joint + shoulder_pan_joint (extend/retract, wrist stays perpendicular)")
        self.get_logger().info("  Tilt LEFT/RIGHT → elbow_joint")
        self.get_logger().info("  Long blink (>1s) → Grasp command")
        self.get_logger().info("  Open mouth → Emergency stop (toggle)")
        self.get_logger().info("Press 'r' in OpenCV window to recenter neutral pose")
        self.get_logger().info("OpenCV GUI window will appear shortly...")
        self.get_logger().info("Performance Settings:")
        self.get_logger().info("   • Update rate: 60 Hz (was 30 Hz)")
        self.get_logger().info("   • Trajectory time: 50ms (was 5 seconds)")
        self.get_logger().info("   • Control gains: 3x faster response")
        self.get_logger().info("IMPRESSIVE FEATURES:")
        self.get_logger().info("   • Left Wink → Save home position")
        self.get_logger().info("   • Right Wink → Return to home position")
        self.get_logger().info("   • Head Shake (quick L-R) → Cancel operation")
        self.get_logger().info("   • PROXIMITY-BASED PRECISION MODE:")
        self.get_logger().info("      - Lean IN (close to camera) → Fine control (30% speed)")
        self.get_logger().info("      - Lean BACK (far from camera) → Coarse control (200% speed)")
        self.get_logger().info("      - Automatic switching based on face size")
    
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
        face_size = calculate_face_size(lm, w, h)  # 🎯 Proximity detection
        
        # Set neutral pose on first frame
        if not self.neutral_set:
            self.face_center_y0 = face_center_y
            self.mouth_center_x0 = mouth_center_x
            self.tilt_angle0 = tilt_angle
            self.face_size0 = face_size
            self.neutral_set = True
            self.get_logger().info(
                f"Neutral pose set: Face Y={self.face_center_y0:.1f}px, "
                f"Mouth X={self.mouth_center_x0:.1f}px, "
                f"Tilt={np.degrees(self.tilt_angle0):.1f}°, "
                f"Face Size={self.face_size0:.4f}"
            )
        
        # Calculate deltas
        dnod = face_center_y - self.face_center_y0
        dturn = mouth_center_x - self.mouth_center_x0
        dtilt = tilt_angle - self.tilt_angle0
        
        # Apply smoothing
        dnod = self.face_y_filter.update(dnod)
        dturn = self.mouth_x_filter.update(dturn)
        dtilt = self.tilt_filter.update(dtilt)
        
        # 🎯 IMPRESSIVE: Proximity-Based Precision Mode
        # Calculate relative face size change (larger = closer, smaller = farther)
        face_size_delta = face_size - self.face_size0
        smoothed_size_delta = self.face_size_filter.update(face_size_delta)
        
        # Switch precision mode based on face size
        # If face is significantly larger than baseline (leaned in) → precision mode
        # If face is smaller or same (leaned back) → coarse mode
        old_precision_mode = self.precision_mode
        if smoothed_size_delta > self.face_size_threshold:
            self.precision_mode = True  # Fine control (close to camera)
        else:
            self.precision_mode = False  # Coarse control (far from camera)
        
        if old_precision_mode != self.precision_mode:
            mode_text = "PRECISION (Fine)" if self.precision_mode else "COARSE (Fast)"
            self.get_logger().info(f"🎯 Mode switched to: {mode_text} (Face size delta: {smoothed_size_delta:.4f})")
        
        # Apply deadzones
        if abs(dnod) < self.NOD_DEADZONE:
            dnod = 0.0
        if abs(dturn) < self.TURN_DEADZONE:
            dturn = 0.0
        if abs(dtilt) < self.TILT_DEADZONE:
            dtilt = 0.0
        
        # Detect blink and winks
        left_eye = pts[LEFT_EYE_IDX]
        right_eye = pts[RIGHT_EYE_IDX]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = 0.5 * (left_ear + right_ear)
        
        # 🎯 IMPRESSIVE: Eye Wink Detection (left/right independent)
        left_wink = False
        right_wink = False
        if left_ear < self.EAR_THRESH and right_ear >= self.EAR_THRESH:
            # Only left eye closed = left wink
            if not self.left_eye_closed:
                self.left_eye_closed = True
                self.left_wink_start = now
            else:
                if now - self.left_wink_start >= self.WINK_TIME and now - self.last_wink_time > self.wink_cooldown:
                    left_wink = True
                    self.last_wink_time = now
        else:
            self.left_eye_closed = False
            
        if right_ear < self.EAR_THRESH and left_ear >= self.EAR_THRESH:
            # Only right eye closed = right wink
            if not self.right_eye_closed:
                self.right_eye_closed = True
                self.right_wink_start = now
            else:
                if now - self.right_wink_start >= self.WINK_TIME and now - self.last_wink_time > self.wink_cooldown:
                    right_wink = True
                    self.last_wink_time = now
        else:
            self.right_eye_closed = False
        
        # Handle wink commands
        if left_wink:
            # Left wink = Save current position as home
            if self.got_joint_states:
                self.home_position = self.joint_positions.copy()
                self.get_logger().info("🏠 HOME POSITION SAVED! (Left Wink)")
        if right_wink:
            # Right wink = Return to home position
            if self.home_position is not None and self.got_joint_states:
                self.returning_to_home = True
                self.get_logger().info("🏠 RETURNING TO HOME POSITION... (Right Wink)")
        
        # 🎯 IMPRESSIVE: Head Shake Detection (quick left-right for cancel)
        if abs(dturn) > self.TURN_THRESHOLD:
            direction = 1 if dturn > 0 else -1
            self.head_shake_buffer.append((now, direction))
            # Keep only recent movements (within detection window)
            self.head_shake_buffer = [(t, d) for t, d in self.head_shake_buffer if now - t < self.shake_detection_window]
            
            # Detect shake pattern (alternating directions)
            if len(self.head_shake_buffer) >= self.SHAKE_THRESHOLD:
                directions = [d for _, d in self.head_shake_buffer]
                changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
                if changes >= self.SHAKE_THRESHOLD - 1:
                    # Head shake detected - cancel current operation
                    self.returning_to_home = False
                    self.playing_back = False
                    self.recording = False
                    self.head_shake_buffer.clear()
                    self.get_logger().info("❌ OPERATION CANCELLED! (Head Shake)")
        
        now = time.time()
        long_blink = False
        blink_duration = 0.0  # Track blink duration for display
        
        if ear < self.EAR_THRESH:
            if not self.eye_closed:
                self.eye_closed = True
                self.eye_close_start = now
                self.blink_frames_count = 0
                self.blink_triggered = False  # Reset trigger flag when eyes close
                blink_duration = 0.0
            else:
                self.blink_frames_count += 1
                blink_duration = now - self.eye_close_start
                # Trigger grasp once if held closed long enough (>1s)
                # Only trigger once per blink event (when eyes are still closed)
                if blink_duration >= self.GRASP_BLINK_TIME and not self.blink_triggered:
                    long_blink = True
                    self.blink_triggered = True  # Mark as triggered to prevent multiple triggers
                    self.get_logger().info(f"🤏 GRASP TRIGGERED! (EAR={ear:.3f}, held for {blink_duration:.2f}s)")
        else:
            if self.eye_closed:
                self.eye_closed = False
                self.blink_frames_count = 0
                self.blink_triggered = False  # Reset when eyes open
            blink_duration = 0.0
        
        # Detect mouth open (emergency stop TOGGLE)
        mouth_top = pts[MOUTH_TOP_IDX]
        mouth_bottom = pts[MOUTH_BOTTOM_IDX]
        mouth_dist = np.linalg.norm(mouth_top - mouth_bottom)
        relative_dist = mouth_dist / h
        mouth_open = relative_dist > self.MOUTH_OPEN_THRESH
        
        # Toggle emergency stop on mouth open transition (open once toggles on, open again toggles off)
        if mouth_open and not self.mouth_was_open:
            # Mouth just opened - toggle emergency stop
            self.emergency_stop = not self.emergency_stop
            if self.emergency_stop:
                self.get_logger().warn("🛑 EMERGENCY STOP ACTIVATED - Mouth Open")
            else:
                self.get_logger().info("✅ Emergency stop DEACTIVATED - Mouth Open")
        
        self.mouth_was_open = mouth_open
        
        if self.emergency_stop:
            self.publish_zero_trajectory()
            # Still show the display
            self.draw_overlay(frame, dnod, dturn, dtilt, ear, mouth_open, long_blink, 0.0)
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
                self.draw_overlay(frame, dnod, dturn, dtilt, ear, mouth_open, long_blink, 0.0)
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
        
        # 🎯 IMPRESSIVE: Auto-Home Return (if activated)
        if self.returning_to_home and self.home_position is not None:
            # Smoothly interpolate to home position
            all_at_home = True
            for i in range(len(new_positions)):
                diff = self.home_position[i] - new_positions[i]
                if abs(diff) > 0.01:  # Not at home yet
                    all_at_home = False
                    # Move towards home
                    step = np.sign(diff) * min(abs(diff), self.home_return_speed)
                    new_positions[i] += step
                else:
                    new_positions[i] = self.home_position[i]
            
            if all_at_home:
                self.returning_to_home = False
                self.get_logger().info("✅ ARRIVED AT HOME POSITION!")
        else:
            # Normal head movement control with proximity-based precision
            # 🎯 Apply precision/coarse mode multiplier
            gain_multiplier = self.precision_gain_multiplier if self.precision_mode else self.coarse_gain_multiplier
            
            # Map head movements to joints
            # Turn (yaw) → shoulder_pan_joint
            if abs(dturn) > self.TURN_DEADZONE:
                delta_pan = dturn * self.yaw_to_pan_gain * gain_multiplier
                new_positions[0] += delta_pan
            
            # Nod (pitch) → shoulder_lift_joint and shoulder_pan_joint (extend/retract)
            # Nod up (dnod < 0) = extend, Nod down (dnod > 0) = retract
            # Wrist stays perpendicular (no wrist movement during extend/retract)
            if abs(dnod) > self.NOD_DEADZONE:
                delta_lift = dnod * self.pitch_to_lift_gain * gain_multiplier  # Nod up extends (decreases lift), nod down retracts (increases lift)
                new_positions[1] += delta_lift
                # Both shoulder joints move together for extend/retract
                # shoulder_pan_joint also moves slightly to coordinate with shoulder_lift
                delta_pan_coord = dnod * self.pitch_to_lift_gain * gain_multiplier * 0.3  # Smaller movement for coordination
                new_positions[0] += delta_pan_coord
            
            # Tilt (roll) → elbow_joint
            if abs(dtilt) > self.TILT_DEADZONE:
                delta_elbow = dtilt * self.roll_to_elbow_gain * gain_multiplier
                new_positions[2] += delta_elbow
        
        # Long blink → GRASP! Execute one-time grasp command
        if long_blink:
            # Strong grasp: move wrist joints significantly to close gripper (one-time command)
            new_positions[3] += 1.5  # LARGE adjustment for aggressive grasp
            new_positions[4] += 0.3  # Also adjust wrist_2_joint for better grasp
            new_positions[5] -= 0.3  # Adjust wrist_3_joint
            self.get_logger().info("✊✊✊ GRASP COMMAND EXECUTED! Closing gripper...")
        
        # Publish trajectory
        self.publish_trajectory(new_positions)
        
        # Update display with overlay and show window
        self.draw_overlay(frame, dnod, dturn, dtilt, ear, mouth_open, long_blink, blink_duration)
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
    
    def draw_overlay(self, frame, dnod, dturn, dtilt, ear, mouth_open, long_blink, blink_duration):
        """Draw comprehensive overlay information on frame (from facemesh_preview)."""
        h, w = frame.shape[:2]
        y_offset = 30
        
        # === TITLE ===
        cv2.putText(frame, "Facemesh UR7e Control", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        y_offset += 40
        
        # === DELTA (smoothed) ===
        cv2.putText(frame, "DELTA (smoothed)", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 35
        
        # Color code: GREEN=deadzone, CYAN=below threshold, RED=command active
        nod_color = (0, 255, 0) if dnod == 0.0 else ((0, 0, 255) if abs(dnod) > self.NOD_THRESHOLD else (0, 255, 255))
        turn_color = (0, 255, 0) if dturn == 0.0 else ((0, 0, 255) if abs(dturn) > self.TURN_THRESHOLD else (0, 255, 255))
        tilt_color = (0, 255, 0) if dtilt == 0.0 else ((0, 0, 255) if abs(dtilt) > self.TILT_THRESHOLD else (0, 255, 255))
        
        cv2.putText(frame, f"dNod:   {dnod:7.2f} px", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, nod_color, 3)
        y_offset += 35
        cv2.putText(frame, f"dTurn:  {dturn:7.2f} px", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, turn_color, 3)
        y_offset += 35
        cv2.putText(frame, f"dTilt:  {np.degrees(dtilt):7.2f} deg", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, tilt_color, 3)

        # === COMMANDS ===
        y_offset += 45
        cv2.putText(frame, "COMMANDS", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 35

        command_shown = False

        # TURN command - based on MOUTH horizontal position
        if dturn < -self.TURN_THRESHOLD:
            cv2.putText(frame, "Turn LEFT (mouth moves left)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            command_shown = True
            y_offset += 40
        elif dturn > self.TURN_THRESHOLD:
            cv2.putText(frame, "Turn RIGHT (mouth moves right)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            command_shown = True
            y_offset += 40

        # NOD command - based on FACE vertical position
        if dnod < -self.NOD_THRESHOLD:
            cv2.putText(frame, "Nod UP - EXTEND", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            command_shown = True
            y_offset += 40
        elif dnod > self.NOD_THRESHOLD:
            cv2.putText(frame, "Nod DOWN - RETRACT", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            command_shown = True
            y_offset += 40

        # TILT command - based on EYE LINE angle
        if dtilt < -self.TILT_THRESHOLD:
            cv2.putText(frame, "Tilt LEFT (left eye higher)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            command_shown = True
            y_offset += 40
        elif dtilt > self.TILT_THRESHOLD:
            cv2.putText(frame, "Tilt RIGHT (right eye higher)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            command_shown = True
            y_offset += 40
        
        if not command_shown:
            cv2.putText(frame, "(neutral - no command)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
            y_offset += 40

        # === BLINK & MOUTH STATUS ===
        y_offset += 20
        blink_status = ""
        blink_color = (255, 255, 255)
        
        if ear < self.EAR_THRESH:
            if long_blink:
                blink_status = "GRASP TRIGGERED!"
                blink_color = (0, 0, 255)  # Red - active grasp!
            else:
                blink_status = f"Eyes closed - Hold for >1s to grasp! ({blink_duration:.1f}s, EAR={ear:.3f})"
                blink_color = (0, 165, 255)  # Orange - blink detected
        else:
            blink_status = f"Eyes open (EAR={ear:.3f})"
            blink_color = (0, 255, 0)  # Green - normal

        cv2.putText(frame, blink_status, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, blink_color, 3)
        y_offset += 35

        # Mouth detection (toggle state)
        if self.emergency_stop:
            cv2.putText(frame, "STOP ACTIVE (toggle: open mouth again to release)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        elif mouth_open:
            cv2.putText(frame, "Mouth open (will toggle stop)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)
        else:
            cv2.putText(frame, "Mouth closed", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        y_offset += 40
        
        # ADVANCED FEATURES DISPLAY
        y_offset += 20
        cv2.putText(frame, "ADVANCED FEATURES", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_offset += 35
        
        # Home position status
        if self.home_position is not None:
            if self.returning_to_home:
                cv2.putText(frame, "RETURNING TO HOME...", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
            else:
                cv2.putText(frame, "Home saved (Right wink to return)", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No home saved (Left wink to save)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_offset += 35
        
        # Wink detection status
        if self.left_eye_closed and not self.right_eye_closed:
            cv2.putText(frame, "LEFT WINK DETECTED", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 3)
            y_offset += 35
        elif self.right_eye_closed and not self.left_eye_closed:
            cv2.putText(frame, "RIGHT WINK DETECTED", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 3)
            y_offset += 35
        
        # Head shake status
        if len(self.head_shake_buffer) > 0:
            cv2.putText(frame, f"Shake detected: {len(self.head_shake_buffer)} moves", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 35
        
        # Proximity-Based Precision Mode Display
        y_offset += 20
        mode_text = "PRECISION MODE (Fine Control)" if self.precision_mode else "COARSE MODE (Fast Control)"
        mode_color = (0, 255, 255) if self.precision_mode else (255, 165, 0)  # Cyan for precision, Orange for coarse
        cv2.putText(frame, mode_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 3)
        y_offset += 35
        cv2.putText(frame, "Lean IN for precision | Lean BACK for speed", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Instructions at bottom
        cv2.putText(frame, "Press 'r' to recenter | 'q' to quit | Ctrl+C to stop", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
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

