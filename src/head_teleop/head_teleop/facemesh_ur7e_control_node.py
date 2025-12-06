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


class FacemeshUR7eControlNode(Node):
    def __init__(self, camera_index: int = 0, mirror: bool = True):
        super().__init__('facemesh_ur7e_control_node')
        
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.joint_positions = [0.0] * 6
        self.got_joint_states = False  # Failsafe: don't publish until joint states received
        
        # Subscribe to joint states
        self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            10
        )
        
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
        
        # Blink tracking
        self.eye_closed = False
        self.eye_close_start = 0.0
        self.EAR_THRESH = 0.21
        self.GRASP_BLINK_TIME = 1.0
        
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
        self.yaw_to_pan_gain = 0.01    # radians per pixel (turn → shoulder_pan)
        self.pitch_to_lift_gain = 0.01  # radians per pixel (nod → shoulder_lift)
        self.roll_to_elbow_gain = 0.5   # radians per radian (tilt → elbow)
        
        # Max joint velocity limits (radians)
        self.max_joint_velocity = 0.5
        
        # Emergency stop flag
        self.emergency_stop = False
        
        # Quit flag
        self.should_quit = False
        
        # Timer for camera loop
        self.timer = self.create_timer(1.0 / 30.0, self.camera_loop)  # 30 Hz
        
        self.get_logger().info("Facemesh UR7e Control Node Started")
        self.get_logger().info(f"Camera index: {camera_index}, Mirror: {mirror}")
        self.get_logger().info("Control Mapping:")
        self.get_logger().info("  Turn head LEFT/RIGHT → shoulder_pan_joint")
        self.get_logger().info("  Nod UP/DOWN → shoulder_lift_joint")
        self.get_logger().info("  Tilt LEFT/RIGHT → elbow_joint")
        self.get_logger().info("  Long blink → wrist_1_joint adjustment")
        self.get_logger().info("  Open mouth → Emergency stop")
        self.get_logger().info("Press 'r' in OpenCV window to recenter neutral pose")
    
    def joint_state_callback(self, msg: JointState):
        """Update current joint positions from joint_states topic."""
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.joint_positions[i] = msg.position[idx]
        self.got_joint_states = True
    
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
        
        # Detect blink
        left_eye = pts[LEFT_EYE_IDX]
        right_eye = pts[RIGHT_EYE_IDX]
        ear = 0.5 * (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye))
        
        now = time.time()
        long_blink = False
        
        if ear < self.EAR_THRESH:
            if not self.eye_closed:
                self.eye_closed = True
                self.eye_close_start = now
            else:
                duration = now - self.eye_close_start
                if duration >= self.GRASP_BLINK_TIME:
                    long_blink = True
        else:
            if self.eye_closed:
                self.eye_closed = False
        
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
            return
        
        # Reset emergency stop if mouth closes
        if self.emergency_stop and not mouth_open:
            self.emergency_stop = False
            self.get_logger().info("✅ Emergency stop released")
        
        # Don't publish if we don't have joint states yet
        if not self.got_joint_states:
            self.get_logger().warn("Waiting for joint states...", throttle_duration_sec=2.0)
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
        
        # Long blink → adjust wrist_1_joint
        if long_blink:
            new_positions[3] += 0.1  # Small adjustment
            self.get_logger().info("✊ Long blink detected - adjusting wrist")
        
        # Publish trajectory
        self.publish_trajectory(new_positions)
        
        # Update display (optional - can be removed for headless operation)
        self.draw_overlay(frame, dnod, dturn, dtilt, ear, mouth_open, long_blink)
        cv2.imshow("Facemesh UR7e Control", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self.recenter_neutral()
        elif key == ord('q'):
            self.get_logger().info("Quit requested")
            self.should_quit = True
    
    def draw_overlay(self, frame, dnod, dturn, dtilt, ear, mouth_open, long_blink):
        """Draw overlay information on frame."""
        h, w = frame.shape[:2]
        y_offset = 30
        
        # Status
        cv2.putText(frame, "Facemesh UR7e Control", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # Deltas
        nod_color = (0, 255, 0) if abs(dnod) < self.NOD_DEADZONE else ((0, 0, 255) if abs(dnod) > self.NOD_THRESHOLD else (0, 255, 255))
        turn_color = (0, 255, 0) if abs(dturn) < self.TURN_DEADZONE else ((0, 0, 255) if abs(dturn) > self.TURN_THRESHOLD else (0, 255, 255))
        tilt_color = (0, 255, 0) if abs(dtilt) < self.TILT_DEADZONE else ((0, 0, 255) if abs(dtilt) > self.TILT_THRESHOLD else (0, 255, 255))
        
        cv2.putText(frame, f"dNod: {dnod:7.2f} px", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, nod_color, 2)
        y_offset += 25
        cv2.putText(frame, f"dTurn: {dturn:7.2f} px", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, turn_color, 2)
        y_offset += 25
        cv2.putText(frame, f"dTilt: {np.degrees(dtilt):7.2f} deg", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, tilt_color, 2)
        y_offset += 30
        
        # Status indicators
        if self.emergency_stop:
            cv2.putText(frame, "🛑 EMERGENCY STOP", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif long_blink:
            cv2.putText(frame, "✊ Long Blink", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Status: OK", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, f"Mouth: {'OPEN' if mouth_open else 'CLOSED'}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if mouth_open else (0, 255, 0), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'r' to recenter | 'q' to quit | Ctrl+C to stop", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    def publish_trajectory(self, positions):
        """Publish joint trajectory to UR7e."""
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 5  # As specified in task
        traj.points.append(point)
        
        self.pub.publish(traj)
        self.joint_positions = positions.copy()
    
    def publish_zero_trajectory(self):
        """Publish zero velocity trajectory for emergency stop."""
        if not self.got_joint_states:
            return
        
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = self.joint_positions.copy()  # Hold current position
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 1
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

