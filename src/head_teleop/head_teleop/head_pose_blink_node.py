"""
head_pose_blink_node.py

Detects facial motions and publishes raw data.
Does NOT control robot directly.

Publishes:
- /head_pose (Vector3): x=yaw, y=pitch, z=roll (deltas from neutral)
- /blink_event (Int8): 0=none, 1=long_blink
- /mouth_open (Bool): True if mouth open
"""

import cv2
import time
import numpy as np
import mediapipe as mp

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int8, Bool


POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]

# Blink events
BLINK_NONE = 0
BLINK_LONG = 1


def rotation_matrix_to_euler_angles(R: np.ndarray) -> tuple:
    """Convert rotation matrix to Euler angles (roll, pitch, yaw)."""
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


def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (yaw, pitch, roll)."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


class HeadPoseBlinkNode(Node):
    def __init__(self):
        super().__init__('head_pose_blink_node')

        # Publishers - raw sensor data only
        self.head_pose_pub = self.create_publisher(Vector3, 'head_pose', 10)
        self.blink_pub = self.create_publisher(Int8, 'blink_event', 10)
        self.mouth_pub = self.create_publisher(Bool, 'mouth_open', 10)

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Camera not opened")
            raise RuntimeError("Camera failed")

        # Neutral pose
        self.neutral_set = False
        self.yaw0 = 0.0
        self.pitch0 = 0.0
        self.roll0 = 0.0

        # Deadzone
        self.deadzone = 0.02

        # Eye landmarks
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [263, 387, 385, 362, 380, 373]
        
        # Mouth landmarks
        self.mouth_top_idx = 13
        self.mouth_bottom_idx = 14
        
        # Blink tracking
        self.eye_closed = False
        self.eye_close_start = 0.0
        self.EAR_THRESH = 0.21
        self.GRASP_BLINK_TIME = 1.0

        # Mouth tracking
        self.MOUTH_OPEN_THRESH = 0.03

        self.timer = self.create_timer(1.0 / 30.0, self.loop)
        self.get_logger().info("Head Pose Detection Node Started")

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            # No face - publish zeros
            self.publish_head_pose(0.0, 0.0, 0.0)
            self.publish_blink(BLINK_NONE)
            self.publish_mouth(False)
            return

        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[l.x * w, l.y * h] for l in lm])

        # Calculate head pose
        roll, pitch, yaw = self.calculate_head_pose(lm, w, h)
        if roll is None:
            self.publish_head_pose(0.0, 0.0, 0.0)
            self.publish_blink(BLINK_NONE)
            self.publish_mouth(False)
            return

        # Set neutral
        if not self.neutral_set:
            self.yaw0 = float(yaw)
            self.pitch0 = float(pitch)
            self.roll0 = float(roll)
            self.neutral_set = True
            self.get_logger().info(
                f"Neutral: yaw={np.degrees(self.yaw0):.1f}° "
                f"pitch={np.degrees(self.pitch0):.1f}° "
                f"roll={np.degrees(self.roll0):.1f}°"
            )

        # Calculate deltas
        dyaw = float(yaw) - self.yaw0
        dpitch = float(pitch) - self.pitch0
        droll = float(roll) - self.roll0

        # Apply deadzone
        if abs(dyaw) < self.deadzone:
            dyaw = 0.0
        if abs(dpitch) < self.deadzone:
            dpitch = 0.0
        if abs(droll) < self.deadzone:
            droll = 0.0

        # Publish raw data
        self.publish_head_pose(dyaw, dpitch, droll)
        self.publish_blink(self.detect_blink(pts))
        self.publish_mouth(self.detect_mouth_open(pts, h))

    def calculate_head_pose(self, lm, w: int, h: int) -> tuple:
        """Calculate head pose using solvePnP."""
        try:
            pts_2d = np.array(
                [[lm[idx].x * w, lm[idx].y * h] for idx in POSE_LANDMARKS],
                dtype=np.float64
            )
            pts_3d = np.array(
                [[lm[idx].x * w, lm[idx].y * h, lm[idx].z * 300] for idx in POSE_LANDMARKS],
                dtype=np.float64
            )

            cam_matrix = np.array([
                [w, 0, w / 2],
                [0, w, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, _ = cv2.solvePnP(
                pts_3d, pts_2d, cam_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                roll, pitch, yaw = rotation_matrix_to_euler_angles(rot_mat)
                return roll, pitch, yaw
            else:
                return None, None, None

        except Exception as e:
            self.get_logger().warn(f"Pose estimation failed: {e}")
            return None, None, None

    def eye_aspect_ratio(self, eye_pts: np.ndarray) -> float:
        p1, p2, p3, p4, p5, p6 = eye_pts
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h = np.linalg.norm(p1 - p4)
        return (v1 + v2) / (2.0 * h + 1e-6)

    def detect_blink(self, pts: np.ndarray) -> int:
        """Detect long blink."""
        left_eye = pts[self.left_eye_idx]
        right_eye = pts[self.right_eye_idx]
        ear = 0.5 * (
            self.eye_aspect_ratio(left_eye) + 
            self.eye_aspect_ratio(right_eye)
        )

        now = time.time()

        if ear < self.EAR_THRESH:
            if not self.eye_closed:
                self.eye_closed = True
                self.eye_close_start = now
            else:
                duration = now - self.eye_close_start
                if duration >= self.GRASP_BLINK_TIME:
                    return BLINK_LONG
        else:
            if self.eye_closed:
                self.eye_closed = False
                
        return BLINK_NONE

    def detect_mouth_open(self, pts: np.ndarray, frame_height: int) -> bool:
        """Detect open mouth."""
        mouth_top = pts[self.mouth_top_idx]
        mouth_bottom = pts[self.mouth_bottom_idx]
        mouth_dist = np.linalg.norm(mouth_top - mouth_bottom)
        relative_dist = mouth_dist / frame_height
        return relative_dist > self.MOUTH_OPEN_THRESH

    def publish_head_pose(self, dyaw: float, dpitch: float, droll: float):
        """Publish head pose deltas: x=yaw, y=pitch, z=roll"""
        msg = Vector3()
        msg.x = float(dyaw)    # yaw (turn left/right)
        msg.y = float(dpitch)  # pitch (nod up/down)
        msg.z = float(droll)   # roll (tilt left/right)
        self.head_pose_pub.publish(msg)

    def publish_blink(self, code: int):
        msg = Int8()
        msg.data = int(code)
        self.blink_pub.publish(msg)

    def publish_mouth(self, is_open: bool):
        msg = Bool()
        msg.data = is_open
        self.mouth_pub.publish(msg)

    def shutdown(self):
        self.cap.release()
        self.face_mesh.close()


def main(args=None):
    rclpy.init(args=args)
    node = HeadPoseBlinkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()