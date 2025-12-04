import cv2
import time
import numpy as np
import mediapipe as mp

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Int8

BLINK_NONE = 0
BLINK_SINGLE = 1
BLINK_DOUBLE = 2
BLINK_LONG = 3


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

        self.head_pose_pub = self.create_publisher(Vector3, 'head_pose', 10)
        self.blink_pub = self.create_publisher(Int8, 'blink_event', 10)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Camera not opened")
            raise RuntimeError("Camera failed")

        # Neutral pose calibration
        self.neutral_set = False
        self.yaw0 = 0.0
        self.pitch0 = 0.0
        self.scale0 = 1.0

        # Deadzone thresholds
        self.eps_yaw = 0.02
        self.eps_pitch = 0.02
        self.eps_scale = 0.005

        # Face landmark indices for solvePnP (canonical stable subset)
        self.pose_landmark_indices = [33, 263, 1, 61, 291, 199]

        # Eye indices for blink detection
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [263, 387, 385, 362, 380, 373]
        self.eye_closed = False
        self.eye_close_start = 0.0
        self.last_single_time = 0.0

        # Blink detection parameters
        self.EAR_THRESH = 0.21
        self.LONG_BLINK_TIME = 0.7
        self.SINGLE_BLINK_MAX = 0.35
        self.DOUBLE_BLINK_WINDOW = 0.4

        self.timer = self.create_timer(1.0 / 30.0, self.loop)

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self.publish_head_pose(0.0, 0.0, 0.0)
            self.publish_blink(BLINK_NONE)
            return

        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[l.x * w, l.y * h] for l in lm])

        # Use solvePnP for more accurate head pose estimation
        pts_2d = []
        pts_3d = []
        for idx in self.pose_landmark_indices:
            x = lm[idx].x * w
            y = lm[idx].y * h
            z = lm[idx].z * 300  # scaled depth estimate
            pts_2d.append([x, y])
            pts_3d.append([x, y, z])

        pts_2d = np.array(pts_2d, dtype=np.float64)
        pts_3d = np.array(pts_3d, dtype=np.float64)

        # Camera matrix
        focal_length = w
        cam_matrix = np.array([[focal_length, 0, w / 2],
                               [0, focal_length, h / 2],
                               [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve head pose using solvePnP
        try:
            _, rot_vec, _ = cv2.solvePnP(pts_3d, pts_2d, cam_matrix, dist_coeffs)
            rot_mat, _ = cv2.Rodrigues(rot_vec)
            rpy = rotation_matrix_to_euler_angles(rot_mat)
            yaw, pitch, roll = rpy
        except Exception as e:
            self.get_logger().warn(f"Pose estimation failed: {e}")
            self.publish_head_pose(0.0, 0.0, 0.0)
            self.publish_blink(BLINK_NONE)
            return

        # Calculate scale from face width (for forward/back movement)
        left_ear_idx, right_ear_idx = 234, 454
        left_ear = pts[left_ear_idx]
        right_ear = pts[right_ear_idx]
        face_width = np.linalg.norm(left_ear - right_ear)
        scale = face_width / w if w > 0 else 0.0

        # Set neutral pose on first detection
        if not self.neutral_set:
            self.yaw0 = float(yaw)
            self.pitch0 = float(pitch)
            self.scale0 = scale if scale > 1e-3 else 1.0
            self.neutral_set = True
            self.get_logger().info("Neutral pose set")

        # Calculate deltas from neutral pose
        dyaw = float(yaw) - self.yaw0
        dpitch = float(pitch) - self.pitch0
        dscale = scale - self.scale0

        # Apply deadzone
        if abs(dyaw) < self.eps_yaw:
            dyaw = 0.0
        if abs(dpitch) < self.eps_pitch:
            dpitch = 0.0
        if abs(dscale) < self.eps_scale:
            dscale = 0.0

        self.publish_head_pose(dyaw, dpitch, dscale)
        self.publish_blink(self.detect_blink(pts))

    def publish_head_pose(self, dyaw, dpitch, dscale):
        msg = Vector3()
        msg.x, msg.y, msg.z = float(dyaw), float(dpitch), float(dscale)
        self.head_pose_pub.publish(msg)

    def publish_blink(self, code):
        msg = Int8()
        msg.data = int(code)
        self.blink_pub.publish(msg)

    def eye_aspect_ratio(self, eye_pts):
        p1, p2, p3, p4, p5, p6 = eye_pts
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h = np.linalg.norm(p1 - p4)
        return (v1 + v2) / (2.0 * h + 1e-6)

    def detect_blink(self, pts):
        left_eye = pts[self.left_eye_idx]
        right_eye = pts[self.right_eye_idx]
        ear = 0.5 * (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye))

        now = time.time()
        event = BLINK_NONE

        if ear < self.EAR_THRESH:
            if not self.eye_closed:
                self.eye_closed = True
                self.eye_close_start = now
        else:
            if self.eye_closed:
                duration = now - self.eye_close_start
                self.eye_closed = False

                if duration >= self.LONG_BLINK_TIME:
                    event = BLINK_LONG
                    self.last_single_time = 0.0
                elif duration <= self.SINGLE_BLINK_MAX:
                    if self.last_single_time > 0 and (now - self.last_single_time) <= self.DOUBLE_BLINK_WINDOW:
                        event = BLINK_DOUBLE
                        self.last_single_time = 0.0
                    else:
                        event = BLINK_SINGLE
                        self.last_single_time = now
        return event


def main(args=None):
    rclpy.init(args=args)
    node = HeadPoseBlinkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.face_mesh.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()