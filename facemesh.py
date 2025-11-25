import cv2
import numpy as np
import mediapipe as mp
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool

mp_face_mesh = mp.solutions.face_mesh

class HeadTrackerNode(Node):
    def __init__(self):
        super().__init__("head_tracker")

        # Publishers
        self.pose_pub = self.create_publisher(Vector3, "/head/pose", 10)
        self.blink_pub = self.create_publisher(Bool, "/head/blink", 10)
        self.mouth_pub = self.create_publisher(Bool, "/head/mouth_open", 10)
        self.eyebrow_pub = self.create_publisher(Bool, "/head/eyebrow_raise", 10)

        # MediaPipe model
        self.mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

        # Indices
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [362, 385, 387, 263, 373, 380]
        self.upper_lip, self.lower_lip = 13, 14
        self.left_brow, self.left_eye_top = 70, 159

        # OpenCV camera
        self.cap = cv2.VideoCapture(0)

        # Timer
        self.create_timer(0.03, self.update)  # ~30 Hz

        # Calibration
        self.brow_baseline = None

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return

        lm = result.multi_face_landmarks[0].landmark

        # Convert face mesh landmarks to 2D,3D numpy arrays
        pts_2d = []
        pts_3d = []
        for idx in [33,263,1,61,291,199]:  # canonical stable subset
            x = lm[idx].x * w
            y = lm[idx].y * h
            z = lm[idx].z * 300  # scaled depth
            pts_2d.append([x, y])
            pts_3d.append([x, y, z])

        pts_2d = np.array(pts_2d, dtype=np.float64)
        pts_3d = np.array(pts_3d, dtype=np.float64)

        # Camera matrix
        focal_length = w
        cam_matrix = np.array([[focal_length, 0, w/2],
                               [0, focal_length, h/2],
                               [0, 0, 1]])

        dist_coeffs = np.zeros((4, 1))

        # Solve head pose (yaw,pitch,roll)
        _, rot_vec, _ = cv2.solvePnP(pts_3d, pts_2d, cam_matrix, dist_coeffs)
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        rpy = rotationMatrixToEulerAngles(rot_mat)
        yaw, pitch, roll = rpy

        # Publish head pose
        pose_msg = Vector3()
        pose_msg.x = float(yaw)
        pose_msg.y = float(pitch)
        pose_msg.z = float(roll)
        self.pose_pub.publish(pose_msg)

        # Gesture detection
        self.detect_blink(lm, w, h)
        self.detect_mouth(lm, w, h)
        self.detect_eyebrow(lm, w, h)

    # ---------- Gesture Metric Helpers ----------

    def eye_aspect_ratio(self, lm, idxs, w, h):
        p = lambda i: np.array([lm[i].x*w, lm[i].y*h])
        A = np.linalg.norm(p(idxs[1]) - p(idxs[5]))
        B = np.linalg.norm(p(idxs[2]) - p(idxs[4]))
        C = np.linalg.norm(p(idxs[0]) - p(idxs[3]))
        return (A + B) / (2.0 * C)

    def detect_blink(self, lm, w, h):
        ear_left = self.eye_aspect_ratio(lm, self.left_eye_idx, w, h)
        ear_right = self.eye_aspect_ratio(lm, self.right_eye_idx, w, h)
        ear = (ear_left + ear_right) / 2

        blink_msg = Bool()
        blink_msg.data = ear < 0.18   # threshold
        self.blink_pub.publish(blink_msg)

    def detect_mouth(self, lm, w, h):
        up = np.array([lm[self.upper_lip].x*w, lm[self.upper_lip].y*h])
        lo = np.array([lm[self.lower_lip].x*w, lm[self.lower_lip].y*h])
        mouth_open = np.linalg.norm(up - lo)

        mouth_msg = Bool()
        mouth_msg.data = mouth_open > 20   # tuned threshold
        self.mouth_pub.publish(mouth_msg)

    def detect_eyebrow(self, lm, w, h):
        brow = np.array([lm[self.left_brow].x*w, lm[self.left_brow].y*h])
        eye = np.array([lm[self.left_eye_top].x*w, lm[self.left_eye_top].y*h])
        dist = np.linalg.norm(brow - eye)

        # initialize baseline
        if self.brow_baseline is None:
            self.brow_baseline = dist

        # detect raise
        ebr_msg = Bool()
        ebr_msg.data = dist > 1.25 * self.brow_baseline
        self.eyebrow_pub.publish(ebr_msg)

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x,y,z])

def main(args=None):
    rclpy.init(args=args)
    node = HeadTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
