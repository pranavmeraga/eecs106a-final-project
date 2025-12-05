"""
facemesh_preview.py

Quick, ROS-free facemesh visualizer for testing head pose detection.
Tests yaw, pitch, roll, blink, and mouth open detection.

Usage:
    python facemesh_preview.py
    python facemesh_preview.py --camera 1

Requirements:
    pip install opencv-python mediapipe numpy
"""

import cv2
import time
import argparse
import numpy as np
import mediapipe as mp


POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14


class SmoothingFilter:
    """Exponential moving average filter."""
    def __init__(self, alpha=0.4):
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


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """Calculate eye aspect ratio (EAR)."""
    p1, p2, p3, p4, p5, p6 = eye_pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    return (v1 + v2) / (2.0 * h + 1e-6)


def main(camera_index: int = 0, mirror: bool = True) -> None:
    mp_face_mesh = mp.solutions.face_mesh
    mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    print("=" * 60)
    print("Face Mesh Preview - ROBUST VERSION")
    print("=" * 60)
    print(f"Camera mirroring: {'ON' if mirror else 'OFF'}")
    print("Commands to test:")
    print("  • Turn head LEFT/RIGHT  → Yaw (base rotation)")
    print("  • Nod UP/DOWN           → Pitch (limb vertical)")
    print("  • Tilt head LEFT/RIGHT  → Roll (limb forward/back)")
    print("  • Hold blink >1s        → Grasp")
    print("  • Open mouth wide       → Emergency Stop")
    print("-" * 60)
    print("Press 'q' quit | 'r' reset | 'm' toggle mirror")
    print("=" * 60)

    # Neutral pose
    neutral_set = False
    yaw0, pitch0, roll0 = 0.0, 0.0, 0.0
    
    # Smoothing filters with moderate smoothing
    yaw_filter = SmoothingFilter(alpha=0.4)
    pitch_filter = SmoothingFilter(alpha=0.4)
    roll_filter = SmoothingFilter(alpha=0.4)
    
    # Blink tracking
    eye_closed = False
    eye_close_start = 0.0
    EAR_THRESH = 0.21
    GRASP_BLINK_TIME = 1.0
    
    # Mouth tracking
    MOUTH_OPEN_THRESH = 0.03
    
    # OPTIMIZED THRESHOLDS for better detection
    # Pitch needs to be MORE sensitive because nodding is smaller motion
    YAW_THRESHOLD = 0.10     # ~5.7 degrees (turn left/right)
    PITCH_THRESHOLD = 0.06   # ~3.4 degrees (nod up/down) - MORE SENSITIVE
    ROLL_THRESHOLD = 0.10    # ~5.7 degrees (tilt left/right)
    DEADZONE = 0.02          # ~1.1 degrees

    # Scaling factors for better visual feedback
    YAW_SCALE = 1.2      # Amplify yaw slightly
    PITCH_SCALE = 1.5    # Amplify pitch more (nodding is subtle)
    ROLL_SCALE = 1.2     # Amplify roll slightly

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed, exiting.")
            break

        # Mirror the frame if enabled (fixes left/right flip)
        if mirror:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            pts = np.array([[l.x * w, l.y * h] for l in lm])

            # Calculate head pose
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

                # FIX: Invert yaw if mirrored (so left is left, right is right)
                if mirror:
                    yaw = -yaw
                    roll = -roll

                # Set neutral pose
                if not neutral_set:
                    yaw0, pitch0, roll0 = yaw, pitch, roll
                    neutral_set = True
                    print(f"✓ Neutral pose set: "
                          f"yaw={np.degrees(yaw0):.1f}° "
                          f"pitch={np.degrees(pitch0):.1f}° "
                          f"roll={np.degrees(roll0):.1f}°")

                # Calculate deltas
                dyaw = yaw - yaw0
                dpitch = pitch - pitch0
                droll = roll - roll0
                
                # Apply SCALING for better sensitivity
                dyaw *= YAW_SCALE
                dpitch *= PITCH_SCALE
                droll *= ROLL_SCALE
                
                # APPLY SMOOTHING
                dyaw = yaw_filter.update(dyaw)
                dpitch = pitch_filter.update(dpitch)
                droll = roll_filter.update(droll)
                
                # Apply deadzone
                if abs(dyaw) < DEADZONE:
                    dyaw = 0.0
                if abs(dpitch) < DEADZONE:
                    dpitch = 0.0
                if abs(droll) < DEADZONE:
                    droll = 0.0

                # Display angles
                y_offset = 30
                cv2.putText(frame, "=== ABSOLUTE ANGLES ===", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
                cv2.putText(frame, f"Yaw:   {np.degrees(yaw):7.2f} deg", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
                cv2.putText(frame, f"Pitch: {np.degrees(pitch):7.2f} deg", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
                cv2.putText(frame, f"Roll:  {np.degrees(roll):7.2f} deg", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                y_offset += 35
                cv2.putText(frame, "=== DELTA (scaled+smoothed) ===", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
                
                # Color code based on threshold
                yaw_color = (0, 0, 255) if abs(dyaw) > YAW_THRESHOLD else (0, 255, 255)
                pitch_color = (0, 0, 255) if abs(dpitch) > PITCH_THRESHOLD else (0, 255, 255)
                roll_color = (0, 0, 255) if abs(droll) > ROLL_THRESHOLD else (0, 255, 255)
                
                cv2.putText(frame, f"dYaw:   {np.degrees(dyaw):7.2f} deg", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, yaw_color, 2)
                y_offset += 25
                cv2.putText(frame, f"dPitch: {np.degrees(dpitch):7.2f} deg", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, pitch_color, 2)
                y_offset += 25
                cv2.putText(frame, f"dRoll:  {np.degrees(droll):7.2f} deg", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, roll_color, 2)

                # Commands
                y_offset += 35
                cv2.putText(frame, "=== COMMANDS ===", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25

                command_shown = False

                # Yaw command (turn left/right)
                if dyaw < -YAW_THRESHOLD:
                    cv2.putText(frame, "Turn LEFT", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30
                elif dyaw > YAW_THRESHOLD:
                    cv2.putText(frame, "Turn RIGHT", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30

                # Pitch command (nod up/down) - FIXED with better threshold
                if dpitch > PITCH_THRESHOLD:  # Positive pitch = nod UP
                    cv2.putText(frame, "Nod UP", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30
                elif dpitch < -PITCH_THRESHOLD:  # Negative pitch = nod DOWN
                    cv2.putText(frame, "Nod DOWN", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30

                # Roll command (tilt left/right)
                if droll < -ROLL_THRESHOLD:
                    cv2.putText(frame, "Rotate LEFT (tilt)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30
                elif droll > ROLL_THRESHOLD:
                    cv2.putText(frame, "Rotate RIGHT (tilt)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30
                
                if not command_shown:
                    cv2.putText(frame, "(neutral - no command)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)

            # Blink detection
            left_eye = pts[LEFT_EYE_IDX]
            right_eye = pts[RIGHT_EYE_IDX]
            ear = 0.5 * (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye))

            now = time.time()
            blink_status = ""
            blink_color = (255, 255, 255)

            if ear < EAR_THRESH:
                if not eye_closed:
                    eye_closed = True
                    eye_close_start = now
                else:
                    duration = now - eye_close_start
                    if duration >= GRASP_BLINK_TIME:
                        blink_status = "GRASP (long blink)"
                        blink_color = (0, 0, 255)
                    else:
                        blink_status = f"Blinking... {duration:.1f}s"
                        blink_color = (0, 255, 255)
            else:
                if eye_closed:
                    eye_closed = False
                blink_status = f"Eyes open (EAR={ear:.3f})"
                blink_color = (0, 255, 0)

            cv2.putText(frame, blink_status, (10, h - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 2)

            # Mouth detection
            mouth_top = pts[MOUTH_TOP_IDX]
            mouth_bottom = pts[MOUTH_BOTTOM_IDX]
            mouth_dist = np.linalg.norm(mouth_top - mouth_bottom)
            relative_dist = mouth_dist / h

            if relative_dist > MOUTH_OPEN_THRESH:
                cv2.putText(frame, "STOP (mouth open)", (10, h - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Mouth closed ({relative_dist:.4f})", (10, h - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw mesh
            for lm_point in result.multi_face_landmarks[0].landmark:
                x, y = int(lm_point.x * w), int(lm_point.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

            # Highlight pose landmarks
            for idx in POSE_LANDMARKS:
                x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)

        else:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Instructions at bottom
        cv2.putText(frame, f"Mirror: {'ON' if mirror else 'OFF'} | q=quit r=reset m=mirror", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Facemesh Preview - Robust", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            neutral_set = False
            yaw_filter.reset()
            pitch_filter.reset()
            roll_filter.reset()
            print("Neutral pose reset - reposition and wait...")
        elif key == ord('m'):
            mirror = not mirror
            print(f"Mirror mode: {'ON' if mirror else 'OFF'}")

    cap.release()
    mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--no-mirror', action='store_true', help='Disable mirroring')
    args = parser.parse_args()
    
    main(camera_index=args.camera, mirror=not args.no_mirror)