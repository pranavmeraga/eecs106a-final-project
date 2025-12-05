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


# Face landmark indices
POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]  # stable subset for solvePnP
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14


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


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """Calculate eye aspect ratio (EAR) for blink detection."""
    p1, p2, p3, p4, p5, p6 = eye_pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    return (v1 + v2) / (2.0 * h + 1e-6)


def main(camera_index: int = 0) -> None:
    # MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Camera setup
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    print("=" * 60)
    print("Face Mesh Preview - Head Pose & Gesture Detection")
    print("=" * 60)
    print("Commands to test:")
    print("  • Turn head LEFT/RIGHT  → Yaw (base rotation)")
    print("  • Nod UP/DOWN           → Pitch (limb vertical)")
    print("  • Tilt head LEFT/RIGHT  → Roll (limb forward/back)")
    print("  • Hold blink >1s        → Grasp")
    print("  • Open mouth wide       → Emergency Stop")
    print("-" * 60)
    print("Press 'q' to quit, 'r' to reset neutral pose")
    print("=" * 60)

    # Neutral pose tracking
    neutral_set = False
    yaw0, pitch0, roll0 = 0.0, 0.0, 0.0
    
    # Blink tracking
    eye_closed = False
    eye_close_start = 0.0
    EAR_THRESH = 0.21
    GRASP_BLINK_TIME = 1.0
    
    # Mouth tracking
    MOUTH_OPEN_THRESH = 0.03

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed, exiting.")
            break

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
                cv2.putText(frame, "=== DELTA (from neutral) ===", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
                
                # Color code based on threshold
                yaw_color = (0, 0, 255) if abs(dyaw) > 0.15 else (0, 255, 255)
                pitch_color = (0, 0, 255) if abs(dpitch) > 0.15 else (0, 255, 255)
                roll_color = (0, 0, 255) if abs(droll) > 0.15 else (0, 255, 255)
                
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

                # Yaw command
                if dyaw < -0.15:
                    cv2.putText(frame, "Turn LEFT", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                elif dyaw > 0.15:
                    cv2.putText(frame, "Turn RIGHT", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y_offset += 25

                # Pitch command
                if dpitch < -0.15:
                    cv2.putText(frame, "Nod DOWN", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                elif dpitch > 0.15:
                    cv2.putText(frame, "Nod UP", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y_offset += 25

                # Roll command
                if droll < -0.15:
                    cv2.putText(frame, "Rotate LEFT (tilt left)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                elif droll > 0.15:
                    cv2.putText(frame, "Rotate RIGHT (tilt right)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Detect blink
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

            # Detect mouth open
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

            # Draw face mesh
            for lm_point in result.multi_face_landmarks[0].landmark:
                x, y = int(lm_point.x * w), int(lm_point.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

            # Highlight key landmarks
            for idx in POSE_LANDMARKS:
                x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)

        else:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Facemesh Preview - Head Pose Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            neutral_set = False
            print("Neutral pose reset - reposition and wait...")

    cap.release()
    mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face mesh preview with head pose detection")
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    args = parser.parse_args()
    
    main(camera_index=args.camera)