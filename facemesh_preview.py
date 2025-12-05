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
    print("Press 'q' quit | 'r' RECENTER | 'm' toggle mirror")
    print("IMPORTANT: Press 'r' to RECENTER at your current position!")
    print("=" * 60)

    # Neutral pose
    neutral_set = False
    face_center_y0 = 0.0
    mouth_center_x0 = 0.0
    tilt_angle0 = 0.0
    
    # Smoothing filters
    face_y_filter = SmoothingFilter(alpha=0.3)
    mouth_x_filter = SmoothingFilter(alpha=0.3)
    tilt_filter = SmoothingFilter(alpha=0.3)
    
    # Blink tracking
    eye_closed = False
    eye_close_start = 0.0
    EAR_THRESH = 0.21
    GRASP_BLINK_TIME = 1.0
    
    # Mouth tracking
    MOUTH_OPEN_THRESH = 0.03
    
    # CLEAR THRESHOLDS - each motion is independent
    TURN_THRESHOLD = 25.0     # pixels - mouth horizontal movement (TURN)
    NOD_THRESHOLD = 25.0      # pixels - face vertical movement (NOD)
    TILT_THRESHOLD = 0.12     # radians ~6.9° - eye line angle (TILT)
    
    # LARGE DEADZONES for stable neutral
    TURN_DEADZONE = 10.0      # pixels - LARGE stable center for turn
    NOD_DEADZONE = 10.0       # pixels - stable center for nod
    TILT_DEADZONE = 0.05      # radians ~2.9° - stable center for tilt

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed, exiting.")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            pts = np.array([[l.x * w, l.y * h] for l in lm])

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
                # Calculate distinct metrics for each motion type
                face_center_y = calculate_face_center_y(lm, h)          # NOD: whole face up/down
                mouth_center_x = calculate_mouth_horizontal_position(lm, w)  # TURN: mouth left/right
                tilt_angle = calculate_head_tilt_angle(lm, w, h)       # TILT: eye line angle

                # Set neutral pose on first frame
                if not neutral_set:
                    face_center_y0 = face_center_y
                    mouth_center_x0 = mouth_center_x
                    tilt_angle0 = tilt_angle
                    neutral_set = True
                    print(f"✓ Initial neutral:")
                    print(f"   Face Y: {face_center_y0:.1f}px")
                    print(f"   Mouth X: {mouth_center_x0:.1f}px")
                    print(f"   Tilt: {np.degrees(tilt_angle0):.1f}°")
                    print("💡 PRESS 'r' TO RECENTER if not comfortable!")

                # Calculate deltas - INDEPENDENT measurements
                dnod = face_center_y - face_center_y0      # Vertical face movement
                dturn = mouth_center_x - mouth_center_x0   # Horizontal mouth movement
                dtilt = tilt_angle - tilt_angle0           # Eye line angle change
                
                # APPLY SMOOTHING
                dnod = face_y_filter.update(dnod)
                dturn = mouth_x_filter.update(dturn)
                dtilt = tilt_filter.update(dtilt)
                
                # Apply LARGE DEADZONES for stable neutral
                if abs(dnod) < NOD_DEADZONE:
                    dnod = 0.0
                if abs(dturn) < TURN_DEADZONE:
                    dturn = 0.0
                if abs(dtilt) < TILT_DEADZONE:
                    dtilt = 0.0

                # Display measurements
                y_offset = 30
                cv2.putText(frame, "=== RAW POSITIONS ===", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
                cv2.putText(frame, f"Face Y:  {face_center_y:7.1f} px", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
                cv2.putText(frame, f"Mouth X: {mouth_center_x:7.1f} px", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
                cv2.putText(frame, f"Tilt:    {np.degrees(tilt_angle):7.2f} deg", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                y_offset += 35
                cv2.putText(frame, "=== DELTA (smoothed) ===", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
                
                # Color code: GREEN=deadzone, YELLOW=below threshold, RED=command active
                nod_color = (0, 255, 0) if dnod == 0.0 else ((0, 0, 255) if abs(dnod) > NOD_THRESHOLD else (0, 255, 255))
                turn_color = (0, 255, 0) if dturn == 0.0 else ((0, 0, 255) if abs(dturn) > TURN_THRESHOLD else (0, 255, 255))
                tilt_color = (0, 255, 0) if dtilt == 0.0 else ((0, 0, 255) if abs(dtilt) > TILT_THRESHOLD else (0, 255, 255))
                
                cv2.putText(frame, f"dNod:   {dnod:7.2f} px", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, nod_color, 2)
                y_offset += 25
                cv2.putText(frame, f"dTurn:  {dturn:7.2f} px", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, turn_color, 2)
                y_offset += 25
                cv2.putText(frame, f"dTilt:  {np.degrees(dtilt):7.2f} deg", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, tilt_color, 2)

                # Commands - CLEAR SEPARATION
                y_offset += 35
                cv2.putText(frame, "=== COMMANDS ===", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25

                command_shown = False

                # TURN command - based on MOUTH horizontal position
                if dturn < -TURN_THRESHOLD:
                    cv2.putText(frame, "Turn LEFT (mouth moves left)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30
                elif dturn > TURN_THRESHOLD:
                    cv2.putText(frame, "Turn RIGHT (mouth moves right)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30

                # NOD command - based on FACE vertical position
                if dnod < -NOD_THRESHOLD:
                    cv2.putText(frame, "Nod UP (face moves up)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30
                elif dnod > NOD_THRESHOLD:
                    cv2.putText(frame, "Nod DOWN (face moves down)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30

                # TILT command - based on EYE LINE angle
                if dtilt < -TILT_THRESHOLD:
                    cv2.putText(frame, "Tilt LEFT (left eye higher)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30
                elif dtilt > TILT_THRESHOLD:
                    cv2.putText(frame, "Tilt RIGHT (right eye higher)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    command_shown = True
                    y_offset += 30
                
                if not command_shown:
                    cv2.putText(frame, "(neutral - no command)", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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

            # Highlight key landmarks
            # Eyes (for tilt)
            cv2.circle(frame, (int(lm[33].x * w), int(lm[33].y * h)), 5, (255, 0, 0), -1)  # Left eye
            cv2.circle(frame, (int(lm[263].x * w), int(lm[263].y * h)), 5, (255, 0, 0), -1)  # Right eye
            
            # Mouth corners (for turn)
            cv2.circle(frame, (int(lm[LEFT_MOUTH_CORNER].x * w), int(lm[LEFT_MOUTH_CORNER].y * h)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(lm[RIGHT_MOUTH_CORNER].x * w), int(lm[RIGHT_MOUTH_CORNER].y * h)), 5, (0, 255, 0), -1)
            
            # Face center points (for nod)
            cv2.circle(frame, (int(lm[NOSE_TIP].x * w), int(lm[NOSE_TIP].y * h)), 5, (255, 255, 0), -1)
            cv2.circle(frame, (int(lm[CHIN].x * w), int(lm[CHIN].y * h)), 5, (255, 255, 0), -1)

        else:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.putText(frame, ">>> PRESS 'r' TO RECENTER <<< | m=mirror | q=quit", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Facemesh Preview - Robust", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if result.multi_face_landmarks and success:
                face_center_y0 = face_center_y
                mouth_center_x0 = mouth_center_x
                tilt_angle0 = tilt_angle
                
                face_y_filter = SmoothingFilter(alpha=0.3)
                mouth_x_filter = SmoothingFilter(alpha=0.3)
                tilt_filter = SmoothingFilter(alpha=0.3)
                
                print(f"\n✅ RECENTERED SUCCESSFULLY!")
                print(f"   Face Y: {face_center_y0:.1f}px (for NOD)")
                print(f"   Mouth X: {mouth_center_x0:.1f}px (for TURN)")
                print(f"   Tilt: {np.degrees(tilt_angle0):.1f}° (for TILT)")
                print(f"   All deltas should now be GREEN and 0.00")
            else:
                print("\n⚠️  Cannot recenter - no face detected")
        elif key == ord('m'):
            mirror = not mirror
            print(f"\n🔄 Mirror mode: {'ON' if mirror else 'OFF'}")

    cap.release()
    mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--no-mirror', action='store_true', help='Disable mirroring')
    args = parser.parse_args()
    
    main(camera_index=args.camera, mirror=not args.no_mirror)