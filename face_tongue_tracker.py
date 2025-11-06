"""
Tongue Detection Meme Display - Robust version for Python 3.12.10
(Neutral/Tongue version)

This version:
- Uses neutral_image.jpg (normal face)
- Uses tongue_image.jpg (tongue out)
- Faster and smoother than the default MediaPipe FaceMesh loop
"""

import cv2
import mediapipe as mp
import numpy as np
import os

# -------------------------
# CONFIGURATION
# -------------------------
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720

TONGUE_OUT_THRESHOLD = 0.12   # Adjust sensitivity; lower = more sensitive
DOWNSCALE_RATIO = 0.5         # Process smaller frame for performance
PROCESS_EVERY_N_FRAMES = 2    # Run detector every N frames
STABILITY_REQUIRED = 3        # Frames to confirm detection
STABILITY_RELEASE = 1         # Frames to confirm not detected
DEBUG_OVERLAY = False         # Toggle with 'd'

# -------------------------
# Helper Functions
# -------------------------
def safe_imread(path):
    
    img = cv2.imread(path)
    
    if img is None:
        print(f"[ERROR] Could not load image: {path}")
    
    return img

def mouth_open_ratio_from_landmarks(face_landmarks):
    """Calculate mouth opening vs. face height (normalized)."""
    
    up = face_landmarks.landmark[13]
    lo = face_landmarks.landmark[14]
    ys = [lm.y for lm in face_landmarks.landmark]
    face_h = max(1e-6, max(ys) - min(ys))
    
    return max(0.0, lo.y - up.y), face_h

# -------------------------
# Load images
# -------------------------
neutral_path = "neutral_image.jpg"
tongue_path = "tongue_image.jpg"

if not os.path.exists(neutral_path) or not os.path.exists(tongue_path):
    print("[ERROR] Required images not found in working directory.")
    print("Make sure 'neutral_image.jpg' and 'tongue_image.jpg' are in the same folder as this script.")
    raise SystemExit(1)

neutral_img = safe_imread(neutral_path)
tongue_img = safe_imread(tongue_path)
if neutral_img is None or tongue_img is None:
    raise SystemExit(1)

# Resize to window
neutral_img = cv2.resize(neutral_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
tongue_img = cv2.resize(tongue_img, (WINDOW_WIDTH, WINDOW_HEIGHT))

# -------------------------
# Initialize webcam
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam. Check permissions or camera use.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

ret, frame = cap.read()
if not ret:
    print("[ERROR] Could not read from webcam.")
    cap.release()
    raise SystemExit(1)

# -------------------------
# MediaPipe Face Mesh
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------
# Display Windows
# -------------------------
cv2.namedWindow("Camera Input", cv2.WINDOW_NORMAL)
cv2.namedWindow("Meme Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Input", WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow("Meme Output", WINDOW_WIDTH, WINDOW_HEIGHT)

print("[OK] Running Tongue Detection Meme Display")
print("[INFO] Press 'q' to quit, '+'/'-' to adjust sensitivity, 'd' for debug overlay.")

# -------------------------
# Main Loop
# -------------------------
frame_counter = 0
tongue_out_state = False
detect_counter = 0
no_detect_counter = 0
threshold = TONGUE_OUT_THRESHOLD

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural feel
        display_frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        small = cv2.resize(display_frame, (0, 0), fx=DOWNSCALE_RATIO, fy=DOWNSCALE_RATIO)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        tongue_detected = False

        if (frame_counter % PROCESS_EVERY_N_FRAMES) == 0:
            results = face_mesh.process(rgb_small)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                mouth_open, face_h = mouth_open_ratio_from_landmarks(face_landmarks)
                ratio = mouth_open / (face_h + 1e-9)
                tongue_detected = ratio > threshold

                if DEBUG_OVERLAY:
                    up = face_landmarks.landmark[13]
                    lo = face_landmarks.landmark[14]
                    x_up = int(up.x * WINDOW_WIDTH)
                    y_up = int(up.y * WINDOW_HEIGHT)
                    x_lo = int(lo.x * WINDOW_WIDTH)
                    y_lo = int(lo.y * WINDOW_HEIGHT)
                    cv2.circle(display_frame, (x_up, y_up), 4, (0,255,0), -1)
                    cv2.circle(display_frame, (x_lo, y_lo), 4, (0,0,255), -1)
                    cv2.putText(display_frame, f"ratio:{ratio:.3f}", (10,70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # Stabilize transitions
        if tongue_detected:
            detect_counter += 1
            no_detect_counter = 0
        else:
            no_detect_counter += 1
            detect_counter = 0

        if detect_counter >= STABILITY_REQUIRED:
            tongue_out_state = True
        elif no_detect_counter >= STABILITY_RELEASE:
            tongue_out_state = False

        current_img = tongue_img.copy() if tongue_out_state else neutral_img.copy()
        status = "TONGUE OUT!" if tongue_out_state else "No tongue detected"
        color = (0,255,0) if tongue_out_state else (0,255,255)
        cv2.putText(display_frame, status, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imshow("Camera Input", display_frame)
        cv2.imshow("Meme Output", current_img)

        frame_counter += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            threshold += 0.01
            print(f"[INFO] Threshold increased -> {threshold:.3f}")
        elif key == ord('-'):
            threshold = max(0.001, threshold - 0.01)
            print(f"[INFO] Threshold decreased -> {threshold:.3f}")
        elif key == ord('d'):
            DEBUG_OVERLAY = not DEBUG_OVERLAY
            print(f"[INFO] DEBUG_OVERLAY = {DEBUG_OVERLAY}")

finally:
    
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[OK] Application closed cleanly.")