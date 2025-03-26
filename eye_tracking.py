import cv2
import numpy as np
import mediapipe as mp
import time

# =========================================
#          MEDIAPIPE & IRIS SETUP
# =========================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,  # needed for iris detection
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye corners (landmark indices)
LEFT_EYE_LEFT_CORNER  = 33    # outer corner of left eye
LEFT_EYE_RIGHT_CORNER = 133   # inner corner
RIGHT_EYE_LEFT_CORNER = 362   # outer corner of right eye
RIGHT_EYE_RIGHT_CORNER= 263   # inner corner

# Iris (landmark indices)
LEFT_IRIS_LANDMARKS  = [468, 469, 470, 471]
RIGHT_IRIS_LANDMARKS = [473, 474, 475, 476]

def average_landmark_x(landmarks, indices):
    """Returns the average x of given landmark indices."""
    xs = [landmarks[i].x for i in indices if i < len(landmarks)]
    return sum(xs)/len(xs) if len(xs) else None

def compute_eye_ratio(landmarks, iris_indices, left_corner_idx, right_corner_idx):
    """
    Returns a horizontal ratio [0..1] indicating how far the iris center is
    between the left and right eye corners. 0 = far left, 1 = far right.
    """
    if not landmarks or len(landmarks) < max(iris_indices+[left_corner_idx,right_corner_idx]):
        return None

    iris_center_x = average_landmark_x(landmarks, iris_indices)
    if iris_center_x is None:
        return None

    left_corner_x = landmarks[left_corner_idx].x
    right_corner_x = landmarks[right_corner_idx].x

    # Ensure left < right
    if left_corner_x > right_corner_x:
        left_corner_x, right_corner_x = right_corner_x, left_corner_x

    width = right_corner_x - left_corner_x
    if width == 0:
        return None

    ratio = (iris_center_x - left_corner_x) / width
    return ratio

def get_avg_eye_ratio(landmarks):
    """
    Average the left and right eye ratios for a single horizontal ratio.
    """
    left_ratio = compute_eye_ratio(
        landmarks, LEFT_IRIS_LANDMARKS, LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER
    )
    right_ratio = compute_eye_ratio(
        landmarks, RIGHT_IRIS_LANDMARKS, RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER
    )
    if left_ratio is not None and right_ratio is not None:
        return (left_ratio + right_ratio) / 2.0
    return None

# =========================================
#           IMAGES FOR THE TEST
# =========================================
# Use the 5 paths you mentioned:
IMAGE_PATHS = [
    "/Users/josemiguel/Downloads/TemplateMatching-FeatureMatching/box_in_scene.png",
    "/Users/josemiguel/Downloads/TemplateMatching-FeatureMatching/box.png",
    "/Users/josemiguel/Downloads/TemplateMatching-FeatureMatching/mario_coin.jpg",
    "/Users/josemiguel/Downloads/TemplateMatching-FeatureMatching/mario.jpg",
    "/Users/josemiguel/Downloads/TemplateMatching-FeatureMatching/messi5.jpg"
]

loaded_images = []
for path in IMAGE_PATHS:
    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image: {path}")
    else:
        # Resize each to 200x200
        img = cv2.resize(img, (200, 200))
        loaded_images.append(img)

if len(loaded_images) < 5:
    raise ValueError("Fewer than 5 images were loaded. Check the IMAGE_PATHS.")

# Combine horizontally
image_strip = np.hstack(loaded_images)
strip_height, strip_width = image_strip.shape[:2]

# We'll display this strip on top and the webcam feed below it.

# =========================================
#           1) CALIBRATION PHASE
# =========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("=== CALIBRATION PHASE ===")
print("We'll show 3 points horizontally. Look at each red dot for ~2s.")
print("Press 'q' any time to skip early.\n")

# We'll calibrate with 3 horizontal target points: 20%, 50%, 80% of the screen width
calib_targets = [0.2, 0.5, 0.8]
calib_ratios = []

for target_x in calib_targets:
    print(f"[CALIB] Look at horizontal {target_x:.1f}")
    start_t = time.time()
    ratio_samples = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        dot_x = int(target_x * width)
        dot_y = height // 2
        # Draw a red dot
        cv2.circle(frame, (dot_x, dot_y), 20, (0, 0, 255), -1)

        # Process with Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ratio = get_avg_eye_ratio(lm)
            if ratio is not None:
                ratio_samples.append(ratio)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if time.time() - start_t > 2.0:
            break

    # Compute average ratio for this point
    if ratio_samples:
        avg_ratio = sum(ratio_samples) / len(ratio_samples)
    else:
        avg_ratio = 0.5  # fallback
    calib_ratios.append(avg_ratio)
    print(f"  -> ratio ~{avg_ratio:.2f} for x={target_x:.1f}")

cv2.destroyWindow("Calibration")

# Fit a linear model: screen_x = a * ratio + b
# Using numpy.polyfit with degree=1
a, b = 1.0, 0.0
if len(calib_targets) >= 2:
    a, b = np.polyfit(np.array(calib_ratios), np.array(calib_targets), 1)

print(f"\nFitted linear model: x_norm = {a:.3f} * ratio + {b:.3f}\n")

# =========================================
#           2) TEST PHASE
# =========================================
print("=== TEST PHASE with 5 images ===")
print("We'll track your gaze horizontally for ~10 seconds.")
print("Segments are 0.0–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.8, 0.8–1.0.\n")

attention_time = [0, 0, 0, 0, 0]
start_test = time.time()
last_frame_time = start_test

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    predicted_x_norm = None
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        ratio = get_avg_eye_ratio(lm)
        if ratio is not None:
            # Apply calibration:
            pred = a*ratio + b
            # clamp [0..1]
            pred = max(0.0, min(1.0, pred))
            predicted_x_norm = pred

    # Determine which image is presumably looked at
    looked_index = -1
    if predicted_x_norm is not None:
        # 5 segments: 0.0–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.8, 0.8–1.0
        if 0.0 <= predicted_x_norm < 0.2:
            looked_index = 0
        elif 0.2 <= predicted_x_norm < 0.4:
            looked_index = 1
        elif 0.4 <= predicted_x_norm < 0.6:
            looked_index = 2
        elif 0.6 <= predicted_x_norm < 0.8:
            looked_index = 3
        else:
            looked_index = 4

    # Make a copy of the strip and highlight the "looked at" image
    display_strip = image_strip.copy()
    if 0 <= looked_index < 5:
        x_start = looked_index * 200
        cv2.rectangle(display_strip, (x_start, 0), (x_start+200, 200), (0,255,0), 2)

    # Combine top strip + webcam feed below
    resized_frame = cv2.resize(frame, (strip_width, 300))
    combined_display = np.vstack((display_strip, resized_frame))

    # Time delta for accumulating attention time
    current_time = time.time()
    dt = current_time - last_frame_time
    last_frame_time = current_time

    if looked_index >= 0:
        attention_time[looked_index] += dt

    # Add text
    if looked_index >= 0:
        cv2.putText(combined_display,
                    f"Looking at image #{looked_index+1}",
                    (10, strip_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0,255,0),
                    2
        )
    else:
        cv2.putText(combined_display,
                    "Eyes not detected",
                    (10, strip_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0,0,255),
                    2
        )

    cv2.imshow("Test with 5 Images", combined_display)

    # End after 10 seconds or if user presses 'q'
    if current_time - start_test > 10.0:
        print("10 seconds elapsed, ending test.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User pressed 'q', ending test early.")
        break

cap.release()
cv2.destroyAllWindows()

# =========================================
#       SUMMARY OF RESULTS
# =========================================
print("\n=== Eye Tracking Results ===")
for i, t in enumerate(attention_time, start=1):
    print(f"Image #{i}: {t:.2f} seconds")

max_i = max(range(len(attention_time)), key=lambda i: attention_time[i])
print(f"\nMost attention spent on Image #{max_i + 1} with ~{attention_time[max_i]:.2f} s.")
