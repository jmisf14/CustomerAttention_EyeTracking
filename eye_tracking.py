import cv2
import time
import numpy as np
import mediapipe as mp

# Paths to your five images
IMAGE_PATHS = [
    "/Users/josemiguel/Downloads/TemplateMatching-FeatureMatching/box_in_scene.png",
    "/Users/josemiguel/Downloads/TemplateMatching-FeatureMatching/box.png",
    "/Users/josemiguel/Downloads/TemplateMatching-FeatureMatching/mario_coin.jpg",
    "/Users/josemiguel/Downloads/TemplateMatching-FeatureMatching/mario.jpg",
    "/Users/josemiguel/Downloads/TemplateMatching-FeatureMatching/messi5.jpg"
]

# Load images and combine them into one horizontal strip
loaded_images = []
for path in IMAGE_PATHS:
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not load {path}. Check file path.")
    else:
        # Resize each image to 200x200 (for simplicity)
        img = cv2.resize(img, (200, 200))
        loaded_images.append(img)

if len(loaded_images) < 1:
    raise ValueError("No images were loaded. Please check IMAGE_PATHS.")

image_strip = np.hstack(loaded_images)  # Combine horizontally
strip_height, strip_width = image_strip.shape[:2]

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,     # Important for iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start capturing from default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# We'll store how much time each image gets
attention_time = [0, 0, 0, 0, 0]

print("Starting 20-second eye-tracking session using MediaPipe with IRIS landmarks...")
print("Press 'q' to quit early.")

start_time = time.time()
last_frame_time = start_time

# ---- EYE CORNER LANDMARKS (These are approximate; feel free to adjust) ----
# Left eye corners
LEFT_EYE_LEFT_CORNER  = 33   # Outer (temporal) corner of left eye
LEFT_EYE_RIGHT_CORNER = 133  # Inner (nasal) corner of left eye

# Right eye corners
RIGHT_EYE_LEFT_CORNER  = 362  # Outer (temporal) corner of right eye
RIGHT_EYE_RIGHT_CORNER = 263  # Inner (nasal) corner of right eye

# ---- IRIS LANDMARKS ----
# Typically for left iris: [468, 469, 470, 471] (and maybe 472)
LEFT_IRIS_LANDMARKS = [468, 469, 470, 471]
# For right iris: [473, 474, 475, 476] (and maybe 477)
RIGHT_IRIS_LANDMARKS = [473, 474, 475, 476]

def average_landmark_x(landmarks, indices):
    """Returns the average x of the given landmark indices."""
    xs = [landmarks[i].x for i in indices if i < len(landmarks)]
    return sum(xs) / len(xs) if len(xs) > 0 else None

def compute_eye_ratio(landmarks, iris_indices, left_corner_idx, right_corner_idx):
    """
    Compute how far the iris center is horizontally within the eye.
    Returns a ratio in [0..1], or None if we cannot compute.
    """
    if not landmarks or len(landmarks) < max(iris_indices + [left_corner_idx, right_corner_idx]):
        return None

    iris_center_x = average_landmark_x(landmarks, iris_indices)
    if iris_center_x is None:
        return None

    left_corner_x = landmarks[left_corner_idx].x
    right_corner_x = landmarks[right_corner_idx].x

    # Make sure left_corner_x < right_corner_x for correct ratio
    # If it's reversed, swap them
    if left_corner_x > right_corner_x:
        left_corner_x, right_corner_x = right_corner_x, left_corner_x

    eye_width = (right_corner_x - left_corner_x)
    if eye_width == 0:
        return None

    ratio = (iris_center_x - left_corner_x) / eye_width
    return ratio

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face mesh
    results = face_mesh.process(rgb_frame)

    horizontal_ratio = None

    if results.multi_face_landmarks:
        # Take the first face detected (we only track one face)
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # Compute left eye ratio
        left_ratio = compute_eye_ratio(
            landmarks,
            LEFT_IRIS_LANDMARKS,
            LEFT_EYE_LEFT_CORNER,
            LEFT_EYE_RIGHT_CORNER
        )
        # Compute right eye ratio
        right_ratio = compute_eye_ratio(
            landmarks,
            RIGHT_IRIS_LANDMARKS,
            RIGHT_EYE_LEFT_CORNER,
            RIGHT_EYE_RIGHT_CORNER
        )

        if left_ratio is not None and right_ratio is not None:
            # Average of both eyes
            horizontal_ratio = (left_ratio + right_ratio) / 2.0
            # horizontal_ratio in range [0..1]: 0 = looking far left, 1 = looking far right

    # Figure out which image is "looked at" based on horizontal_ratio
    looked_index = -1
    if horizontal_ratio is not None:
        # 5 segments: 0.0–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.8, 0.8–1.0
        if 0.0 <= horizontal_ratio < 0.2:
            looked_index = 0
        elif 0.2 <= horizontal_ratio < 0.4:
            looked_index = 1
        elif 0.4 <= horizontal_ratio < 0.6:
            looked_index = 2
        elif 0.6 <= horizontal_ratio < 0.8:
            looked_index = 3
        else:
            looked_index = 4

    # Make a copy of the strip for display
    display_strip = image_strip.copy()
    if 0 <= looked_index < 5:
        x_start = looked_index * 200
        x_end = (looked_index + 1) * 200
        cv2.rectangle(display_strip, (x_start, 0), (x_end, 200), (0, 255, 0), 2)

    # Combine the top strip + the webcam feed
    resized_frame = cv2.resize(frame, (strip_width, 300))
    combined_display = np.vstack((display_strip, resized_frame))

    # Calculate time delta
    current_time = time.time()
    delta_time = current_time - last_frame_time
    last_frame_time = current_time

    # Accumulate time
    if looked_index >= 0:
        attention_time[looked_index] += delta_time

    # Show text overlay
    if looked_index >= 0:
        cv2.putText(
            combined_display,
            f"Looking at image #{looked_index+1}",
            (10, strip_height + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            combined_display,
            "Eyes not detected",
            (10, strip_height + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    cv2.imshow("MediaPipe Iris Tracking Demo (20s)", combined_display)

    # Check if 20 seconds have passed or user presses 'q'
    if (current_time - start_time) >= 20:
        print("20 seconds elapsed. Ending session.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User pressed 'q'. Ending session early.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

# Print final results
print("\n=== Eye Tracking Results ===")
for i, t in enumerate(attention_time, start=1):
    print(f"Image #{i}: {t:.2f} seconds")

most_looked_index = max(range(len(attention_time)), key=lambda i: attention_time[i])
print(f"\nMost attention spent on Image #{most_looked_index + 1} "
      f"with ~{attention_time[most_looked_index]:.2f} seconds.")
