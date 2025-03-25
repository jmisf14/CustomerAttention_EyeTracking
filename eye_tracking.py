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
    refine_landmarks=True,     # For iris tracking, but we'll do a simpler approach
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start capturing from default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# We'll store how much time each image gets
attention_time = [0, 0, 0, 0, 0]

print("Starting 20-second eye-tracking session using MediaPipe...")
print("Press 'q' to quit early.")

start_time = time.time()
last_frame_time = start_time

# Eye-region landmark indices (approximate) from the 468-point Face Mesh:
# Left eye (including upper/lower eyelid) typically includes indices around [33, 133, 160-173...].
# Right eye typically includes indices around [263, 362, 384-397...].
# Below are smaller subsets representing the central area of each eye:
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 153, 145, 144]
RIGHT_EYE_INDICES = [263, 362, 387, 386, 385, 380, 374, 373]

def get_eye_center_x(landmarks, indices):
    """Compute the average x-coordinate of selected eye landmarks."""
    x_sum = 0
    count = 0
    for i in indices:
        x_sum += landmarks[i].x
        count += 1
    return x_sum / count if count > 0 else None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face mesh
    results = face_mesh.process(rgb_frame)

    # Default "no gaze" ratio
    horizontal_ratio = None

    if results.multi_face_landmarks:
        # Take the first face detected (we only track one face)
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # Get bounding box for the entire face to normalize
        xs = [lm.x for lm in landmarks]
        min_x, max_x = min(xs), max(xs)

        # Average left eye center
        left_eye_center_x = get_eye_center_x(landmarks, LEFT_EYE_INDICES)
        # Average right eye center
        right_eye_center_x = get_eye_center_x(landmarks, RIGHT_EYE_INDICES)

        if left_eye_center_x is not None and right_eye_center_x is not None:
            # We'll take the average of both eyes' center x
            eyes_avg_x = (left_eye_center_x + right_eye_center_x) / 2.0

            # Normalize: (eyes_avg_x - min_x) / (max_x - min_x)
            if max_x - min_x > 0:
                horizontal_ratio = (eyes_avg_x - min_x) / (max_x - min_x)
                # 0 => far left, 1 => far right

    # Figure out which image is "looked at" based on horizontal_ratio
    looked_index = -1
    if horizontal_ratio is not None:
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

    cv2.imshow("MediaPipe Eye Tracking Demo (20s)", combined_display)

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

