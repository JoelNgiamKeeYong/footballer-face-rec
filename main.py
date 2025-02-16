import cv2
import threading
import time
from SimpleFacerec import SimpleFacerec

# Initialize face recognition
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera with buffer optimization
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce frame buffer delay

# Reduce frame size for speed
FRAME_RESIZE_RATIO = 0.4  # Try 0.3 or lower for even more speed

# Threaded face recognition variables
face_locations = []
face_names = []
frame_small = None
lock = threading.Lock()
process_this_frame = True  # Toggle to skip frames for efficiency


def recognize_faces():
    """Thread function for face recognition processing."""
    global face_locations, face_names, frame_small, process_this_frame

    while True:
        if frame_small is not None and process_this_frame:
            small_rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            # Perform face detection & recognition
            with lock:
                face_locations, face_names = sfr.detect_known_faces(small_rgb_frame)

            process_this_frame = False  # Skip next frame to reduce lag


# Start background thread
recognition_thread = threading.Thread(target=recognize_faces, daemon=True)
recognition_thread.start()

# FPS Calculation
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame_small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_RATIO, fy=FRAME_RESIZE_RATIO)

    # Process every alternate frame
    process_this_frame = not process_this_frame

    # Draw rectangles & labels
    with lock:
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back to original size
            top, right, bottom, left = (
                int(top / FRAME_RESIZE_RATIO),
                int(right / FRAME_RESIZE_RATIO),
                int(bottom / FRAME_RESIZE_RATIO),
                int(left / FRAME_RESIZE_RATIO),
            )

            # Draw face bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # FPS Display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Exit if the X button is pressed or Esc key is hit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Escape key
        break
    if cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:  # Check if the window was closed
        break

cap.release()
cv2.destroyAllWindows()
