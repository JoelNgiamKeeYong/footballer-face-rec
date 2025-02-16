import face_recognition
import cv2
import os
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
        """Loads face encodings from images in the given folder."""
        for filename in os.listdir(images_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(images_path, filename)
                img = face_recognition.load_image_file(img_path)

                # Encode face
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(os.path.splitext(filename)[0])  # Use filename as name
                else:
                    print(f"Warning: No face detected in {filename}")

    def detect_known_faces(self, frame):
        """Detects faces in a frame and matches them with known faces."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
            name = "Unknown"

            # Use the first match found
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            face_names.append(name)

        return face_locations, face_names
