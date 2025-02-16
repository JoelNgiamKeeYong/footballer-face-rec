import streamlit as st
import cv2
import time
import threading
from PIL import Image
import os

# Import your face recognition module
from SimpleFacerec import SimpleFacerec

class FaceRecognitionApp:
    def __init__(self):
        self.sfr = SimpleFacerec()
        self.lock = threading.Lock()
        self.face_locations = []
        self.face_names = []
        self.frame_small = None
        self.process_this_frame = True
        self.FRAME_RESIZE_RATIO = 0.2
        self.is_running = False
        self.known_faces = []
        
        # Automatically load encodings and get footballer names
        self.load_encodings("images/")
        
    def load_encodings(self, images_path):
        """Load face encodings from the specified directory and store footballer names"""
        self.sfr.load_encoding_images(images_path)
        # Get list of footballer names from image files
        self.known_faces = [os.path.splitext(f)[0] for f in os.listdir(images_path) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def get_footballer_images(self, images_path):
        """Get dictionary of footballer names and their images"""
        footballer_images = {}
        for filename in os.listdir(images_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(images_path, filename)
                footballer_images[name] = image_path
        return footballer_images
        
    def recognize_faces_thread(self):
        """Background thread for face recognition processing"""
        while self.is_running:
            if self.frame_small is not None and self.process_this_frame:
                small_rgb_frame = cv2.cvtColor(self.frame_small, cv2.COLOR_BGR2RGB)
                
                with self.lock:
                    self.face_locations, self.face_names = self.sfr.detect_known_faces(small_rgb_frame)
                
                self.process_this_frame = False

def main():
    st.set_page_config(
        page_title="Footballer Face Recognition",
        page_icon="⚽",
        layout="centered"
    )
    
    # Initialize session state for camera button
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    # Title and description
    st.title("⚽ Footballer Face Recognition System")
    st.markdown("""
    This application performs real-time face recognition of footballers using your webcam. 
    Point your camera at a picture of a footballer to see if they're recognized!
    """)
    
    # Initialize the face recognition app
    if 'face_app' not in st.session_state:
        with st.spinner("Loading footballer face encodings..."):
            st.session_state.face_app = FaceRecognitionApp()
        st.success("Footballer face encodings loaded successfully!")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Live Recognition", "Available Footballers"])
    
    with tab1:
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:

            
            # Video feed placeholder
            video_placeholder = st.empty()
            
            # FPS counter
            fps_placeholder = st.empty()
            
        with col2:
            # Detection info
            st.subheader("Detection Information")
            detection_info = st.empty()
    
    
    with tab2:
        st.subheader("Registered Footballers")
        # Get footballer images
        footballer_images = st.session_state.face_app.get_footballer_images("images/")

        # Set standard size for all images
        TARGET_SIZE = (300, 300)  # You can adjust these dimensions as needed

        # Display footballers in a grid
        cols = st.columns(3)  # Adjust number of columns as needed
        for idx, (name, image_path) in enumerate(footballer_images.items()):
            with cols[idx % 3]:
                try:
                    # Open image
                    img = Image.open(image_path)
                    
                    # Convert RGBA to RGB if necessary
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    
                    # Get current dimensions
                    width, height = img.size
                    
                    # Calculate dimensions to crop to
                    aspect_ratio = TARGET_SIZE[0] / TARGET_SIZE[1]
                    
                    if width / height > aspect_ratio:
                        # Image is wider than target ratio
                        new_width = int(height * aspect_ratio)
                        offset = (width - new_width) // 2
                        img = img.crop((offset, 0, offset + new_width, height))
                    else:
                        # Image is taller than target ratio
                        new_height = int(width / aspect_ratio)
                        offset = (height - new_height) // 2
                        img = img.crop((0, offset, width, offset + new_height))
                    
                    # Resize to final dimensions
                    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    
                    # Display the standardized image
                    st.image(img, caption=name, use_container_width=True)
             
                except Exception as e:
                    st.error(f"Error loading image for {name}: {str(e)}")
    
    # Start/Stop button in sidebar with dynamic text
    if st.sidebar.button("Stop Camera" if st.session_state.camera_active else "Start Camera"):
        st.session_state.camera_active = not st.session_state.camera_active
        st.session_state.face_app.is_running = st.session_state.camera_active
        
        if st.session_state.face_app.is_running:
            # Initialize video capture
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Start recognition thread
            recognition_thread = threading.Thread(
                target=st.session_state.face_app.recognize_faces_thread,
                daemon=True
            )
            recognition_thread.start()
            
            # FPS calculation
            prev_time = time.time()
            
            try:
                while st.session_state.face_app.is_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Resize frame
                    st.session_state.face_app.frame_small = cv2.resize(
                        frame, (0, 0),
                        fx=st.session_state.face_app.FRAME_RESIZE_RATIO,
                        fy=st.session_state.face_app.FRAME_RESIZE_RATIO
                    )
                    
                    # Process alternate frames
                    st.session_state.face_app.process_this_frame = not st.session_state.face_app.process_this_frame
                    
                    # Draw detections
                    with st.session_state.face_app.lock:
                        for (top, right, bottom, left), name in zip(
                            st.session_state.face_app.face_locations,
                            st.session_state.face_app.face_names
                        ):
                            # Scale back to original size
                            top = int(top / st.session_state.face_app.FRAME_RESIZE_RATIO)
                            right = int(right / st.session_state.face_app.FRAME_RESIZE_RATIO)
                            bottom = int(bottom / st.session_state.face_app.FRAME_RESIZE_RATIO)
                            left = int(left / st.session_state.face_app.FRAME_RESIZE_RATIO)
                            
                            # Draw rectangle and name
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                            cv2.putText(frame, name, (left, top - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Calculate and display FPS
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time
                    
                    # Convert frame to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Update video feed
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Update FPS counter
                    fps_placeholder.markdown(f"**FPS:** {int(fps)}")
                    
                    # Update detection info
                    detection_info.markdown(f"""
                    **Detected Faces:** {len(st.session_state.face_app.face_locations)}
                    
                    **Footballers Detected:**
                    {', '.join(set(st.session_state.face_app.face_names)) if st.session_state.face_app.face_names else 'None'}
                    """)
                    
            finally:
                cap.release()
                st.session_state.face_app.is_running = False
                st.session_state.camera_active = False

if __name__ == "__main__":
    main()