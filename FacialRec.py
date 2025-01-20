import cv2
import streamlit as st
import numpy as np
import os

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_faces(scale_factor, min_neighbors, rectangle_color):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Unable to access webcam. Please check your device.")
        return

    # Initialize session state for the stop button and image saving
    if "stop_detection" not in st.session_state:
        st.session_state.stop_detection = False  # Create a flag for stopping detection

    if "save_image" not in st.session_state:
        st.session_state.save_image = False  # Create a flag for saving images

    # Display the webcam feed
    image_placeholder = st.empty()

    # Save Image checkbox to persist state
    st.session_state.save_image = st.checkbox(
        "Save images with detected faces to your device",
        value=st.session_state.save_image,  # Maintain previous value using session state
        key="save_faces_checkbox"
    )

    # Stop Detection button (rendered only once)
    if st.button("Stop Detection"):
        st.session_state.stop_detection = True

    while not st.session_state.stop_detection:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret or frame is None:
            st.error("Error: Unable to detect frames. Ensure your webcam is functional.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            b, g, r = rectangle_color
            cv2.rectangle(frame, (x, y), (x + w, y + h), (int(b), int(g), int(r)), 2)

        # Display the frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_placeholder.image(frame_rgb, channels="RGB")

        # Save the frame with detected faces if checkbox is checked
        if st.session_state.save_image and len(faces) > 0:
            cv2.imwrite("detected_faces.jpg", frame)
            st.success("Image saved as 'detected_faces.jpg' in the current directory.", icon="✅")
            st.session_state.save_image = False  # Avoid saving on every loop iteration

    # Release the webcam when finished
    cap.release()
    st.info("Detection stopped. Thank you for using the application.")


def app():
    st.title("Face Detection using Viola-Jones Algorithm")

    st.markdown("""
        ### Instructions:
        - Use "Detect Faces" to start the application.
        - Adjust the detection sensitivity using the sliders below.
        - Choose a color for the rectangles around detected faces.
        - Save detected face images if you wish via the checkbox.
        - Click "Stop Detection" to exit the webcam feed.
    """)

    # Rectangle color selection
    rectangle_color_hex = st.color_picker("Select Rectangle Color", "#00FF00", key="color_picker_rectangle")
    rectangle_color_bgr = tuple(int(rectangle_color_hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    # Sliders for scale factor and min neighbors
    scale_factor = st.slider(
        "Scale Factor (Detection Accuracy)",
        min_value=1.1,
        max_value=2.0,
        step=0.1,
        value=1.3,
        key="scale_factor_slider"
    )

    min_neighbors = st.slider(
        "Min Neighbors (Detection Sensitivity)",
        min_value=3,
        max_value=10,
        step=1,
        value=5,
        key="min_neighbors_slider"
    )

    # Detect Faces button
    if st.button("Detect Faces", key="detect_faces"):
        st.session_state.stop_detection = False  # Reset stop button state
        detect_faces(scale_factor, min_neighbors, rectangle_color_bgr)


if __name__ == "__main__":
    app()
