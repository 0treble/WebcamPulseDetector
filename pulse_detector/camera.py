import numpy as np
import cv2
import logging


class Webcam:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        self.available = self.camera.isOpened()

        if not self.available:
            logging.error("No cameras found. Ensure your device has a connected and supported camera.")
            self.shape = (480, 640, 3)
        else:
            # Get frame width and height
            self.shape = (
                int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                3
            )
            logging.info(f"Camera initialized with shape: {self.shape}")

    def get_frame(self):
        """Capture a frame from the webcam"""
        if not self.available:
            logging.warning("Camera not available. Returning blank frame.")
            return np.ones(self.shape, dtype=np.uint8) * 255

        ret, frame = self.camera.read()
        if not ret:
            logging.error("Failed to capture frame. Returning blank frame.")
            return np.ones(self.shape, dtype=np.uint8) * 255

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    def release(self):
        """Release the camera resources"""
        if self.available:
            self.camera.release()
            logging.info("Camera released successfully.")
