import time
import numpy as np
import logging
from PIL import Image, ImageDraw
import cv2
from scipy import signal as scipy_signal
import mediapipe as mp
import math


class PulseDetector:
    def __init__(self):
        self.frame_in = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frame_out = np.zeros((480, 640, 3), dtype=np.uint8)
        self.buffer_size = 250
        self.times = []
        self.bpm = 0
        self.fps = 60

        # Initialize MediaPipe Face Detection and Face Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.7,
            model_selection=1  # Range 0-180 degrees
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_rect = None
        self.face_rotation = 0  # Rotation angle in degrees
        self.face_scale = 1.0  # Scale factor
        self.tracking_faces = True
        self.detection_interval = 1
        self.frame_count = 0

        # Initialize regions
        self.regions = {
            'forehead': {
                'rect': None,
                'values': [],
                'color': (0, 255, 0),
                'offset_y': -0.13,
                'offset_x': 0,
                'scale_w': 0.35,
                'scale_h': 0.25,
                'enabled': True
            },
            'left_cheek': {
                'rect': None,
                'values': [],
                'color': (255, 165, 0),
                'offset_y': 0,
                'offset_x': -0.25,
                'scale_w': 0.2,
                'scale_h': 0.2,
                'enabled': True
            },
            'right_cheek': {
                'rect': None,
                'values': [],
                'color': (255, 165, 0),
                'offset_y': 0,
                'offset_x': 0.25,
                'scale_w': 0.2,
                'scale_h': 0.2,
                'enabled': True
            }
        }

        self.region_weights = {'forehead': 0.80, 'left_cheek': 0.10, 'right_cheek': 0.10}
        self.bpm_history = []
        self.valid_pulse_range = (40, 180)
        self.smoothing_factor = 0.7
        self.min_region_size = (30, 20)
        self.snr = 0
        self.bandpass_low = 45
        self.bandpass_high = 240

    def find_faces(self, frame):
        """Detect faces and estimate rotation using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        mesh_results = self.face_mesh.process(rgb_frame)

        faces = []

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw = frame.shape[:2]
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Ensure coordinates are within frame bounds
                x, y = max(0, x), max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)

                faces.append([x, y, w, h])

        # Estimate face rotation and scale from face mesh
        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0]

            # Get key landmarks for rotation estimation
            nose_tip = face_landmarks.landmark[4]  # Nose tip
            chin = face_landmarks.landmark[152]  # Chin
            left_eye = face_landmarks.landmark[33]  # Left eye outer corner
            right_eye = face_landmarks.landmark[263]  # Right eye outer corner

            # Calculate rotation angle
            dx = right_eye.x - left_eye.x
            dy = right_eye.y - left_eye.y
            rotation = math.degrees(math.atan2(dy, dx))

            # Calculate scale based on face width
            face_height = math.sqrt((chin.x - nose_tip.x) ** 2 + (chin.y - nose_tip.y) ** 2)
            face_width = math.sqrt(dx ** 2 + dy ** 2)
            scale = face_height

            self.face_rotation = rotation
            self.face_scale = scale

        return faces

    def get_region_mean(self, frame, rect):
        """Get average pixel value in specified region"""
        x, y, w, h = rect
        h_frame, w_frame = frame.shape[:2]

        # Ensure ROI is within bounds
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)

        region = frame[y:y + h, x:x + w]
        return np.mean(region) if region.size > 0 else 0

    def _update_face_regions(self, face_rect):
        """Calculate positions for all tracking regions with rotation"""
        x, y, w, h = face_rect

        # Apply scale factor to the face dimensions
        scaled_w = w * 1.1 #self.face_scale
        scaled_h = h * 1.1 #self.face_scale

        # Face center point
        center_x = x + w / 2
        center_y = y + h / 2

        for region_name, region in self.regions.items():
            if not region['enabled']:
                continue

            # Calculate region dimensions
            region_w = max(self.min_region_size[0], int(scaled_w * region['scale_w']))
            region_h = max(self.min_region_size[1], int(scaled_h * region['scale_h']))

            # Calculate position relative to face center
            offset_x = scaled_w * region['offset_x']
            offset_y = scaled_h * region['offset_y']

            # Rotate the offset vector
            angle_rad = math.radians(self.face_rotation)
            rotated_offset_x = offset_x * math.cos(angle_rad) - offset_y * math.sin(angle_rad)
            rotated_offset_y = offset_x * math.sin(angle_rad) + offset_y * math.cos(angle_rad)

            # Calculate final position
            region_x = int(center_x + rotated_offset_x - region_w / 2)
            region_y = int(center_y + rotated_offset_y - region_h / 2)

            region['rect'] = [region_x, region_y, region_w, region_h]
            region['rotation'] = self.face_rotation

    def draw_rotated_rectangle(self, draw, rect, color, width=2):
        """Draw a rectangle rotated around its center"""
        x, y, w, h, rotation = rect
        center_x = x + w / 2
        center_y = y + h / 2

        # Create rectangle polygon
        points = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ]

        # Rotate points around center
        angle_rad = math.radians(rotation)
        rotated_points = []
        for px, py in points:
            # Translate to origin
            px -= center_x
            py -= center_y

            # Rotate
            new_px = px * math.cos(angle_rad) - py * math.sin(angle_rad)
            new_py = px * math.sin(angle_rad) + py * math.cos(angle_rad)

            # Translate back
            new_px += center_x
            new_py += center_y

            rotated_points.append((new_px, new_py))

        # Draw rotated rectangle
        draw.polygon(rotated_points, outline=color, width=width)

    def process_frame(self, frame):
        self.frame_in = frame
        self.frame_out = frame.copy()
        img = Image.fromarray(self.frame_out)
        draw = ImageDraw.Draw(img)
        self.frame_count += 1

        # Face detection and region updates
        if self.tracking_faces and (self.frame_count % self.detection_interval == 0 or self.face_rect is None):
            faces = self.find_faces(frame)
            if len(faces) > 0:
                self.face_rect = max(faces, key=lambda f: f[2] * f[3])
                self._update_face_regions(self.face_rect)

        # Process all regions if face detected
        if self.face_rect is not None:
            current_time = time.time()
            for region_name, region in self.regions.items():
                if region['enabled'] and region['rect']:
                    # Get and store region value
                    value = self.get_region_mean(frame, region['rect'])
                    region['values'].append(value)

                    # Maintain buffer size
                    if len(region['values']) > self.buffer_size:
                        region['values'] = region['values'][-self.buffer_size:]

                    # Draw rotated region rectangle
                    rect = region['rect'] + [region.get('rotation', 0)]
                    self.draw_rotated_rectangle(draw, rect, region['color'])

            # Store time for this frame
            self.times.append(current_time)
            if len(self.times) > self.buffer_size:
                self.times = self.times[-self.buffer_size:]

            # Calculate combined pulse signal
            enabled_regions = [r for r in self.regions.values() if r['enabled'] and r['rect']]
            if len(enabled_regions) > 0 and all(len(r['values']) > 10 for r in enabled_regions):
                self._calculate_combined_pulse()

            # Draw BPM text
            if hasattr(self, 'bpm'):
                forehead = self.regions['forehead']
                if forehead['rect']:
                    x, y, w, h = forehead['rect']
                    status = f"BPM: {self.bpm:.1f}" if self.bpm > self.valid_pulse_range[0] else "Finding pulse..."
                    draw.text((x, y - 30), status, fill=forehead['color'])

        self.frame_out = np.array(img)
        return self.frame_out

    def _calculate_combined_pulse(self):
        """Calculate pulse from multiple regions with weighted average"""
        try:
            # Check if we have enough data
            if len(self.times) < 10 or self.times[-1] <= self.times[0]:
                return

            # Combine signals using weighted average
            min_length = min(len(r['values']) for r in self.regions.values() if r['enabled'] and r['rect'])
            if min_length < 10:  # Need at least 10 samples
                return

            combined_signal = np.zeros(min_length)
            total_weight = 0

            for region_name, region in self.regions.items():
                if region['enabled'] and region['rect'] and len(region['values']) >= min_length:
                    signal_data = np.array(region['values'][-min_length:])
                    signal_data = signal_data - np.mean(signal_data)  # Remove DC
                    combined_signal += signal_data * self.region_weights.get(region_name, 0)
                    total_weight += self.region_weights.get(region_name, 0)

            if total_weight == 0 or len(combined_signal) < 10:
                return

            combined_signal /= total_weight
            times = np.array(self.times[-min_length:])
            time_span = times[-1] - times[0]

            # Calculate FPS and validate
            if time_span <= 0:
                return

            fps = len(combined_signal) / time_span
            if fps <= 0:
                return

            # Bandpass filter - validate frequencies
            nyquist = 0.5 * fps
            low = self.bandpass_low / (nyquist * 60)
            high = self.bandpass_high / (nyquist * 60)

            # Validate filter parameters
            if low <= 0 or high >= 1 or low >= high:
                return

            try:
                b, a = scipy_signal.butter(3, [low, high], btype='band')
                filtered = scipy_signal.filtfilt(b, a, combined_signal)
            except ValueError as e:
                logging.error(f"Filter error: {e}")
                return

            # FFT calculation
            n = len(filtered)
            if n < 10:  # Need minimum samples for FFT
                return

            window = np.hamming(n)
            fft = np.abs(np.fft.rfft(filtered * window))
            freqs = np.fft.rfftfreq(n, d=1. / fps)

            # Find peak in valid range
            bpm = freqs * 60
            mask = (bpm >= self.valid_pulse_range[0]) & (bpm <= self.valid_pulse_range[1])

            if not np.any(mask):
                return

            peak_idx = np.argmax(fft[mask])
            raw_bpm = bpm[mask][peak_idx]
            self.snr = (np.max(fft[mask]) / np.mean(fft[mask])) if np.mean(fft[mask]) > 0 else 0

            # Store FFT data for visualization
            self.fft = fft
            self.freqs = freqs

            # Temporal smoothing
            if self.snr >= 2.0:
                self.bpm_history.append(raw_bpm)
                if len(self.bpm_history) > 5:
                    self.bpm_history.pop(0)
                self.bpm = np.mean(self.bpm_history)
            else:
                self.bpm = 0

        except Exception as e:
            logging.error(f"Pulse calculation error: {e}")

    def update_settings(self, params):
        """Update processor settings from visualizer"""
        self.buffer_size = params['buffer_size']
        self.valid_pulse_range = (params['min_bpm'], params['max_bpm'])
        self.smoothing_factor = params['smoothing']
        self.bandpass_low = params['bandpass_low']
        self.bandpass_high = params['bandpass_high']
        self.detection_interval = params['detection_interval']

        # Update regions
        for region_name in self.regions:
            if region_name in params['regions']:
                self.regions[region_name]['enabled'] = params['regions'][region_name]['enabled']
                self.regions[region_name]['offset_y'] = params['regions'][region_name]['offset_y']
                self.regions[region_name]['offset_x'] = params['regions'][region_name]['offset_x']
                self.regions[region_name]['scale_w'] = params['regions'][region_name]['scale_w']
                self.regions[region_name]['scale_h'] = params['regions'][region_name]['scale_h']

        # Update weights
        total_weight = sum(params['regions'][r]['weight'] for r in params['regions'] if params['regions'][r]['enabled'])
        if total_weight > 0:
            self.region_weights = {
                r: params['regions'][r]['weight'] / total_weight
                for r in params['regions']
                if params['regions'][r]['enabled']
            }
        else:
            self.region_weights = {'forehead': 0.80, 'left_cheek': 0.10, 'right_cheek': 0.10}
