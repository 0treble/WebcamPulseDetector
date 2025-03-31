import time
import numpy as np
import logging
from PIL import Image, ImageDraw
import cv2
import os
import sys
from scipy import signal as scipy_signal


def resource_path(relative_path):
    """ Get absolute path to resource """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class PulseDetector:
    def __init__(self):
        self.frame_in = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frame_out = np.zeros((480, 640, 3), dtype=np.uint8)
        self.buffer_size = 250
        self.times = []
        self.bpm = 0
        self.fps = 60

        # Face detection
        self.face_cascade = cv2.CascadeClassifier(resource_path("haarcascade_frontalface_alt.xml"))
        if self.face_cascade.empty():
            logging.error("Failed to load Haar Cascade classifier.")
            raise Exception("Haar Cascade classifier not found.")

        self.face_rect = None
        self.tracking_faces = True
        self.detection_interval = 5
        self.frame_count = 0

        # Initialize with default regions (will be updated from visualizer)
        self.regions = {
            'forehead': {
                'rect': None,
                'values': [],
                'color': (0, 255, 0),
                'offset_y': 0.18,
                'offset_x': 0,
                'scale_w': 0.35,
                'scale_h': 0.25,
                'enabled': True
            },
            'left_cheek': {
                'rect': None,
                'values': [],
                'color': (255, 165, 0),
                'offset_y': 0.50,
                'offset_x': -0.2,
                'scale_w': 0.2,
                'scale_h': 0.15,
                'enabled': True
            },
            'right_cheek': {
                'rect': None,
                'values': [],
                'color': (255, 165, 0),
                'offset_y': 0.50,
                'offset_x': 0.2,
                'scale_w': 0.2,
                'scale_h': 0.15,
                'enabled': True
            }
        }

        # Default weights (will be updated from visualizer)
        self.region_weights = {'forehead': 0.80, 'left_cheek': 0.10, 'right_cheek': 0.10}

        # Signal processing
        self.bpm_history = []
        self.valid_pulse_range = (40, 180)  # Will be updated from visualizer
        self.smoothing_factor = 0.7  # Will be updated from visualizer
        self.min_region_size = (30, 20)
        self.snr = 0
        self.bandpass_low = 45  # Will be updated from visualizer
        self.bandpass_high = 240  # Will be updated from visualizer

    def update_settings(self, params):
        """Update processor settings from visualizer"""
        self.buffer_size = params['buffer_size']
        self.valid_pulse_range = (params['min_bpm'], params['max_bpm'])
        self.smoothing_factor = params['smoothing']
        self.bandpass_low = params['bandpass_low']
        self.bandpass_high = params['bandpass_high']

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
            # Fallback to default weights if all weights are zero
            self.region_weights = {'forehead': 0.80, 'left_cheek': 0.10, 'right_cheek': 0.10}

    def find_faces(self, frame):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def _update_face_regions(self, face_rect):
        """Calculate positions for all tracking regions"""
        x, y, w, h = face_rect

        for region_name, region in self.regions.items():
            if not region['enabled']:
                continue

            # Calculate region dimensions
            region_w = max(self.min_region_size[0], int(w * region['scale_w']))
            region_h = max(self.min_region_size[1], int(h * region['scale_h']))

            # Calculate position
            center_x = x + w / 2
            if 'offset_x' in region:
                center_x += w * region['offset_x']

            region['rect'] = [
                int(center_x - region_w / 2),
                int(y + h * region['offset_y'] - region_h / 2),
                region_w,
                region_h
            ]

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

                    # Draw region rectangle
                    x, y, w, h = region['rect']
                    draw.rectangle([x, y, x + w, y + h], outline=region['color'], width=2)

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
            combined_signal = np.zeros_like(self.regions['forehead']['values'])
            total_weight = 0

            for region_name, region in self.regions.items():
                if region['enabled'] and region['rect'] and len(region['values']) > 0:
                    signal_data = np.array(region['values'])
                    signal_data = signal_data - np.mean(signal_data)  # Remove DC
                    combined_signal += signal_data * self.region_weights.get(region_name, 0)
                    total_weight += self.region_weights.get(region_name, 0)

            if total_weight == 0:
                return

            combined_signal /= total_weight
            times = np.array(self.times[-len(combined_signal):])

            # Bandpass filter
            fps = len(combined_signal) / (times[-1] - times[0])
            nyquist = 0.5 * fps
            low = self.bandpass_low / (nyquist * 60)
            high = self.bandpass_high / (nyquist * 60)

            b, a = scipy_signal.butter(3, [low, high], btype='band')
            filtered = scipy_signal.filtfilt(b, a, combined_signal)

            # FFT calculation
            n = len(filtered)
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
            self.snr = (np.max(fft[mask]) / np.mean(fft[mask]))

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
