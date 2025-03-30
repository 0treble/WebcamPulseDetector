"""
pulse_detector package

This package provides webcam-based pulse detection functionality including:
- Camera capture
- Face and forehead detection
- Pulse rate calculation
- Visualization
"""

from .camera import Webcam
from .processor import PulseDetector
from .visualizer import PulseVisualizer

__all__ = ['Webcam', 'PulseDetector', 'PulseVisualizer']
__version__ = '1.0.0'

# Initialize logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Pulse detector package initialized")