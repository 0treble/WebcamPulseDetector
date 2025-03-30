import matplotlib.pyplot as plt
import numpy as np
import logging
from matplotlib.gridspec import GridSpec


class PulseVisualizer:
    def __init__(self, show_plots=True):
        self.fig = None
        self.cam_ax = None
        self.signal_ax = None
        self.fft_ax = None
        self.show_plots = show_plots
        self._setup_display()

    def _setup_display(self):
        """Initialize the display layout with original styling"""
        plt.close('all')
        self.fig = plt.figure(figsize=(15, 6), facecolor='#f0f0f0')

        # Create grid layout with original spacing
        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Webcam view (left side) - maintain 1:1 aspect ratio
        self.cam_ax = self.fig.add_subplot(gs[:, 0])
        self.cam_ax.set_title('Webcam View', pad=10, fontsize=12)
        self.cam_ax.axis('off')
        self.cam_img = self.cam_ax.imshow(np.zeros((480, 640, 3)),
                                          aspect='auto',
                                          vmin=0,
                                          vmax=255)

        if self.show_plots:
            # Signal plot (top right)
            self.signal_ax = self.fig.add_subplot(gs[0, 1])
            self.signal_ax.set_title('Pulse Signal', pad=10, fontsize=12)
            self.signal_ax.grid(True, linestyle='--', alpha=0.6)
            self.signal_ax.set_facecolor('#f8f8f8')
            self.signal_line, = self.signal_ax.plot([], [], 'b-', linewidth=1.5)
            self.signal_ax.set_xlabel('Time (s)', fontsize=10)
            self.signal_ax.set_ylabel('Intensity', fontsize=10)
            self.signal_ax.tick_params(labelsize=9)

            # FFT plot (bottom right) - with original grid style
            self.fft_ax = self.fig.add_subplot(gs[1, 1])
            self.fft_ax.set_title('Frequency Spectrum (40-180 BPM)', pad=10, fontsize=12)
            self.fft_ax.grid(True, which='both', linestyle='--', alpha=0.6)
            self.fft_ax.set_facecolor('#f8f8f8')
            self.fft_line, = self.fft_ax.plot([], [], 'g-', linewidth=1.5)
            self.fft_ax.set_xlim(40, 180)
            self.fft_ax.set_xlabel('BPM', fontsize=10)
            self.fft_ax.set_ylabel('Amplitude', fontsize=10)
            self.fft_ax.tick_params(labelsize=9)

            # Add minor grid lines for better readability
            self.fft_ax.grid(True, which='minor', linestyle=':', alpha=0.4)
            self.fft_ax.minorticks_on()

        plt.ion()
        plt.tight_layout()
        plt.show()

    def update_plots(self, processor):
        """Update all visualizations with proper styling"""
        try:
            # Update webcam view (maintain 1:1 aspect ratio)
            self.cam_img.set_array(processor.frame_out)

            if self.show_plots and hasattr(processor, 'regions'):
                # Update signal plot
                if (processor.regions['forehead']['values'] and
                        hasattr(processor, 'times') and
                        len(processor.times) >= len(processor.regions['forehead']['values'])):
                    times = np.array(processor.times)
                    signal = np.array(processor.regions['forehead']['values'])
                    signal = signal - np.mean(signal)  # Remove DC

                    self.signal_line.set_data(times[-len(signal):] - times[0], signal)
                    self.signal_ax.relim()
                    self.signal_ax.autoscale_view()
                    self.signal_ax.grid(True, linestyle='--', alpha=0.6)

                # Update FFT plot with original grid style
                if (hasattr(processor, 'freqs') and
                        hasattr(processor, 'fft') and
                        len(processor.freqs) > 0 and
                        len(processor.fft) > 0):

                    bpm = processor.freqs * 60
                    mask = (bpm >= 40) & (bpm <= 180)

                    if np.any(mask):
                        self.fft_line.set_data(bpm[mask], processor.fft[mask])
                        self.fft_ax.relim()
                        self.fft_ax.autoscale_view()
                        self.fft_ax.grid(True, which='both', linestyle='--', alpha=0.6)

                        if (processor.bpm > 40 and
                                hasattr(processor, 'snr') and
                                len(bpm[mask]) > 0 and
                                len(processor.fft[mask]) > 0):

                            peak_idx = np.argmin(np.abs(bpm[mask] - processor.bpm))
                            if peak_idx < len(bpm[mask]) and peak_idx < len(processor.fft[mask]):
                                # Remove previous peak marker if it exists
                                if hasattr(self, 'peak_marker'):
                                    self.peak_marker.remove()
                                self.peak_marker, = self.fft_ax.plot(
                                    bpm[mask][peak_idx],
                                    processor.fft[mask][peak_idx],
                                    'ro', markersize=8)

                                # Remove previous text if it exists
                                if hasattr(self, 'bpm_text'):
                                    self.bpm_text.remove()
                                self.bpm_text = self.fft_ax.text(
                                    processor.bpm + 5,
                                    processor.fft[mask][peak_idx],
                                    f'{processor.bpm:.1f} BPM\nSNR: {processor.snr:.1f}',
                                    color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

            plt.pause(0.001)

        except Exception as e:
            logging.error(f"Visualization error: {e}")
            self._setup_display()

    def close(self):
        """Clean up resources"""
        plt.close('all')