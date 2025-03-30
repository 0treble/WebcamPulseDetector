import time
from pulse_detector import Webcam, PulseDetector, PulseVisualizer
import matplotlib.pyplot as plt


def main():
    try:
        # Initialize components
        camera = Webcam()
        processor = PulseDetector()
        visualizer = PulseVisualizer(show_plots=True)

        # Main loop
        while True:
            start_time = time.time()

            # Process frame
            frame = camera.get_frame()
            processed_frame = processor.process_frame(frame)

            # Update display
            visualizer.update_plots(processor)

            # Maintain ~30 FPS
            elapsed = time.time() - start_time
            time.sleep(max(0, 1 / 60 - elapsed))

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.release()
        plt.close('all')


if __name__ == "__main__":
    main()