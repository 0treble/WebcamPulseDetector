from pulse_detector import Webcam, PulseDetector, PulseVisualizer
from ttkthemes import ThemedTk


def main():
    try:
        # Initialize Tkinter root
        root = ThemedTk(theme="equilux")

        # Initialize components
        camera = Webcam()
        processor = PulseDetector()
        visualizer = PulseVisualizer(root, show_plots=True)

        def update():
            """Main update loop"""
            # Get current parameters from visualizer and update processor
            processor.update_settings(visualizer.params)

            # Process frame with current parameters
            frame = camera.get_frame()
            processed_frame = processor.process_frame(frame)

            # Update display
            visualizer.update_plots(processor)

            # Schedule next update (~30 FPS)
            root.after(33, update)

        # Start the update loop
        update()
        root.mainloop()

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.release()
        if 'visualizer' in locals():
            visualizer.close()


if __name__ == "__main__":
    main()
