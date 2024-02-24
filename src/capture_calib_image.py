from pyk4a import PyK4A, Config, ColorResolution, FPS
import cv2

def capture_image(k4a, window_name):
    """
    Capture an image from the given Kinect device and display it in a window.
    """
    capture = k4a.get_capture()
    if capture.color is not None:
        frame = capture.color[:, :, :3]
        cv2.imshow(window_name, frame)
        return frame
    return None

def main():
    """
    Main function to initialize two cameras, capture, and save images.
    """
    config = Config(
        color_resolution=ColorResolution.RES_720P,
        camera_fps=FPS.FPS_30,
    )

    # Initialize two Kinect devices
    k4a1 = PyK4A(config=config, device_id=0)
    k4a2 = PyK4A(config=config, device_id=1)

    # Start both devices
    k4a1.start()
    k4a2.start()

    try:
        while True:
            frame1 = capture_image(k4a1, 'Camera 1')
            frame2 = capture_image(k4a2, 'Camera 2')

            # Save images from both cameras
            if cv2.waitKey(1) & 0xFF == ord('s'):
                if frame1 is not None:
                    cv2.imwrite('color_test_1.png', frame1)
                    print("Saved image from Camera 1.")
                if frame2 is not None:
                    cv2.imwrite('color_test_2.png', frame2)
                    print("Saved image from Camera 2.")
                break

            # Exit loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

    finally:
        # Stop both devices
        k4a1.stop()
        k4a2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
