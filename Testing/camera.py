import cv2
import time

def capture_and_display(filename='captured_image.jpg'):
    # Initialize the camera
    camera = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)

    # Check if the camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    # Give a few seconds for the camera to initialize
    time.sleep(1)

    # Capture a single frame
    ret, frame = camera.read()

    if not ret:
        print("Error: Failed to capture image.")
        camera.release()
        return

    # Release the camera
    camera.release()

    # Save the captured frame to file
    cv2.imwrite(filename, frame)

    # Display the captured image
    cv2.imshow('Captured Image', frame)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed


    cv2.destroyAllWindows()
    print(f"Image captured and saved as {filename}")

# Example usage
if __name__ == "__main__":
    capture_and_display('captured_image.jpg')
