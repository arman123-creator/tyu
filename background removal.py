import cv2
import numpy as np
import cv2
import numpy as np

def add_mountain_background(frame, mountain_img):
    # Convert the frame to HSV (Hue, Saturation, Value) color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green in HSV
    lower_green = np.array([20, 20, 20])
    upper_green = np.array([90, 255, 255])

    # Create a mask for the green screen
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Invert the mask to get the area that is not green
    mask_inv = cv2.bitwise_not(mask)

    # Ensure the mountain image has the same size as the frame
    mountain_img_resized = cv2.resize(mountain_img, (frame.shape[1], frame.shape[0]))

    # Create a mask with the mountain image where the green screen is present
    mountain_mask = cv2.bitwise_and(mountain_img_resized, mountain_img_resized, mask=mask)

    # Create a mask with the original frame where the green screen is not present
    frame_mask = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine the mountain mask and the frame mask to get the final frame
    result_frame = cv2.add(mountain_mask, frame_mask)

    return result_frame

# The rest of the code remains the same as in the previous examples






def main():
    # Load the mountain background image
    mountain_img = cv2.imread('mountain_back.jpg')

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Add the mountain background to the frame
        frame = add_mountain_background(frame, mountain_img)

        # Display the resulting frame
        cv2.imshow('Anywhere Selfie Booth', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
