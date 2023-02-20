import cv2
import numpy as np

# Define the range of colors to be tracked
lower_color_range = np.array([0,100,100])
upper_color_range = np.array([30,255,255])

# Initialize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

cv2.namedWindow('Tracking')

while True:
    # Read the frames from the webcam
    ret, frame = cap.read()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only the pixels within the color range
    mask = cv2.inRange(hsv, lower_color_range, upper_color_range)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a rectangle and a circle around the detected contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 5, (0, 0, 255), -1)

    # Show the output with coordinates
    cv2.imshow('Tracking', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()