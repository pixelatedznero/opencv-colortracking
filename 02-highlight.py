import cv2
import numpy as np

# Define the range of colors to be tracked and other variables
lower_color_range = np.array([0,100,100])
upper_color_range = np.array([30,255,255])

allareas = []

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
        allareas.append(area)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 5, (0, 0, 255), -1)

    # add a special color and coordinate display for the biggest
    if len(contours) > 0:
        contour = contours[allareas.index(max(allareas))]
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"X: {int(x + w / 2)}, Y: {int(y + h / 2)}", 
                    (x + 10, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    allareas = []

    # Show the output with coordinates
    cv2.imshow('Tracking', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()