import cv2
import numpy as np

# Define the range of colors to be tracked
lower_color_range = np.array([0,100,100])
upper_color_range = np.array([30,255,255])

# Variables for specification
color_range_size = 5            # size of generated range for color
saturation_range_size = 50      # size of generated range for color
brightness_range_size = 50      # size of generated range for color
detection_size = 7              # size of region the color is read

# Other variables
mouse_coords = (0,0)
allareas = []

# Get the color from a certain position in BGR
def get_color(x, y, frame):
    floatcolor = cv2.mean(frame[y-int(detection_size/2):y+int(detection_size/2), 
                                x-int(detection_size/2):x+int(detection_size/2)])
    color = []

    for i in floatcolor:
        color.append(int(i))
    
    return color[0:3]

# Initialize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

# Funktion to set new color range
def mouse_action(event, x, y, flags, param):
    global lower_color_range, upper_color_range, mouse_coords

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_coords = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        color = cv2.cvtColor(np.array([[get_color(x, y, read_frame)]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
        
        lower_color_range = np.array([0 if color[0]-color_range_size < 0 else color[0]-color_range_size, 
                                    0 if color[1]-saturation_range_size < 0 else color[1]-saturation_range_size, 
                                    0 if color[2]-brightness_range_size < 0 else color[2]-brightness_range_size])
        upper_color_range = np.array([255 if color[0]+color_range_size > 255 else color[0]+color_range_size, 
                                    255 if color[1]+saturation_range_size > 255 else color[1]+saturation_range_size, 
                                    255 if color[2]+brightness_range_size > 255 else color[2]+brightness_range_size])
        

cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', mouse_action)

while True:
    # Read the frames from the webcam
    ret, frame = cap.read()
    read_frame = frame

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only the pixels within the color range
    mask = cv2.inRange(hsv, lower_color_range, upper_color_range)

    # Apply morphological closing to fill small gaps and holes in the detected objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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

    # color from mouspostion and detections square
    x_mouse = mouse_coords[0]
    y_mouse = mouse_coords[1]
    cv2.circle(frame, (int(50 + 10 / 2), int(50 + 10 / 2)), 30, get_color(x_mouse, y_mouse, read_frame), -1)
    cv2.rectangle(frame, (x_mouse-1 - int(detection_size/2), y_mouse-1 - int(detection_size/2)), 
                  (x_mouse+1 + int(detection_size/2), y_mouse+1 + int(detection_size/2)), 
                  (255, 0, 0), 1)
    
    # Show the output with coordinates
    cv2.imshow('Tracking', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()