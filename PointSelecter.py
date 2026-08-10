#for loacl Machine
import cv2
import numpy as np

# Array to store the 4 clicked points
points = []

def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Points", frame)

        if len(points) == 4:
            print("\nCopy this array into your Colab notebook:")
            print(f"source_points = np.array({points}, dtype=np.float32)")

# Load your video locally
cap = cv2.VideoCapture("your_traffic_video.mp4")
ret, frame = cap.read()
cap.release()

cv2.imshow("Select 4 Points", frame)
cv2.setMouseCallback("Select 4 Points", select_points)

# Press any key to close the window after clicking 4 times
cv2.waitKey(0)
cv2.destroyAllWindows()