import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def upload_video():
    file_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("Video files", "*.mp4;*.avi")])
    if file_path:
        play_video(file_path)

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")

    # Create a window with a fixed size
    cv2.namedWindow('Snooker Ball', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Snooker Ball', 1000, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_balls(frame)  # Detect snooker balls using Hough Circle Transform

        cv2.imshow('Snooker Ball', frame)

        key = cv2.waitKey(25)

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_balls(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    circles = cv2.HoughCircles(
        gray_frame, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=35
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, radius = circle

            # Crop a region of interest (ROI) around the detected ball
            roi = hsv_frame[y - radius:y + radius, x - radius:x + radius]

            # Calculate the brightness (value) of the ROI
            brightness = np.mean(roi[:, :, 2])

            # Desaturate the ball pixels by setting the saturation channel to a constant value
            saturation = 30  # Adjust the saturation value as needed
            roi[:, :, 1] = saturation

            # Calculate the number of white pixels (with value > brightness) and total pixels within the ball area
            white_pixels = np.sum(roi[:, :, 2] > brightness)
            total_pixels = roi.shape[0] * roi.shape[1]

            # Calculate the proportion of white pixels
            proportion_white = white_pixels / total_pixels

            # Threshold on counts: classify as striped if proportion is greater than threshold q
            q = 0.5  # Adjust the threshold as needed
            ball_type = "Striped" if proportion_white > q else "Solid"

            # Draw the circle with a label indicating the ball type
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
            cv2.putText(frame, ball_type, (x - 2 * radius, y - 2 * radius), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



# Create the main window
root = tk.Tk()
root.title("Snooker Ball")

# Create an "Upload Video" button
upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(pady=20)

# Run the GUI
root.mainloop()

