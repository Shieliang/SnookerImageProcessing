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

    # Create a window with fixed size
    cv2.namedWindow('Snooker Ball', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Snooker Ball', 1000, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred_frame = cv2.GaussianBlur(grayscale_frame, (15, 15), 0)

        edges = cv2.Canny(blurred_frame, 10, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Snooker Ball', frame)

        key = cv2.waitKey(25)

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Create the main window
root = tk.Tk()
root.title("Snooker Ball")

# Create an "Upload Video" button
upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(pady=20)

# Run the GUI
root.mainloop()