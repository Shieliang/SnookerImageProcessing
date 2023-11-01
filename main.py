import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

color_ranges = {
    'green': ([40, 40, 40], [80, 255, 255]),
}

threshold_area = 500  # Adjust this threshold based on your needs

def upload_video():
    file_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("Video files", "*.mp4;*.avi")])
    if file_path:
        play_video(file_path)

def apply_color_mask(frame, color):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range, upper_range = color_ranges[color]
    mask = cv2.inRange(hsv_frame, np.array(lower_range), np.array(upper_range))
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")

    # Create a window with fixed size
    cv2.namedWindow('Comparison', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Comparison', 1500, 600)

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

        # Filter contours based on area
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > threshold_area]

        # Create a mask for the table contours
        table_mask = np.zeros_like(frame)
        cv2.drawContours(table_mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        # Invert the table mask
        inverted_table_mask = cv2.bitwise_not(table_mask[:, :, 0])
        inverted_table_mask = cv2.merge([inverted_table_mask, inverted_table_mask, inverted_table_mask])

        # Apply green color mask to the frame
        green_mask = apply_color_mask(frame, 'green')

        # Exclude table contours from the original frame using the green mask
        frame_no_table = cv2.subtract(frame, green_mask)

        # Display original and modified frames side by side
        side_by_side = np.hstack((frame, frame_no_table))
        cv2.imshow('Comparison', side_by_side)

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
