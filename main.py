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

        proccessed_frame = find_balls(frame)  # Detect snooker balls using Hough Circle Transform

        cv2.imshow('Snooker Ball', proccessed_frame)

        key = cv2.waitKey(16)

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def find_balls(frame):
    ctrs_threshold_frame = []
    ctrs_filtered_list = []

    transformed_blur = cv2.GaussianBlur(frame, (5, 5), 2)  # blur applied
    HSV_frame = cv2.cvtColor(transformed_blur, cv2.COLOR_BGR2HSV)  # rgb version
    gray_frame = cv2.cvtColor(transformed_blur, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, threshold_frame = cv2.threshold(gray_frame, 130, 255, cv2.THRESH_BINARY)  # Thresholding

    # Additional processing for contour separation
    threshold_frame = cv2.erode(threshold_frame, None, iterations=1)
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=7)

    # hsv colors of the snooker table
    lower = np.array([50, 120, 30])
    upper = np.array([70, 255, 255])

    ctrs, hierarchy = cv2.findContours(threshold_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_drawn = frame.copy()
    for i, contour in enumerate(ctrs):
        #cv2.drawContours(contours_drawn, [contour], -1, (0, 255, 0), 2)  # -1 indicates drawing all contours
        # Get the coordinates of the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        ctrs_threshold_frame.append((x, y, w, h))

    mask = cv2.inRange(HSV_frame, lower, upper)  # table's mask
    kernel = np.ones((10, 10), np.uint8)
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # dilate->erode
    _, mask_inv = cv2.threshold(mask_closing, 5, 255, cv2.THRESH_BINARY_INV)  # mask inv
    mask_inv = cv2.erode(mask_inv, None, iterations=1)
    mask_inv = cv2.dilate(mask_inv, None, iterations=10)

    # invert mask to focus on objects on table
    ctrs, hierarchy = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    ctrs_filtered = filter_ctrs(ctrs)
    ctrs_filtered_list.append(ctrs_filtered)

    for coord in ctrs_threshold_frame:
        for c in ctrs_filtered:
            if is_point_inside_contour(coord, cv2.boundingRect(c)):
                x, y, w, h = coord
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return frame


def is_point_inside_contour(point, contour_rect):
    x, y, w, h = contour_rect
    x_center = x + w // 2
    y_center = y + h // 2
    return x <= point[0] <= x + w and y <= point[1] <= y + h


def filter_ctrs(ctrs, min_s=300, max_s=20000, alpha=2):
    filtered_ctrs = []  # list for filtered contours

    for x in range(len(ctrs)):  # for all contours

        rot_rect = cv2.minAreaRect(ctrs[x])  # area of rectangle around contour
        w = rot_rect[1][0]  # width of rectangle
        h = rot_rect[1][1]  # height
        area = cv2.contourArea(ctrs[x])  # contour area

        # Adjusted parameters for better detection
        if (h * alpha < w) or (w * alpha < h):  # if the contour isn't the size of a snooker ball
            continue  # do nothing

        if (area < min_s) or (area > max_s):  # if the contour area is too big/small
            continue  # do nothing

        # if it passed the previous statements, then it is most likely a ball
        filtered_ctrs.append(ctrs[x])  # add contour to filtered contours list

    return filtered_ctrs  # returns filtered contours


# Create the main window
root = tk.Tk()
root.title("Snooker Ball")

# Create an "Upload Video" button
upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(pady=20)

# Run the GUI
root.mainloop()

