import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


color_ranges = {
    'Red': (np.array([0, 128, 78]), np.array([10, 255, 255])),
    'Blue': (np.array([105, 100, 100]), np.array([135, 255, 255])),
    'Yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
    'Brown': (np.array([10, 80, 60]), np.array([30, 255, 255])),
    'Pink': (np.array([145, 100, 100]), np.array([175, 255, 255])),
    'Black': (np.array([0, 0, 0]), np.array([180, 255, 30]))
}

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

        key = cv2.waitKey(25)

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def find_balls(frame):
    transformed_blur = cv2.GaussianBlur(frame, (5, 5), 2)  # blur applied
    blur_RGB = cv2.cvtColor(transformed_blur, cv2.COLOR_BGR2RGB)  # rgb version

    # hsv colors of the snooker table
    lower = np.array([50, 120, 30])
    upper = np.array([70, 255, 255])

    hsv = cv2.cvtColor(blur_RGB, cv2.COLOR_RGB2HSV)  # convert to hsv
    mask = cv2.inRange(hsv, lower, upper)  # table's mask

    # apply closing
    kernel = np.ones((5, 5), np.uint8)
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # dilate->erode

    # invert mask to focus on objects on table
    _, mask_inv = cv2.threshold(mask_closing, 5, 255, cv2.THRESH_BINARY_INV)  # mask inv

    masked_img = cv2.bitwise_and(frame,frame,mask=mask_inv)

    ctrs, hierarchy = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    detected_objects = draw_rectangles(ctrs, masked_img)
    ctrs_filtered = filter_ctrs(ctrs)
    detected_objects_filtered = draw_rectangles(ctrs_filtered, masked_img)
    ctrs_color = find_ctrs_color(ctrs_filtered, masked_img)
    ctrs_color = cv2.addWeighted(ctrs_color, 0.5, masked_img, 0.5, 0)  # contours color image + transformed image
    return detected_objects_filtered


def draw_rectangles(ctrs, img):
    output = img.copy()

    for i in range(len(ctrs)):
        M = cv2.moments(ctrs[i])  # moments
        rot_rect = cv2.minAreaRect(ctrs[i])
        w = rot_rect[1][0]  # width
        h = rot_rect[1][1]  # height

        box = np.int64(cv2.boxPoints(rot_rect))
        cv2.drawContours(output, [box], 0, (255, 100, 0), 2)  # draws box

    return output

def filter_ctrs(ctrs, min_s=300, max_s=15000, alpha=2):
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


def find_ctrs_color(ctrs, input_img):
    K = np.ones((3, 3), np.uint8)  # filter
    output = input_img.copy()  # np.zeros(input_img.shape,np.uint8) # empty img
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)  # gray version
    mask = np.zeros(gray.shape, np.uint8)  # empty mask

    for i in range(len(ctrs)):  # for all contours

        # find center of contour
        M = cv2.moments(ctrs[i])
        cX = int(M['m10'] / M['m00'])  # X pos of contour center
        cY = int(M['m01'] / M['m00'])  # Y pos

        mask[...] = 0  # reset the mask for every ball

        cv2.drawContours(mask, ctrs, i, 255,
                         -1)  # draws the mask of current contour (every ball is getting masked each iteration)

        mask = cv2.erode(mask, K, iterations=3)  # erode mask to filter green color around the balls contours

        output = cv2.circle(output,  # img to draw on
                            (cX, cY),  # position on img
                            20,  # radius of circle - size of drawn snooker ball
                            cv2.mean(input_img, mask),
                            # color mean of each contour-color of each ball (src_img=transformed img)
                            -1)  # -1 to fill ball with color
    return output

# Create the main window
root = tk.Tk()
root.title("Snooker Ball")

# Create an "Upload Video" button
upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(pady=20)

# Run the GUI
root.mainloop()

