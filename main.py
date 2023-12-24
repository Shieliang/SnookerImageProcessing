import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Add these global variables at the beginning of your code
color_scoring = {
    "Red": 1,
    "Blue": 5,
    "Green": 3,
    "Yellow": 2,
    "Pink": 6,
    "Black": 7,
    "White": 0,
    "Brown": 4,
}

hole_pockets = {
    "LeftTop": (50, 50),
    "MiddleTop": (500, 50),
    "RightTop": (950, 50),
    "RightBottom": (950, 550),
    "MiddleBottom": (500, 550),
    "LeftBottom": (50, 550),
}


player1_score = 0
player2_score = 0

detected_ball_list = []
def update_player_scores(ball_color):
    global player1_score, player2_score

    if ball_color in color_scoring:
        player1_score += color_scoring[ball_color]

color_masks = [
    {"name": "Red", "lower": np.array([0, 0, 140]), "upper": np.array([20, 255, 255])},
    {"name": "Blue", "lower": np.array([100, 155, 100]), "upper": np.array([140, 255, 255])},
    {"name": "Green", "lower": np.array([70, 200, 100]), "upper": np.array([85, 255, 255])},
    {"name": "Yellow", "lower": np.array([40, 150, 0]), "upper": np.array([60, 255, 255])},
    {"name": "Pink", "lower": np.array([170, 50, 150]), "upper": np.array([255, 255, 255])},
    {"name": "Black", "lower": np.array([0, 0, 0]), "upper": np.array([255, 255, 30])},
    {"name": "White", "lower": np.array([0, 50, 150]), "upper": np.array([100, 100, 255])},
    {"name": "Brown", "lower": np.array([10, 60, 60]), "upper": np.array([20, 255, 255])},
]
def upload_video():
    global video_path
    video_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("Video files", ".mp4;.avi")])
    if video_path:
        play_button.config(state=tk.NORMAL)

def play_video():
    if video_path:
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

        key = cv2.waitKey(5)

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
                ball_color = determine_ball_color(frame[y:y + h, x:x + w])
                detected_ball_list.append(coord)
                cv2.putText(frame, ball_color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                            cv2.LINE_AA)

    return frame

def determine_ball_color(ball_roi):
    # Convert the ball region to HSV
    hsv_ball = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)

    # Iterate through color masks to find the best match
    best_match_color = "Unknown"
    max_match_count = 0

    for color_mask in color_masks:
        lower = color_mask["lower"]
        upper = color_mask["upper"]

        mask = cv2.inRange(hsv_ball, lower, upper)
        match_count = cv2.countNonZero(mask)

        if match_count > max_match_count:
            max_match_count = match_count
            best_match_color = color_mask["name"]

    return best_match_color

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
# Create the main window
root = tk.Tk()
root.title("Snooker Ball")

# Set up the background image
background_image_path =  "1200px-Snooker_Table_Start_Positions.png" # Replace with the path to your image
background_image = tk.PhotoImage(file=background_image_path)

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the center coordinates
x = (screen_width - 1000) // 2
y = (screen_height - 600) // 2

# Set the window size and position
root.geometry(f"1000x600+{x}+{y}")

# Create a larger label for the background image
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Set up button style
button_style = ("Helvetica", 14)
button_bg_color = "lightblue"

# Create an "Upload Video" button
upload_button = tk.Button(root, text="Upload Video", command=upload_video, font=button_style, bg=button_bg_color,
                          highlightthickness=0, bd=0)
upload_button.place(relx=0.5, rely=0.5, anchor="center")

# Create a "Play Video" button
play_button = tk.Button(root, text="Play Video", command=play_video, bg="lightgreen", font=button_style,
                        highlightthickness=0, bd=0)
play_button.place(relx=0.5, rely=0.6, anchor="center")

# Run the GUI
root.mainloop()



