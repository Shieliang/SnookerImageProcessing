import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time

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

last_seen_time = {}
detected_ball_list = []

player_score = 0

def update_player_scores(ball_color):
    global player_score
    global last_seen_time

    if ball_color in color_scoring:
        current_time = time.time()

        # Check if 1 second has passed since the last detection
        if ball_color not in last_seen_time or (current_time - last_seen_time[ball_color]) > 1:
            player_score += color_scoring[ball_color]
            last_seen_time[ball_color] = current_time


color_masks = [
    {"name": "Red", "lower": np.array([0, 0, 140]), "upper": np.array([20, 255, 255])},
    {"name": "Blue", "lower": np.array([100, 155, 100]), "upper": np.array([140, 255, 255])},
    {"name": "Green", "lower": np.array([80, 120, 100]), "upper": np.array([85, 200, 150])},
    {"name": "Yellow", "lower": np.array([40, 150, 0]), "upper": np.array([60, 255, 255])},
    {"name": "Pink", "lower": np.array([170, 50, 150]), "upper": np.array([255, 100, 255])},
    {"name": "Black", "lower": np.array([0, 0, 0]), "upper": np.array([255, 255, 30])},
    {"name": "White", "lower": np.array([0, 0, 150]), "upper": np.array([100, 90, 255])},
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

        processedframe = find_balls(frame)  # Detect snooker balls using Hough Circle Transform
        draw_pocket_rectangles(processedframe)

        cv2.putText(processedframe, f"Player Score: {player_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Snooker Ball', processedframe)

        key = cv2.waitKey(5)

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def find_balls(frame):
    ctrs_threshold_frame = []
    ctrs_filtered_list = []
    global last_seen_time
    current_time = time.time()


    transformed_blur = cv2.GaussianBlur(frame, (5, 5), 2)  # blur applied
    HSV_frame = cv2.cvtColor(transformed_blur, cv2.COLOR_BGR2HSV)  # rgb version
    gray_frame = cv2.cvtColor(transformed_blur, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, threshold_frame = cv2.threshold(gray_frame, 130, 255, cv2.THRESH_BINARY)  # Thresholding

    # Additional processing for contour separation
    threshold_frame = cv2.erode(threshold_frame, None, iterations=1)
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=6)

    ctrs, hierarchy = cv2.findContours(threshold_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(ctrs):
        # Get the coordinates of the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        ctrs_threshold_frame.append((x, y, w, h))

        # hsv colors of the snooker table
    lower = np.array([50, 120, 30])
    upper = np.array([70, 255, 255])
    light_blue_lower = np.array([90, 50, 140])
    light_blue_upper = np.array([102, 255, 215])

    green_mask = cv2.inRange(HSV_frame, lower, upper)  # table's mask
    ligh_blue_mask = cv2.inRange(HSV_frame,light_blue_lower,light_blue_upper)
    mask = cv2.bitwise_or(green_mask,ligh_blue_mask)
    kernel = np.ones((15, 15), np.uint8)
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # dilate->erode
    _, mask_inv = cv2.threshold(mask_closing, 5, 255, cv2.THRESH_BINARY_INV)  # mask inv
    mask_inv = cv2.erode(mask_inv, None, iterations=1)
    mask_inv = cv2.dilate(mask_inv, None, iterations=9)

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
                for pocket, (pocket_x, pocket_y) in hole_pockets.items():
                    hole_rect = (pocket_x - 2, pocket_y - 2, 54, 54)

                    # Check if the ball touches the border of the hole
                    if is_point_on_border_of_hole((x + w // 2, y + h // 2), hole_rect):
                        update_player_scores(ball_color)
                    else:
                        continue

    return frame

def is_point_on_border_of_hole(point, hole_rect, buffer=5):
    x, y, w, h = hole_rect

    # Check if the point is on the border of the hole within the specified buffer
    return (
        (x - buffer) <= point[0] <= (x + w + buffer) and
        (y - buffer) <= point[1] <= (y + h + buffer) and
        (
            abs(point[0] - x) < buffer or
            abs(point[0] - (x + w)) < buffer or
            abs(point[1] - y) < buffer or
            abs(point[1] - (y + h)) < buffer
        )
    )
def draw_pocket_rectangles(frame):
    for pocket, (x, y) in hole_pockets.items():
        square_size = 2  # Adjust the size of the square as needed
        cv2.rectangle(frame, (x - square_size, y - square_size), (x + 50 + square_size, y + 50 + square_size),
                      (0, 255, 255), 2)

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


def filter_ctrs(ctrs, min_s=700, max_s=17000, alpha=2):
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

# Set up the background image
background_image_path =  "1200px-Snooker_Table_Start_Positions.png" # Replace with the path to your image
background_image = tk.PhotoImage(file=background_image_path)

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

hole_pockets = {
    "LeftTop": (int(screen_width - screen_width), int(screen_height - screen_height + 40)),
    "MiddleTop": (int((screen_width / 2) - 20), int(screen_height - screen_height)+20),
    "RightTop": (int(screen_width - 40), int(screen_height - screen_height + 30)),
    "RightBottom": (int(screen_width - screen_width - 10), int(screen_height-80)),
    "MiddleBottom": (int((screen_width / 2) - 20), int(screen_height-60)),
    "LeftBottom": (int(screen_width-30), int(screen_height-80)),
}

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
