import cv2
import numpy as np

last_lines = []

def crop_frame(frame, mode="day"):
    """
    Crops the input frame based on the specified mode by applying a mask.

    Parameters:
    frame (numpy.ndarray): The input image frame to be cropped.
    mode (str): The mode for cropping the frame. Options are "day", "night", "crosswalk", and "collision".
                Default is "day".

    Returns:
    numpy.ndarray: The cropped image frame with the applied mask.
    """
    height, width = frame.shape[:2]
    mask = np.zeros_like(frame)
    if mode == "day":
        points = [(width * 3 // 20, height), (width * 18 // 20, height), (width // 2, height * 14 // 24)]
    elif mode == "night":
        points = [(width * 4 // 20, height), (width * 18 // 20, height), (width // 2, height * 16 // 24)]
    elif mode == "crosswalk":
        points = [(width * 5 // 32, height), (width * 29 // 64, height * 3 // 4), (width * 57 // 64, height)]
    elif mode == "collision":
        points = [(width * 5 // 32, height), (width * 29 // 64, height * 22 // 30), (width * 48 // 64, height)]
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], (255, 255, 255))
    return cv2.bitwise_and(frame, mask)


def isolate_lane_colors(frame, mode="day"):
    """
    Isolate lane colors in a given frame based on the specified mode.

    Parameters:
    frame (numpy.ndarray): The input image frame in BGR format.
    mode (str): The mode for thresholding. Options are "day", "night", "crosswalk", and "collision".
                Default is "day".

    Returns:
    numpy.ndarray: A binary mask where the lane colors are isolated based on the specified mode.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresholds = {
        "day": (200, 255),
        "night": (70, 255),
        "crosswalk": (100, 255),
        "collision": (67, 255)
    }
    mask_white = cv2.inRange(gray, *thresholds[mode])
    return mask_white


def detect_lane_lines(mask):
    """
    Detects lane lines in a given binary mask image using the Hough Line Transform.

    Args:
        mask (numpy.ndarray): A binary mask image where the lane lines are highlighted.

    Returns:
        numpy.ndarray: An array of detected lines, where each line is represented by a 4-element vector (x1, y1, x2, y2).
    """
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=60, minLineLength=50, maxLineGap=50)
    return lines


def filter_lines_by_geometry(lines, frame_height, mode = "day"):
    """
        Filters lines based on their geometric properties and the specified mode.

        Parameters:
        lines (list of list of int): A list of lines, where each line is represented by a list of four integers [x1, y1, x2, y2].
        frame_height (int): The height of the frame, used to determine the minimum length of lines to retain.
        mode (str, optional): The mode of operation, which affects the filtering criteria. Default is "day".
            - "day": Retains lines with an absolute slope greater than 0.55 and length greater than 10% of the frame height.
            - "night": Retains lines with an absolute slope greater than 0.5 and length greater than 10% of the frame height.
            - "crosswalk": Retains lines with an absolute slope greater than 0.5 and length greater than 20% of the frame height.
            - "collision": Retains lines with an absolute slope greater than 0.5 and length greater than 2% of the frame height.

        Returns:
        list of tuple: A list of filtered lines, where each line is represented by a tuple (x1, y1, x2, y2).
    """
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        # Retain lines that are mostly vertical and long enough
        if mode == "night":
            if abs(slope) > 0.5 and length > frame_height * 0.1:  
                filtered_lines.append((x1, y1, x2, y2))
        elif mode == "day":
            if abs(slope) > 0.55 and length > frame_height * 0.1:  
                filtered_lines.append((x1, y1, x2, y2))
        elif mode == "crosswalk":
            if abs(slope) > 0.5 and length > frame_height * 0.2:  
                filtered_lines.append((x1, y1, x2, y2))
        elif mode == "collision":
            if abs(slope) > 0.5 and length > frame_height * 0.02:  
                filtered_lines.append((x1, y1, x2, y2))
    return filtered_lines


def group_and_verify_lines(filtered_lines, frame_width):
    """
    Groups and verifies lines into left and right lanes based on their slopes and positions.

    Args:
        filtered_lines (list of tuples): A list of tuples where each tuple contains four integers (x1, y1, x2, y2) representing the coordinates of a line segment.
        frame_width (int): The width of the frame/image.

    Returns:
        list: A list containing two elements:
            - left_lane (tuple or None): A tuple (slope, intercept) representing the average left lane line, or None if no left lane lines are found.
            - right_lane (tuple or None): A tuple (slope, intercept) representing the average right lane line, or None if no right lane lines are found.
    """
    left_lines = []
    right_lines = []
    for x1, y1, x2, y2 in filtered_lines:
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        intercept = y1 - slope * x1
        # Classify lines as left or right based on slope and position
        if slope < 0 and x1 < frame_width // 2:  # Left lane
            left_lines.append((slope, intercept))
        elif slope > 0 and x1 > frame_width // 2:  # Right lane
            right_lines.append((slope, intercept))
    # Take the median of left and right lane lines
    left_lane = np.median(left_lines, axis=0) if left_lines else None
    right_lane = np.median(right_lines, axis=0) if right_lines else None
    return [left_lane, right_lane]


def draw_lane_rectangle(frame, left_lane, right_lane, mode = "day"):
    """
        Draws a filled polygon representing the lane area on the given frame.

        Parameters:
        frame (numpy.ndarray): The image frame on which to draw the lane rectangle.
        left_lane (tuple): The slope and intercept of the left lane line.
        right_lane (tuple): The slope and intercept of the right lane line.
        mode (str, optional): The mode of operation which affects the height of the polygon. 
                              Options are "day", "night", "crosswalk", and "collision". 
                              Default is "day".

        Returns:
        numpy.ndarray: The frame with the lane rectangle drawn on it.
    """
    def make_line_points(y1, y2, line):
        if line is not None:
            slope, intercept = line
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return (x1, y1), (x2, y2)
        return None, None
    height, _ = frame.shape[:2]
    if mode == "day":
        y1 = height  
        y2 = int(height * 0.65) 
    elif mode == "night":
        y1 = height  
        y2 = int(height * 0.75)  
    elif mode == "crosswalk":
        y1 = height  
        y2 = int(height * 0.8) 
    elif mode == "collision":
        y1 = height  
        y2 = int(height * 0.85)  
    # Get points for left and right lanes
    left_bottom, left_top = make_line_points(y1, y2, left_lane)
    right_bottom, right_top = make_line_points(y1, y2, right_lane)
    # Define the polygon points
    if left_bottom and left_top and right_bottom and right_top:
        polygon_points = np.array([left_bottom, left_top, right_top, right_bottom], dtype=np.int32)
        # Draw the filled polygon
        cv2.fillPoly(frame, [polygon_points], ((147,20,255)))  # Pink color (BGR format)
    return frame

def calculate_lane():
    """
    Calculate the average slope and intercept for the left and right lanes.

    This function processes the global variable `last_lines`, which is expected to be a list of tuples.
    Each tuple contains two elements: the first element represents the left lane line parameters (slope and intercept),
    and the second element represents the right lane line parameters (slope and intercept).

    Returns:
        list: A list containing two elements:
            - The first element is a tuple representing the average slope and intercept of the left lane, or None if no left lane data is available.
            - The second element is a tuple representing the average slope and intercept of the right lane, or None if no right lane data is available.
    """
    global last_lines
    if not last_lines:
        return [None, None]
    left_slopes = []
    left_intercepts = []
    right_slopes = []
    right_intercepts = []
    for lines in last_lines:
        if lines[0] is not None:  # Left lane
            left_slopes.append(lines[0][0])
            left_intercepts.append(lines[0][1])
        if lines[1] is not None:  # Right lane
            right_slopes.append(lines[1][0])
            right_intercepts.append(lines[1][1])
    left_lane = None
    if left_slopes:
        avg_slope = np.mean(left_slopes)
        avg_intercept = np.mean(left_intercepts)
        left_lane = (avg_slope, avg_intercept)
    right_lane = None
    if right_slopes:
        avg_slope = np.mean(right_slopes)
        avg_intercept = np.mean(right_intercepts)
        right_lane = (avg_slope, avg_intercept)
    return [left_lane, right_lane]


def detect_draw_vehicle(image):
    """
    Detects and draws a rectangle around the largest object in the specified region of the image if it meets certain criteria.

    Args:
        image (numpy.ndarray): The input image in which to detect the object.

    Returns:
        tuple: A tuple containing:
            - image (numpy.ndarray): The image with the rectangle and text drawn if an object is detected.
            - detected (bool): A boolean indicating whether an object was detected.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask_white = cv2.inRange(gray, 100, 255)
    points = np.array([[(200, height), (600, 510), (width, height)]])
    mask = np.zeros_like(mask_white)
    cv2.fillPoly(mask, [points], 255)
    masked_image = cv2.bitwise_and(mask_white, mask)
    kernel = np.ones((5, 5), np.uint8)
    opened_image = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_size = 0
    detected = False
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        size = cv2.arcLength(contour, True)
        if size > 150 and 540 < y + h / 2 < 575:  # Middle y-coordinate of the contour
            if max_size < size:
                max_size = size
                temp = contour
            detected = True
    if detected:        
        x, y, w, h = cv2.boundingRect(temp)
        cv2.rectangle(image, (x , y - 20), (x + w , y + h + 10), (147,20,255), 3)
        cv2.putText(image, "TOO CLOSE!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (147,20,255) , 2)
    return image, detected


def show_lanes(frame, direct_to_right):
    """
    Draws a rectangle and text on the given frame to indicate lane switching direction.

    Parameters:
    frame (numpy.ndarray): The image frame on which to draw.
    direct_to_right (bool): If True, indicates switching to the right lane; otherwise, indicates switching to the left lane.

    Returns:
    numpy.ndarray: The modified image frame with the rectangle and text drawn on it.
    """
    if direct_to_right:
        frame = cv2.rectangle(frame, (0, 0), (350, 40), (0, 0, 0), -1)
        return cv2.putText(frame, "Switch to Right Lane", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (147,20,255) , 3,
                           cv2.LINE_AA)
    else:
        frame = cv2.rectangle(frame, (0, 0), (350, 40), (0, 0, 0), -1)
        return cv2.putText(frame, "Switch to Left Lane", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (147,20,255), 3,
                           cv2.LINE_AA)


def detect_draw_crosswalk(frame, res):
    """
    Detects and draws a rectangle around crosswalks in a given video frame.

    Args:
        frame (numpy.ndarray): The input video frame in which to detect crosswalks.
        res (numpy.ndarray): The output frame where the detected crosswalks will be drawn.

    Returns:
        tuple: A tuple containing:
            - res (numpy.ndarray): The output frame with rectangles drawn around detected crosswalks.
            - detect (bool): A boolean indicating whether a crosswalk was detected in the frame.
    """
    detect = False
    height, width = frame.shape[:2]
    rectangle = np.array([[(300, height - 100), (300, height - 150),
                           (width - 500, height - 150), (width - 500, height - 100)]])
    mask = np.zeros_like(frame)
    mask = cv2.fillPoly(mask, rectangle, 255)
    frame = cv2.bitwise_and(frame,mask)  
    frame = cv2.dilate(frame, np.ones((1, 30)))
    frame = cv2.erode(frame, np.ones((7, 40)))
    frame = cv2.dilate(frame, np.ones((15, 60)))
    counters, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in counters:
        if cv2.arcLength(cnt, True) > 800:  # if the component is big enough, this is a crosswalk
            detect = True
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 20  # Adjust padding size to make the rectangle wider
            cv2.rectangle(res, (x - padding, y), (x + w + padding, y + h), (147,20,255), 3)
    return res, detect


def show_crosswalk(res):
    res = cv2.rectangle(res, (0, 0), (350, 40), (0, 0, 0), -1)
    return cv2.putText(res, "CrossWalk Detected!", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (147,20,255), 3,
                       cv2.LINE_AA)


def proccess_video():
    """
    Processes a video to detect lanes, crosswalks, and potential collisions.
    This function loads a video file, processes each frame to detect lanes, crosswalks, and objects,
    and displays the processed frames with visual alerts for detected events.
    Global Variables:
    - last_lines: A list to store the detected lane lines from the last few frames for smoothing.
    """
    global last_lines
    left_count = 0
    right_count = 0
    flag = 0 
    alert_timeout = 0
    mode = "day"
    # Load the video
    video_path = "regular_drive.mp4"
    #video_path = "night_drive.mp4"
    #video_path = "crosswalk_drive.mp4"
    #video_path = "collision_drive.mp4"
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video.")
        exit()

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        origin_frame = frame.copy()
        frame = crop_frame(frame, mode)
        mask_white = isolate_lane_colors(frame, mode)
        
        blur = cv2.GaussianBlur(mask_white, (5, 5), 0)
        img_gray = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2GRAY)
        mask_white_temp = cv2.inRange(img_gray, 100, 255)
        # crosswalk detection
        if mode == "crosswalk":
            res_temp, detect = detect_draw_crosswalk(mask_white_temp, origin_frame)
            if detect:
                origin_frame = show_crosswalk(res_temp)
                alert_timeout = -1         
        # collision detection
        if mode == "collision":
            res_temp, detect = detect_draw_vehicle(origin_frame)
            if detect:
                origin_frame = res_temp
                flag = -1
        # Apply Canny edge detection
        edges = cv2.Canny(blur, 30, 60)  # Adjust thresholds as needed
        lines = detect_lane_lines(edges)
        if lines is not None:
            filtered_lines = filter_lines_by_geometry(lines, frame.shape[0], mode)
            lines = group_and_verify_lines(filtered_lines, frame.shape[1])
            if lines is not None:
                if lines[0] is not None and lines[1] is not None:
                    last_lines.append(lines)
                if alert_timeout == 0 or flag == -1:
                    if lines[0] is None:
                        left_count += 1
                    else:
                        left_count = 0

                    if lines[1] is None:
                        right_count += 1
                    else:
                        right_count = 0

                    if mode != "collision" and left_count == 12:
                        alert_timeout = 1
                        left_count = 0
                    elif mode =="collision" and right_count == 40:
                        alert_timeout = 2
                        right_count = 0
                    elif right_count == 12:
                        alert_timeout = 2
                        right_count = 0
                if len(last_lines) > 12:  # Use the last 12 frames for smoothing
                    last_lines.pop(0)

                smoothed_lines = calculate_lane()
                if alert_timeout == 0 and flag == 0:
                    origin_frame = draw_lane_rectangle(origin_frame, smoothed_lines[0], smoothed_lines[1],mode)
                elif alert_timeout > 0:  
                    if alert_timeout > 0 and alert_timeout % 2 == 1:
                        origin_frame = show_lanes(origin_frame, True)
                    elif alert_timeout > 0 and alert_timeout % 2 == 0:
                        origin_frame = show_lanes(origin_frame, False)
                    alert_timeout += 2
                    if mode == "collision" and alert_timeout >= 150:  
                        alert_timeout = 0
                    elif alert_timeout >= 230:  
                        alert_timeout = 0

                elif alert_timeout < 0:  # if this is crosswalk mode
                    alert_timeout -= 1
                    if alert_timeout <= -50:  # if we dont want to show the alert anymore
                        alert_timeout = 0
                        left_count = 0
                        right_count = 0
                elif flag < 0:  # if this is collision mode
                    flag -= 1
                    if flag <= -20:  # if we dont want to show the alert anymore
                        alert_timeout = 0
                        flag = 0
                        left_count = 0
                        right_count = 0
                

                cv2.imshow("Lane Detection", origin_frame)
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    #Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    proccess_video()
