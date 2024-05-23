import cv2
import numpy as np

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    Draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def fit_lines(lines, y_min, y_max):
    """
    Fit lines to the detected lane markings.
    
    Returns two lines representing left and right lanes.
    """
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append((x1, y1))
                left_lines.append((x2, y2))
            else:
                right_lines.append((x1, y1))
                right_lines.append((x2, y2))
                
    if left_lines and right_lines:
        left_points = np.array(left_lines)
        right_points = np.array(right_lines)
        
        left_coeffs = np.polyfit(left_points[:, 1], left_points[:, 0], 1)
        right_coeffs = np.polyfit(right_points[:, 1], right_points[:, 0], 1)
        
        left_lane = np.poly1d(left_coeffs)
        right_lane = np.poly1d(right_coeffs)
        
        left_line = [(int(left_lane(y_min)), int(y_min)), (int(left_lane(y_max)), int(y_max))]
        right_line = [(int(right_lane(y_min)), int(y_min)), (int(right_lane(y_max)), int(y_max))]
        
        return left_line, right_line
    else:
        return None, None

def process_frame(frame):
    """
    Apply lane detection on a single frame.
    """
    # Convert image to HLS color space
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    # Apply color thresholding to detect white and yellow lanes
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hls, lower_white, upper_white)
    
    lower_yellow = np.array([10, 0, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
    
    # Combine the masks
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Define region of interest
    height, width = frame.shape[:2]
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(blur, vertices)
    
    # Apply Canny edge detection
    edges = cv2.Canny(roi, 50, 150)
    
    # Apply Hough transform to detect lines
    lines = hough_lines(edges, rho=1, theta=np.pi/180, threshold=15, min_line_len=40, max_line_gap=20)
    
    # Fit lines to the detected lane markings
    left_line, right_line = fit_lines(lines, height * 0.6, height)
    
    # Create a blank image to draw lines on
    line_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    
    # Draw the lane lines
    if left_line is not None and right_line is not None:
        cv2.line(line_img, left_line[0], left_line[1], [0, 0, 255], 10)
        cv2.line(line_img, right_line[0], right_line[1], [0, 0, 255], 10)
    
    # Overlay lines on original frame
    result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
    
    return result

# Open video file
cap = cv2.VideoCapture('input_video.mp4')

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    processed_frame = process_frame(frame)
    
    # Display processed frame
    cv2.imshow('Lane Detection', processed_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
