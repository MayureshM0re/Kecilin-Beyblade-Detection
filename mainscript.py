import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv
import time

# Load the YOLO model for detecting Beyblades
model = YOLO('best.pt')  # Update with your correct model path

# Open the video file
cap = cv2.VideoCapture('C:/Beyblade/Beybladefight.mp4')

# Directory to save winner and loser images
output_dir = 'C:/Beyblade/output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize parameters for Optical Flow
prev_gray = None
beyblade_1_motion = []
beyblade_2_motion = []
motion_threshold = 0.3  # Threshold to consider motion as stopped
tracking_window_size = 30  # Number of frames to track motion

# Variables to track stopped status
beyblade_1_stopped = False
beyblade_2_stopped = False
game_over = False
saved_screenshot = False

# Variables to track spin duration
battle_start_time = time.time()  # Start the timer at the beginning
remaining_spin_duration = 0.0  # Remaining spin duration of the winning beyblade

# CSV file path for saving results
csv_file_path = 'C:/Beyblade/battle_results.csv'

# Write CSV header if file doesn't exist
if not os.path.isfile(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Total Spin Duration (seconds)', 'Winner', 'Remaining Spin Duration (seconds)'])

# Function to save the screenshot of the winner and loser
def save_beyblade_image(frame, label, bbox, filename):
    x1, y1, x2, y2 = bbox
    cropped_beyblade = frame[int(y1):int(y2), int(x1):int(x2)]
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cropped_beyblade)
    print(f'{label} image saved as {filename}')

# Function to calculate optical flow and determine if a Beyblade has stopped
def calculate_optical_flow(prev_gray, gray, bbox):
    p0 = np.array([[((bbox[0] + bbox[2]) / 2), ((bbox[1] + bbox[3]) / 2)]], dtype=np.float32)  # Center point
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
    
    # Calculate displacement (motion) between two frames
    if p1 is not None and st[0] == 1:
        displacement = np.linalg.norm(p1 - p0)
        return displacement
    return 0.0

# Function to estimate remaining spin duration (customize based on your logic)
def estimated_remaining_time_function():
    # Example estimation logic (replace with your own)
    average_spin_duration = 5  # Assume an average of 5 seconds remaining
    return average_spin_duration

# Main loop for video processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to avoid zoom-in issue (keeping original resolution)
    frame = cv2.resize(frame, (640, 360))

    # Convert frame to grayscale for optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform inference on the current frame using YOLO model
    results = model(frame)
    annotated_frame = results[0].plot()

    # Extract detections (bounding boxes)
    beyblade_1_bbox = None
    beyblade_2_bbox = None

    for detection in results[0].boxes.data:
        # Move the bounding box tensor to CPU before processing
        detection = detection.cpu().numpy()
        x1, y1, x2, y2, conf, cls = detection
        bbox = (x1, y1, x2, y2)

        # Track Beyblade 1 and Beyblade 2 based on their class IDs
        if cls == 0:  # Assuming class 0 is Beyblade 1
            beyblade_1_bbox = bbox
        elif cls == 1:  # Assuming class 1 is Beyblade 2
            beyblade_2_bbox = bbox

    # Track motion using Optical Flow for Beyblade 1
    if prev_gray is not None and beyblade_1_bbox is not None:
        motion_1 = calculate_optical_flow(prev_gray, gray, beyblade_1_bbox)
        beyblade_1_motion.append(motion_1)
        if len(beyblade_1_motion) > tracking_window_size:
            beyblade_1_motion.pop(0)

    # Track motion using Optical Flow for Beyblade 2
    if prev_gray is not None and beyblade_2_bbox is not None:
        motion_2 = calculate_optical_flow(prev_gray, gray, beyblade_2_bbox)
        beyblade_2_motion.append(motion_2)
        if len(beyblade_2_motion) > tracking_window_size:
            beyblade_2_motion.pop(0)

    # Update previous frame
    prev_gray = gray

    # Determine if either Beyblade has stopped
    if not game_over:
        if np.mean(beyblade_1_motion[-tracking_window_size:]) < motion_threshold and not beyblade_1_stopped:
            beyblade_1_stopped = True
            print("Beyblade 1 has stopped!")
        if np.mean(beyblade_2_motion[-tracking_window_size:]) < motion_threshold and not beyblade_2_stopped:
            beyblade_2_stopped = True
            print("Beyblade 2 has stopped!")

        # Declare the winner once one Beyblade has stopped
        if beyblade_1_stopped and not beyblade_2_stopped and not saved_screenshot:
            total_spin_duration = time.time() - battle_start_time
            remaining_spin_duration = max(0, estimated_remaining_time_function())  # Estimate remaining duration
            print("Beyblade 2 WINS!")
            save_beyblade_image(frame, "Winner (Beyblade 2)", beyblade_2_bbox, "Beyblade_2_winner.png")
            save_beyblade_image(frame, "Loser (Beyblade 1)", beyblade_1_bbox, "Beyblade_1_loser.png")
            # Save results to CSV
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([total_spin_duration, "Beyblade 2", remaining_spin_duration])
            game_over = True
            saved_screenshot = True
        elif beyblade_2_stopped and not beyblade_1_stopped and not saved_screenshot:
            total_spin_duration = time.time() - battle_start_time
            remaining_spin_duration = max(0, estimated_remaining_time_function())  # Estimate remaining duration
            print("Beyblade 1 WINS!")
            save_beyblade_image(frame, "Winner (Beyblade 1)", beyblade_1_bbox, "Beyblade_1_winner.png")
            save_beyblade_image(frame, "Loser (Beyblade 2)", beyblade_2_bbox, "Beyblade_2_loser.png")
            # Save results to CSV
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([total_spin_duration, "Beyblade 1", remaining_spin_duration])
            game_over = True
            saved_screenshot = True
        elif beyblade_1_stopped and beyblade_2_stopped:
            print("DRAW!")
            game_over = True

    # Display the frame with annotations
    cv2.imshow('Detections', annotated_frame)

    # Press 'q' to exit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
