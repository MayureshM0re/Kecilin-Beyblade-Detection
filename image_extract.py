import cv2
import os

# Video file path
video_path = r"C:\Beyblade\Beybladefight.mp4"
output_dir = r"C:\Beyblade\extimages"  # Change to your desired output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get total frames and fps
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate total duration in seconds
seconds = round(frames / fps)

# Total number of frames to extract
frame_total = 120
i = 0

print(f"Extracting {frame_total} frames from {video_path}...")

while cap.isOpened():
    # Set the position of the video to the desired time
    cap.set(cv2.CAP_PROP_POS_MSEC, (i * ((seconds / frame_total) * 1000)))
    
    # Read the frame
    flag, frame = cap.read()
    
    if not flag:  # If frame not read successfully, exit loop
        break

    # Construct image path
    image_path = os.path.join(output_dir, f"img_{i}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Saved: {image_path}")  # Notify which frame is saved

    i += 1
    # Stop extraction if the desired number of frames is reached
    if i >= frame_total:
        break

# Release the video capture object and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Frame extraction completed.")
