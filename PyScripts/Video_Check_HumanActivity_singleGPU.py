import cv2
import os
from ultralytics import YOLO
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the YOLOv8 model for person detection on the specified device
model = YOLO('yolov8n.pt').to(device)  # Change 'yolov8n.pt' to other versions if needed (e.g., 'yolov8s.pt')

def detect_human(frame):
    """Detect humans in the given frame using YOLOv8."""
    # Convert frame to RGB as YOLOv8 expects RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the YOLO model on the frame
    results = model(rgb_frame)

    human_detected = False

    for result in results:
        for box in result.boxes:
            # Check if the class is "person" and confidence is above threshold
            if box.cls == 0 and box.conf > 0.5:  # class 0 corresponds to "person" in YOLO
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
                # Draw a blue rectangle around the detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                human_detected = True

    return human_detected, frame

def process_video(video_path):
    """Process the video to check for human activity and mark detections."""
    cap = cv2.VideoCapture(video_path)
    human_activity_detected = False
    frames_with_detections = []

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        human_detected, marked_frame = detect_human(frame)
        if human_detected:
            human_activity_detected = True
            frames_with_detections.append(marked_frame)

    cap.release()

    # Save or delete video based on human activity detection
    if not human_activity_detected:
        print(f"No human activity detected, deleting video: {video_path}")
        os.remove(video_path)
    else:
        # If human activity is detected, save the video with annotations
        save_path = os.path.splitext(video_path)[0] + "_marked.mp4"
        save_marked_video(save_path, frames_with_detections)
        print(f"Human activity detected, saved marked video: {save_path}")

def save_marked_video(output_path, frames):
    """Save frames with detections as a new video."""
    if not frames:
        print("No frames to save.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for the output video
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

if __name__ == "__main__":
    # Replace with the path to the directory containing videos to process
    videos_directory = "/home/jeff/HomeSecurity_VideoRecording/videos/20240903"
    
    # Iterate over all video files in the directory
    for video_file in os.listdir(videos_directory):
        if video_file.endswith((".mp4", ".avi", ".mov")):  # Adjust file types as necessary
            video_path = os.path.join(videos_directory, video_file)
            process_video(video_path)
