import cv2
import os
from ultralytics import YOLO
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Set the path to your YOLOv8 model
MODEL_PATH = 'yolov8n.pt'

def setup_ddp(rank, world_size):
    """Initialize the DDP environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Ensure this port is free
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_ddp():
    """Cleanup the DDP environment."""
    dist.destroy_process_group()

def detect_human(model, device, frame):
    """Detect humans in the given frame using YOLOv8."""
    # Convert frame to RGB as YOLOv8 expects RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame to tensor and move to the correct device
    tensor_frame = torch.from_numpy(rgb_frame).float().to(device)
    tensor_frame = tensor_frame.permute(2, 0, 1).unsqueeze(0) / 255.0  # Add batch dimension and normalize

    # Run the YOLO model on the frame
    results = model(tensor_frame)

    human_detected = False

    for result in results:
        for box in result.boxes:
            # Check if the detected class is "person" (class index 0 in YOLO)
            if box.cls == 0 and box.conf > 0.5:  # class 0 corresponds to "person" in YOLO
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
                # Draw a blue rectangle around the detected person on the original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                human_detected = True

    return human_detected, frame

def process_video(rank, world_size, video_path):
    """Process the video to check for human activity and mark detections using DDP."""
    setup_ddp(rank, world_size)

    # Set the device for this process
    device = torch.device(f'cuda:{rank}')

    # Load and wrap the model with DDP
    model = YOLO(MODEL_PATH)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    cap = cv2.VideoCapture(video_path)
    human_activity_detected = False
    frames_with_detections = []

    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        cleanup_ddp()
        return

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        human_detected, marked_frame = detect_human(model, device, frame)
        if human_detected:
            human_activity_detected = True
            frames_with_detections.append(marked_frame)

    cap.release()

    # Save or delete the video based on human activity detection
    if not human_activity_detected and rank == 0:
        print(f"No human activity detected, deleting video: {video_path}")
        os.remove(video_path)
    elif human_activity_detected and rank == 0:
        save_path = os.path.splitext(video_path)[0] + "_marked.mp4"
        save_marked_video(save_path, frames_with_detections)
        print(f"Human activity detected, saved marked video: {save_path}")

    cleanup_ddp()

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

def main(videos_directory, world_size):
    """Main function to spawn DDP processes for video processing."""
    video_files = [f for f in os.listdir(videos_directory) if f.endswith((".mp4", ".avi", ".mov"))]

    # Use multiprocessing to spawn one process per GPU
    mp.spawn(process_videos, args=(world_size, videos_directory, video_files), nprocs=world_size)

def process_videos(rank, world_size, videos_directory, video_files):
    """Process multiple videos in parallel across GPUs."""
    for video_file in video_files[rank::world_size]:  # Distribute videos among GPUs evenly
        video_path = os.path.join(videos_directory, video_file)
        process_video(rank, world_size, video_path)

if __name__ == "__main__":
    videos_directory = "/home/jeff/HomeSecurity_VideoRecording/videos/20240901"
    world_size = torch.cuda.device_count()

    if world_size < 1:
        print("No GPUs found. Exiting.")
    else:
        main(videos_directory, world_size)



# videos_directory = "/home/jeff/HomeSecurity_VideoRecording/videos/20240901"
# END
