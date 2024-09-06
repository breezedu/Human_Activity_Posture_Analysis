import cv2
from ultralytics import YOLO
import torch
import os
import numpy as np

# Initialize the YOLOv8 pose model and set to use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n-pose.pt').to(device)  # Load model and move it to GPU if available

def detect_and_analyze_posture(frame):
    """
    Detects humans in the frame and analyzes their posture to determine if they are sitting or standing
    and whether the posture is good or bad.
    """
    # Convert the frame to RGB as YOLO expects RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO model to detect and estimate pose keypoints on the specified device
    results = model(rgb_frame, save=False, device=device) 
    human_activity = False 

    # process each detected (person )
    for result in results:
        # Loop through each detected person
        if len(result.keypoints) == 0:
            continue
        
        for i, person in enumerate(result.keypoints):
            keypoints = person.xy.cpu().numpy()  # Extract keypoints as a 2D matrix

            # Analyze sitting and standing posture             
            is_sitting = analyze_sitting_posture(keypoints)
            is_good_posture, posture_slope_or_angle = analyze_good_posture(keypoints, is_sitting)

            # Initialize default bounding box coordinates
            x1, y1, x2, y2 = 0, 0, 0, 0
            color = (0, 0, 255)  # Default color: red (bad posture)

            # Draw bounding box and keypoints if bbox is available
            if i < len(result.boxes):
                box = result.boxes[i]  # Extract bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Set color based on posture quality
                color = (0, 255, 0) if is_good_posture else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw keypoints and lines on the frame
            draw_keypoints(frame, keypoints) 

            # Draw lines to show nose-to-ear-to-shoulder alignment
            if is_sitting:
                draw_sitting_posture_lines(frame, keypoints)
           
            draw_posture_lines(frame, keypoints) 
            

            # Display posture status above the bounding box if coordinates are valid
            if x1 != 0 and y1 != 0:
                human_activity = True
                posture_status = "Good Posture" if is_good_posture else "Bad Posture"
                cv2.putText(frame, posture_status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Optionally display the slope of the head-to-neck line
                if posture_slope_or_angle is not None and is_sitting:
                    cv2.putText(frame, f"Sit_Slope: {posture_slope_or_angle:.2f}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2) 
                elif posture_slope_or_angle is not None:
                    cv2.putText(frame, f"Stand_Angle: {posture_slope_or_angle:.2f}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2) 
    return human_activity, frame

def analyze_sitting_posture(keypoints):
    """
    Analyzes keypoints to determine if the detected person is sitting.
    Key points to consider: shoulder, hip, knee, and ankle alignment.
    """
    try:
        # Keypoint indices based on YOLOv8 Pose model
        left_shoulder  = keypoints[0][5]       # Left shoulder (x, y)
        right_shoulder = keypoints[0][6]       # Right shoulder (x, y)
        
        left_hip       = keypoints[0][11]      # Hip (x, y)
        right_hip      = keypoints[0][12] 
        
        left_knee      = keypoints[0][13]      # Knee (x, y)
        right_knee     = keypoints[0][14]
        
        left_ankle     = keypoints[0][15]      # Ankle (x, y)
        right_ankle    = keypoints[0][16]

        # Check alignment of body parts to determine sitting posture
        # the [0, 0] coordinate locates on the top left corner, so 
        if left_knee[1] < left_hip[1] and right_knee[1] < right_hip[1]:
            print("   ------ LKnee: " + str( left_knee[1]) + " LHip: " + str( left_hip[1]) )
            return True
        knee_hip_shoulder_angle_left  = calculate_angle( left_shoulder, left_hip, left_knee) 
        knee_hip_shoulder_angle_right = calculate_angle( right_shoulder, right_hip, right_knee) 

        if knee_hip_shoulder_angle_left < 100 and knee_hip_shoulder_angle_right < 100:
            print( "  ------ Angle: " + str(knee_hip_shoulder_angle_left) ) 
            return True 
            
    except IndexError:
        # In case any keypoints are missing
        return False

    return False

def calculate_angle(point1, point2, point3):
    """
    Calculates the angle formed by three points using the law of cosines.
    In this code, it will calculate ear to shoulder to hip angle 
    """
    # Calculate vectors
    vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

    # Calculate the angle between the vectors
    angle = np.degrees(np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0]))

    # Adjust the angle to be within 0-180 degrees
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle

    return angle


def analyze_good_posture(keypoints, is_sitting):
    """
    Analyzes the quality of posture (good or bad) based on keypoints of the head, neck, and shoulders.
    If is_sitting is true, calculate the slope of ear to shoulder; 
    If is_sitting is false, calculate the angle of ear to shoulder to hip; 
    """
    try:
        # Keypoint indices for posture evaluation
        # nose    = keypoints[0][0]           # nose (x, y)
        left_eye  = keypoints[0][1]       # L eye (x, y)
        right_eye = keypoints[0][2]
        
        left_ear  = keypoints[0][3]       # L ear (x, y)
        right_ear = keypoints[0][4]
        
        left_shoulder  = keypoints[0][5]  # Left shoulder (x, y)
        right_shoulder = keypoints[0][6] # Right shoulder (x, y)
        
        left_hip  = keypoints[0][11]      # left hip 
        right_hip = keypoints[0][12]     
        
        # Calculate the slope of the ear-to-shoulder line
        head_to_neck_slope_left  = abs(left_ear[0]  - left_shoulder[0] ) / max(abs(left_ear[1]  - left_shoulder[1] ), 1e-5)  # Avoid division by zero
        head_to_neck_slope_right = abs(right_ear[0] - right_shoulder[0]) / max(abs(right_ear[1] - right_shoulder[1]), 1e-5)  # Avoid division by zero

        # Evaluate posture based on slope and alignment criteria
        if is_sitting and left_shoulder.all() > 0 and left_ear.all() > 0:
            # Sitting: check if head is vertically aligned with neck and shoulders
            if head_to_neck_slope_left < 0.4 and abs(left_shoulder[1] - right_shoulder[1]) < 20:
                return True, head_to_neck_slope_left
        elif is_sitting and right_shoulder.all() > 0 and right_ear.all() > 0:
            if head_to_neck_slope_right < 0.4 and abs(left_shoulder[1] - right_shoulder[1]) < 20:
                return True, head_to_neck_slope_right
        else:
            # Standing: ensure head-to-shoulder-to-hip-angle is big and shoulders are level
            head_to_shoulder_to_hip_angle = calculate_angle(left_ear, left_shoulder, left_hip)
            if head_to_shoulder_to_hip_angle > 150 and abs(left_shoulder[1] - right_shoulder[1]) < 20:
                return True, head_to_shoulder_to_hip_angle

    except IndexError:
        # Return False if keypoints are missing
        return False, None

    return False, None


def draw_keypoints(frame, keypoints):
    """
    Draws keypoints and lines between them to illustrate pose estimation.
    """
    # Iterate through the keypoints matrix and draw each point
    for keypoint in keypoints:          # each keypoint represent 17-dots of one person in current frame/image
        if( len( keypoint) >1):      
            print( keypoint )   
            for coordinate in keypoint:                
                # x, y = keypoint[:2]  # Extract only x, y coordinates
                x = coordinate[0]
                y = coordinate[1]
                if x > 0 and y > 0:  # Only draw valid points
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1) 
    #return frame 



def draw_sitting_posture_lines(frame, keypoints):
    """
    Draws lines to show head-to-neck-to-shoulder alignment.
    """
    try:
        nose           = keypoints[0][0]    # nose (x, y)
        left_eye       = keypoints[0][1]    # left eye (x, y)
        right_eye      = keypoints[0][2]    # right eye (x, y)
        left_shoulder  = keypoints[0][5]    # Left shoulder (x, y)
        right_shoulder = keypoints[0][6]    # Right shoulder (x, y)

        # Ensure keypoints are valid before drawing lines
        if all(kp[0] > 0 and kp[1] > 0 for kp in [nose, left_eye, left_shoulder]):
            # Draw lines: Head to Neck and Neck to Shoulders
            cv2.line(frame, (int(nose [0]), int(nose [1])), (int(left_eye [0]), int(left_eye [1])), (255, 0, 0), 2)
            cv2.line(frame, (int(left_eye [0]), int(left_eye [1])), (int(left_shoulder[0]), int(left_shoulder[1])), (0, 255, 255), 2)
            
        if all(kp[0] > 0 and kp[1] > 0 for kp in [nose, right_eye, right_shoulder]):
            # Draw lines: Head to Neck and Neck to Shoulders
            cv2.line(frame, (int(nose [0]), int(nose [1])), (int(right_eye [0]), int(right_eye [1])), (0, 255, 255), 2)
            cv2.line(frame, (int(right_eye [0]), int(right_eye [1])), (int(right_shoulder[0]), int(right_shoulder[1])), (0, 255, 255), 2)
            #cv2.line(frame, (int(right_eye [0]), int(right_eye [1])), (int(right_shoulder[0]), int(right_shoulder[1])), (0, 255, 255), 2)
        
    except IndexError:
        # Skip drawing if keypoints are missing
        pass
        
def draw_posture_lines(frame, keypoints):
    """
    Draws lines to show head-to-neck-to-shoulder alignment.
    """
    try:
        nose = keypoints[0][0]              # nose (x, y)         
        left_eye = keypoints[0][1]          # left eye (x, y)       
        right_eye = keypoints[0][2] 

        left_ear = keypoints[0][3]          # ears 
        right_ear = keypoints[0][4] 

        left_shoulder = keypoints[0][5]     # Left shoulder (x, y)   
        right_shoulder = keypoints[0][6]    # Right shoulder (x, y) 

        left_elbow = keypoints[0][7]        # left elbow 
        right_elbow = keypoints[0][8] 

        left_wrist = keypoints[0][9]        # left wrist 
        right_wrist = keypoints[0][10] 


        left_hip = keypoints[0][11]         # left hip
        right_hip = keypoints[0][12]        # right hip 

        
        left_knee = keypoints[0][13]        # left knee 
        right_knee = keypoints[0][14] 
 
        # Ensure keypoints are valid before drawing lines 
        # draw left arm lines
        if all(kp[0] > 0 for kp in [ left_shoulder, left_elbow, left_wrist ]):
            cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])), (int(left_elbow [0]), int(left_elbow [1])), (0, 255, 255), 2) 
            cv2.line(frame, (int(left_elbow[0]), int(left_elbow[1])), (int(left_wrist [0]), int(left_wrist[1])), (0, 255, 255), 2) 
        
        # draw right arm lines
        if all(kp[0] > 0 for kp in [ right_shoulder, right_elbow, right_wrist ]):
            cv2.line(frame, (int(right_shoulder[0]), int(right_shoulder[1])), (int(right_elbow [0]), int(right_elbow [1])), (0, 255, 255), 2) 
            cv2.line(frame, (int(right_elbow[0]), int(right_elbow[1])), (int(right_wrist [0]), int(right_wrist[1])), (0, 255, 255), 2) 
        
        # draw left shoulder-hip-knee    
        if all(kp[0] > 0 for kp in [ left_shoulder, left_hip, left_knee ]):
            cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])), (int(left_hip [0]), int(left_hip [1])), (0, 255, 255), 2) 
            cv2.line(frame, (int(left_hip[0]), int(left_hip[1])), (int(left_knee [0]), int(left_knee[1])), (0, 255, 255), 2) 

        # draw right shoulder-hip-knee    
        if all(kp[0] > 0 for kp in [ right_shoulder, right_hip, right_knee ]):
            cv2.line(frame, (int(right_shoulder[0]), int(right_shoulder[1])), (int(right_hip [0]), int(right_hip [1])), (0, 255, 255), 2) 
            cv2.line(frame, (int(right_hip[0]), int(right_hip[1])), (int(right_knee [0]), int(right_knee[1])), (0, 255, 255), 2) 

        # draw ear-shoulder
        if all(kp[0] > 0 for kp in [ left_ear, left_shoulder ]):
            cv2.line(frame, (int(left_ear[0]), int(left_ear[1])), (int(left_shoulder [0]), int(left_shoulder [1])), (0, 255, 255), 2) 
        if all(kp[0] > 0 for kp in [ right_ear, right_shoulder ]):
            cv2.line(frame, (int(right_ear[0]), int(right_ear[1])), (int(right_shoulder [0]), int(right_shoulder[1])), (0, 255, 255), 2) 
        
        # draw shoulder line
        if all(kp[0] > 0 for kp in [ left_shoulder, right_shoulder ]):
            cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])), (int(right_shoulder [0]), int(right_shoulder[1])), (0, 255, 255), 2) 
        

    except IndexError:
        # Skip drawing if keypoints are missing
        pass

def main():
    # Specify the path to the input video file
    input_video_path  = "/home/jeff/HomeSecurity_VideoRecording/videos/20240905/camera_0_20240905_083312.avi"

    # Specify the path to save the output video
    output_video_path = "/home/jeff/HomeSecurity_VideoRecording/videos/20240905/camera_0_20240905_083312_Short_POSTUREWITHLINES0906pm.avi"  # Replace with your desired output path
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)  # Create directory if it doesn't exist

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video {input_video_path}")
        return

    # Get properties of the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the code-cv and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze posture for each frame
        human_activity, analyzed_frame = detect_and_analyze_posture(frame)

        # if there's human detected in current frame, write the analyzed frame to the output video
        if human_activity:
            out.write(analyzed_frame)

    # Release the resources
    cap.release()
    out.release()
    cv2.destroyAllWindows() 
    print("--- Output file saved to: " + str(output_video_path) )

if __name__ == "__main__":
    main()




    # Specify the path to the input video file
#    input_video_path = "/home/jeff/HomeSecurity_VideoRecording/videos/20240905/camera_0_20240905_083312.avi"  # Replace with your input video file path

    # Specify the path to save the output video
#    output_video_path = "/home/jeff/HomeSecurity_VideoRecording/videos/20240905/camera_0_20240905_083312_POSTUREWITHLINES.avi"  # Replace with your desired output path
#    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)  # Create directory if it doesn't exist


# END 