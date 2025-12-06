import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Load your meme image (replace with your actual path)
meme_image = cv2.imread("C:/Users/grw23969/Downloads/yoo-michael-jordan.gif")  # Change this to your image path

if meme_image is None:
    print("Error: Could not load meme image. Please check the path.")
    exit()

# Resize meme image for display window
display_height = 480
aspect_ratio = meme_image.shape[1] / meme_image.shape[0]
display_width = int(display_height * aspect_ratio)
meme_display = cv2.resize(meme_image, (display_width, display_height))

# Create a blank/black image for when pose is not detected
blank_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)
cv2.putText(blank_image, "Make the pose!", (50, display_height//2), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit")
print("Make the meme pose to trigger the image!")

def check_meme_pose(keypoints):
    """
    Detect the 'shocked/excited' pose:
    - At least one hand extended toward camera (detected by being close to shoulder width)
    - OR one hand toward camera and one toward chest
    - Hands roughly at upper body level
    
    Keypoints: 5=left_shoulder, 6=right_shoulder, 
               7=left_elbow, 8=right_elbow,
               9=left_wrist, 10=right_wrist
    """
    if len(keypoints) < 11:
        return False
    
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    # Check if keypoints are visible (reduced threshold for higher sensitivity)
    if (left_shoulder[2] < 0.4 or right_shoulder[2] < 0.4):
        return False
    
    # At least one wrist must be visible
    left_wrist_visible = left_wrist[2] > 0.4
    right_wrist_visible = right_wrist[2] > 0.4
    
    if not (left_wrist_visible or right_wrist_visible):
        return False
    
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    
    # Check each hand separately
    left_hand_raised = False
    right_hand_raised = False
    
    if left_wrist_visible:
        # Hand is in upper body region (more lenient range)
        left_in_range = left_wrist[1] < shoulder_y + 200 and left_wrist[1] > shoulder_y - 200
        # Hand is either: toward camera (near center) OR pulled toward chest
        left_toward_camera = abs(left_wrist[0] - left_shoulder[0]) < shoulder_width * 0.8
        left_hand_raised = left_in_range and left_toward_camera
    
    if right_wrist_visible:
        # Hand is in upper body region (more lenient range)
        right_in_range = right_wrist[1] < shoulder_y + 200 and right_wrist[1] > shoulder_y - 200
        # Hand is either: toward camera (near center) OR pulled toward chest
        right_toward_camera = abs(right_wrist[0] - right_shoulder[0]) < shoulder_width * 0.8
        right_hand_raised = right_in_range and right_toward_camera
    
    # Trigger if at least one hand is in position
    return left_hand_raised or right_hand_raised

# Initialize display image
current_display = blank_image.copy()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Run pose detection
    results = model(frame, verbose=False)
    
    # Draw pose keypoints
    annotated_frame = results[0].plot()
    
    # Check for the meme pose
    pose_detected = False
    
    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        for person_keypoints in results[0].keypoints:
            # Get keypoints as numpy array [x, y, confidence]
            kpts = person_keypoints.data[0].cpu().numpy()
            
            if check_meme_pose(kpts):
                pose_detected = True
                break
    
    # Update the display based on pose detection
    if pose_detected:
        current_display = meme_display.copy()
        # Add indicator on webcam feed
        cv2.putText(annotated_frame, "POSE DETECTED!", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        current_display = blank_image.copy()
    
    # Display instructions
    cv2.putText(annotated_frame, "Put hand(s) toward camera to trigger", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show both windows
    cv2.imshow('Webcam Feed', annotated_frame)
    cv2.imshow('Meme Display', current_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()