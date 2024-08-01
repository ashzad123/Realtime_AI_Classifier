import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculateAngle(point1, point2, point3):
    '''
    Calculate the angle between three points.
    Args:
        point1, point2, point3: Tuple of (x, y) coordinates of the points.
    Returns:
        angle: The angle in degrees between the three points.
    '''
    a = np.array(point1)  # First point
    b = np.array(point2)  # Mid point
    c = np.array(point3)  # End point

    # Calculate the angle
    ab = a - b
    bc = c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def analyzeImage(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Pose
    results = pose.process(image_rgb)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        landmark_coords = [(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in landmarks]

        # Define keypoints for angle calculation
        left_shoulder = landmark_coords[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmark_coords[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmark_coords[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_shoulder = landmark_coords[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmark_coords[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmark_coords[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_hip = landmark_coords[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmark_coords[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmark_coords[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmark_coords[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmark_coords[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmark_coords[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Calculate angles
        angles = {
            'left_elbow': calculateAngle(left_shoulder, left_elbow, left_wrist),
            'right_elbow': calculateAngle(right_shoulder, right_elbow, right_wrist),
            'left_shoulder': calculateAngle(left_elbow, left_shoulder, left_hip),
            'right_shoulder': calculateAngle(right_elbow, right_shoulder, right_hip),
            'left_knee': calculateAngle(left_hip, left_knee, left_ankle),
            'right_knee': calculateAngle(right_hip, right_knee, right_ankle)
        }

        # Print angles
        for angle_name, angle_value in angles.items():
            print(f'{angle_name.replace("_", " ").title()}: {angle_value:.2f} degrees')

        # Display image with pose landmarks
        plt.figure(figsize=[10,10])
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Pose Analysis")
        plt.axis('off')
        plt.show()

# Test the function with an image
analyzeImage('media/t1.jpg')
