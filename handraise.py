import cv2
import mediapipe as mp
import numpy as np
from multiple_pose_estimation import PoseEstimation
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
  while cap.isOpened():
    ret, frame = cap.read()
    #Recolor BGR -> RGB
    image.flags.writeable = False
    #Make Detection
    results = pose.process(image)
    #Recolor RGB -> BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    #Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)       
                              )
    #Extract landmarks
    try:
      landmarks = results.pose_landmarks.landmark
      print(landmarks)
    except:
      pass
    
    cv2.imshow("Feed",image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows