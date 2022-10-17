from Detector import *
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = Detector(model_type="IS")

def main():
  while cap.isOpened():
    ret, frame = cap.read()
    keypoint_frame = detector.onTime(frame)
    image = cv2.cvtColor(keypoint_frame, cv2.COLOR_BGR2RGB)

    

    cv2.imshow("Feed",image)

    if cv2.waitKey(1) == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows

if __name__ == '__main__':
  main()