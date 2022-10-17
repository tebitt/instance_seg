import cv2
import mediapipe as mp
import numpy as np
from multiple_pose_estimation import PoseEstimation

def draw_bbox(frame, res):
  for id in res:
    x, y, w, h, hand_raised = (res[id][k] for k in ("x", "y", "w", "h", "hand_raised"))
    # max_x, min_x, max_y, min_y, hand_raised = res[id]
    color = (0, 255, 0) if hand_raised else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

PES = [PoseEstimation(min_pose_detect_conf=0.8, min_pose_track_conf=0.5) for _ in range(5)]


def main(): 
  frame = cv2.imread("images/sit.jpeg")
  frame = cv2.resize(frame, (1280, 720))
  res = dict()
  img = np.frombuffer(frame, dtype=np.uint8).reshape(720, 1280, 3)
  image = np.copy(img)

  for id, PE in enumerate(PES):
    PE.process_frame(image)

    if PE.pose_detected:
      image = np.copy(PE.draw_over())
      max_x, min_x, max_y, min_y = PE.get_max_min_x_y()
      print(min_x, min_y, max_x, max_y)
      res[int(id)] = {"x": int(min_x), "y": int(min_y), "w": int(max_x - min_x),
                      "h": int(max_y - min_y), "sit": bool(PE.sit())}

  print(res)
  draw_bbox(img, res)
  cv2.imshow("frame", img)
  cv2.waitKey(0)

if __name__ == '__main__':
  main()