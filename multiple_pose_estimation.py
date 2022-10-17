import mediapipe as mp
import cv2
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles


class PoseEstimation:
    def __init__(self, min_pose_detect_conf=0.8, min_pose_track_conf=0.5):

        self.pose = mp_pose.Pose(min_detection_confidence=min_pose_detect_conf,
                                 min_tracking_confidence=min_pose_track_conf)

    def process_frame(self, frame):
        self.frame_height, self.frame_width = frame.shape[:-1]

        image = frame.copy()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.pose_results = self.pose.process(image)
        self.pose_detected = bool(self.pose_results.pose_landmarks)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.image = image
        return self.image


    # POSE

    def get_pose_coords(self, landmark_index):
        if self.pose_detected:
            return tuple(np.multiply(
                np.array((self.pose_results.pose_landmarks.landmark[landmark_index].x,
                          self.pose_results.pose_landmarks.landmark[landmark_index].y,
                          self.pose_results.pose_landmarks.landmark[landmark_index].z)),
                [self.frame_width, self.frame_height, self.frame_width]).astype(int))

    def get_exact_pose_coords(self, landmark_index):
        if self.pose_detected:
            return tuple(np.multiply(
                np.array((self.pose_results.pose_landmarks.landmark[landmark_index].x,
                          self.pose_results.pose_landmarks.landmark[landmark_index].y,
                          self.pose_results.pose_landmarks.landmark[landmark_index].z)),
                [self.frame_width, self.frame_height, self.frame_width]))

    def get_pose_joint_angle(self, joint):
        if self.pose_detected:
            co1, co2, co3 = [self.get_exact_pose_coords(joint[i]) for i in range(3)]

            radxy = np.arctan2(co3[1] - co2[1], co3[0] - co2[0]) - np.arctan2(co1[1] - co2[1], co1[0] - co2[0])
            anglexy = np.abs(radxy * 180 / np.pi)
            anglexy = min(anglexy, 360 - anglexy)
            return anglexy

    def show_pose_joint_angles(self, image, joint_list):
        if self.pose_detected:
            for joint in joint_list:
                joint_angle = self.get_pose_joint_angle(joint)

                cv2.putText(image, str(round(joint_angle, 2)), self.get_pose_coords(joint[1])[:2],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)

            return image

    def get_pose_slope_angle(self, index1, index2):
        if self.pose_detected:
            co1, co2 = self.get_exact_pose_coords(index1), self.get_exact_pose_coords(index2)

            slope_radxy = np.arctan2(co1[1] - co2[1], co1[0] - co2[0])
            slope_anglexy = np.abs(slope_radxy * 180 / np.pi)

            return slope_anglexy

    def draw_pose(self):
        if self.pose_detected:
            mp_drawing.draw_landmarks(
                self.image,
                self.pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    def detect_hand_raise(self, print_result=False, screen_label=False):
        if self.pose_detected:
            raised_hand_list = []

            right_shoulder = self.get_exact_pose_coords(12)
            right_elbow = self.get_exact_pose_coords(14)
            right_wrist = self.get_exact_pose_coords(16)

            if right_wrist[1] <= right_shoulder[1] and \
                    right_wrist[1] <= right_elbow[1] and \
                    (120 >= self.get_pose_joint_angle((12, 14, 16)) >= 30 or
                     135 >= self.get_pose_slope_angle(14, 16) >= 45) and 10 <= self.get_pose_slope_angle(14, 16) <= 170:
                if print_result:
                    print("Right hand raised", self.get_pose_coords(16)[:-1])
                raised_hand_list.append(("R", right_wrist[:-1]))

            left_shoulder = self.get_exact_pose_coords(11)
            left_elbow = self.get_exact_pose_coords(13)
            left_wrist = self.get_exact_pose_coords(15)

            if left_wrist[1] <= left_shoulder[1] and \
                    left_wrist[1] <= left_elbow[1] and \
                    (120 >= self.get_pose_joint_angle((11, 13, 15)) >= 30 or
                     135 >= self.get_pose_slope_angle(13, 15) >= 45) and 10 <= self.get_pose_slope_angle(13, 15) <= 170:
                if print_result:
                    print("Left hand raised", self.get_pose_coords(15)[:-1])
                raised_hand_list.append(("L", left_wrist[:-1]))

            if raised_hand_list and screen_label:
                cv2.putText(self.image,
                            " ".join(
                                [{"R": "Right hand raised", "L": "Left hand raised"}[h[0]] for h in raised_hand_list]),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            return raised_hand_list

    def get_distance(self, p1, p2):
        if self.pose_detected or self.hands_detected:
            dx, dy, dz = (p2[i] - p1[i] for i in range(3))
            dxy = (dx ** 2 + dy ** 2) ** 0.5

            return dx, dy, dz, dxy

    def get_max_min_x_y(self):
        if self.pose_detected:
            lx = [self.get_pose_coords(ix)[0] for ix in range(33)]
            ly = [self.get_pose_coords(iy)[1] for iy in range(33)]
            max_lx, min_lx = max(lx), min(lx)
            max_ly, min_ly = max(ly), min(ly)
            # c = max_lx - min_lx
            # d = max_ly - min_ly
            return min(self.frame_width, max_lx), max(0, min_lx), \
                   min(self.frame_height, max_ly), max(0, min_ly - 100)

    def draw_over(self):
        if self.pose_detected:
            max_x, min_x, max_y, min_y = self.get_max_min_x_y()

            # lsd, rsd = self.get_pose_coords(11)[0], self.get_pose_coords(12)[0]
            # if lsd > rsd:
            #     max_x, min_x = lsd, rsd
            # else:
            #     min_x, max_x = lsd, rsd

            # print(max_x, min_x, max_y, min_y)
            new_frame = np.copy(self.image)
            new_frame[min_y:max_y, min_x:max_x] = 0
            return new_frame
    
    def sit(self, print_result = False, screen_label = False):
        if self.pose_detected:
            sit_list = []

            left_shoulder = self.get_exact_pose_coords(11)
            right_shoulder = self.get_exact_pose_coords(12)
            left_hip = self.get_exact_pose_coords(23)
            right_hip = self.get_exact_pose_coords(24)
            left_knee = self.get_exact_pose_coords(25)
            right_knee = self.get_exact_pose_coords(26)
            
            #from the side
            if 100 >= self.get_pose_joint_angle((11,23,25)) >= 50 or 100 >= self.get_pose_joint_angle((12,24,26)) >= 50:
                if print_result:
                    print("Sitting Down", self.get_pose_coords((23)[:-1],(24)[:-1]))

                return sit_list
