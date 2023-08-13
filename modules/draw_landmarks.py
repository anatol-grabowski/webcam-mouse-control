import cv2
import numpy as np
from .gaze_predictor import train_indices


def draw_landmarks(img, faces):
    if faces is None:
        return img

    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    img_h, img_w, _ = img.shape
    for points in faces:
        points = np.multiply(points, [img_w, img_h]).astype(int)
        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(points[RIGHT_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        cv2.circle(img, center_left, int(l_radius), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(img, center_right, int(r_radius), (0, 255, 0), 1, cv2.LINE_AA)

        for x, y in points[train_indices]:
            cv2.circle(img, (x, y), 1, (255, 255, 0), -1)
        cv2.circle(img, center_right, 1, (0, 255, 0), 1)
