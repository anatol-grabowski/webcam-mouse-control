import dlib
import cv2
import mediapipe as mp
import pynput
import glob
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import re
import sys


def get_xy_from_filename(filename):
    pattern = r'\[(\d+) (\d+)\]'
    match = re.search(pattern, filename)
    x, y = map(int, match.groups())
    return x, y


def mp_landmarks_to_points(multi_face_landmarks):
    if multi_face_landmarks is None:
        return None
    points_list = [[[p.x, p.y] for p in face.landmark] for face in multi_face_landmarks]
    points = np.array(points_list)
    return points


def mp_detect_faces(rgb, num_warmup=3, num_avg=3):
    for i in range(num_warmup):
        output = face_mesh.process(rgb)
    output = face_mesh.process(rgb)
    if output is None:
        return None
    faces = mp_landmarks_to_points(output.multi_face_landmarks)
    for i in range(num_avg - 1):
        output = face_mesh.process(rgb)
        if output is None:
            return None
        faces += mp_landmarks_to_points(output.multi_face_landmarks)
    faces /= num_avg
    return faces


def draw_landmarks(img, faces):
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    img_h, img_w, _ = img.shape
    print(len(faces) if faces is not None else None)
    if faces is None:
        return img

    for points in faces:
        points = np.multiply(points, [img_w, img_h]).astype(int)
        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(points[RIGHT_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        cv2.circle(img, center_left, int(l_radius), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(img, center_right, int(r_radius), (0, 255, 0), 1, cv2.LINE_AA)

        for x, y in points:
            cv2.circle(img, (x, y), 1, (255, 255, 0), -1)
        cv2.circle(img, center_right, 1, (0, 255, 0), 1)
    cv2.imshow("Photo with Landmarks", img)
    cv2.waitKey(1)


def get_paths(globs):
    glpaths = []
    for gl in globs:
        glpaths.extend(glob.glob(gl))
    return [str(p) for p in glpaths]


photo_globs = [
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T15:57:06.820873-continuous-ok/brio *.jpeg',
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T16:33:38.163179-3-ok/brio *-1 *.jpeg',
    '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-09T15:37:18.761700-first-spiral-ok/brio *.jpeg',
]
photo_paths = get_paths(photo_globs)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    # static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
dataset = []
for filepath in photo_paths:
    x, y = get_xy_from_filename(filepath)
    img = cv2.imread(filepath)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mp_detect_faces(rgb)
    print(filepath, x, y, faces is not None)
    if faces is not None:
        datapoint = {
            'filename': filepath,
            'cursor': (x, y),
            'face': faces[0],
        }
        dataset.append(datapoint)
    draw_landmarks(img, faces)

print(dataset)

cv2.destroyAllWindows()
