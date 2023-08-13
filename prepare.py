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
import pickle
from modules.draw_landmarks import draw_landmarks
from modules.dataset import Dataset
from modules.mediapipe_detect_faces import mediapipe_detect_faces
from tqdm import tqdm


def get_xy_from_filename(filename):
    pattern = r'\[(\d+) (\d+)\]'
    match = re.search(pattern, filename)
    x, y = map(int, match.groups())
    return x, y


def get_paths(globs):
    glpaths = []
    for gl in globs:
        glpaths.extend(glob.glob(gl))
    return [str(p) for p in glpaths]


photo_globs = [
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T15:57:06.820873-continuous-ok/brio *.jpeg',
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T16:33:38.163179-3-ok/brio *-1 *.jpeg',
    '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/*/brio *.jpeg',
]
photo_paths = get_paths(photo_globs)


def detect_blink(face):
    blink_threshold = 0.35
    left_h = np.linalg.norm(face[386] - face[374])
    left_w = np.linalg.norm(face[362] - face[263])
    left_blink = left_h < blink_threshold * left_w
    right_h = np.linalg.norm(face[145] - face[159])
    right_w = np.linalg.norm(face[133] - face[33])
    right_blink = right_h < blink_threshold * right_w
    return left_blink, right_blink


face_mesh = mp.solutions.face_mesh.FaceMesh(
    # static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
dataset = []
num_blinks = 0
for filepath in photo_paths:
    xy = np.array(get_xy_from_filename(filepath))
    monsize = np.array([2560, 1440])
    img = cv2.imread(filepath)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mediapipe_detect_faces(face_mesh, rgb)
    print(filepath, xy, faces is not None)
    if faces is not None:
        datapoint = {
            'filename': filepath,
            'cursor': xy,
            'cursor_norm': xy / monsize * 2 - 1,
            'landmarks': faces[0],
        }
        left_blink, right_blink = detect_blink(faces[0])
        if not left_blink and not right_blink:
            dataset.append(datapoint)
        else:
            num_blinks += 1
            print('skip blink')
            cv2.putText(img, f"{'L' if left_blink else ' '} {'R' if right_blink else ''}",
                        (100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    draw_landmarks(img, faces)

Dataset.save_dataset(dataset)
print(f'{num_blinks=}')

cv2.destroyAllWindows()
