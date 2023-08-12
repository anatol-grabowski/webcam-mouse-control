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

face_mesh = mp.solutions.face_mesh.FaceMesh(
    # static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
dataset = []
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
        dataset.append(datapoint)
    draw_landmarks(img, faces)

Dataset.save_dataset(dataset)

cv2.destroyAllWindows()
