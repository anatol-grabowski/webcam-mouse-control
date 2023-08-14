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
from modules.detect_blink import detect_blink
from modules.get_paths import get_paths, get_xy_from_filename
from tqdm import tqdm
from modules.predict_cursor import pixelxy_to_cursor, cursor_to_pixelxy
from screeninfo import get_monitors

photo_globs = [
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T15:57:06.820873-continuous-ok/brio *.jpeg',
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T16:33:38.163179-3-ok/brio *-1 *.jpeg',
    '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/*/*.jpeg',
]
photo_paths = get_paths(photo_globs)[::-1]  # reverse for tqdm to work better


face_mesh = mp.solutions.face_mesh.FaceMesh(
    # static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
monname = 'eDP-1'  # 'eDP-1' (integrated) or 'DP-3' (Dell)
mon = next((mon for mon in get_monitors() if mon.name == monname))
monsize = np.array([mon.width, mon.height])
dataset = Dataset()
ds = Dataset.load()

num_blinks = 0
for filepath in tqdm(photo_paths):
    xy = np.array(get_xy_from_filename(filepath))
    cur = pixelxy_to_cursor(xy, monsize)
    datapoint = next((dp for dp in ds.datapoints if dp['label'] == filepath), None)
    if datapoint is not None:
        dataset.add_datapoint(filepath, datapoint['face'], cur)
        continue
    img = cv2.imread(filepath)
    imsize = np.array([img.shape[1], img.shape[0]])
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mediapipe_detect_faces(face_mesh, rgb)

    # print(filepath, cur, faces is not None)
    if faces is not None:
        face = faces[0]
        left_blink, right_blink = detect_blink(face)
        if not left_blink and not right_blink:
            dataset.add_datapoint(filepath, face, cur)
        else:
            num_blinks += 1
            print('skipped')
            cv2.putText(img, f"blink", (100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.circle(img, cursor_to_pixelxy(cur, imsize).astype(int), 2, (0, 255, 0))
    draw_landmarks(img, faces)
    cv2.imshow('img', img)
    cv2.waitKey(1)

dataset.store()
print(f'{num_blinks=}')

cv2.destroyAllWindows()
