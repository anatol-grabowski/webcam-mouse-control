import re
from modules.draw_landmarks import draw_landmarks
from modules.mp_landmarks_to_points import mp_landmarks_to_points
import cv2
import mediapipe as mp
import pyautogui
from pynput.mouse import Listener
from pynput import keyboard, mouse
import pynput
import uuid
import time
import datetime
import os
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from modules.eye_position_predictor import EyePositionPredictor, train_indices
from modules.mediapipe_detect_faces import mediapipe_detect_faces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'{device=}')


def list_webcams():
    from collections import defaultdict
    import re
    import subprocess

    # Command as a list of strings

    completed_process = subprocess.run(
        'v4l2-ctl --list-devices 2>/dev/null',
        shell=True, stdout=subprocess.PIPE, text=True
    )

    stdout_output = completed_process.stdout
    # print("Stdout Output:")
    # print(stdout_output)

    device_info = defaultdict(list)
    current_device = ""

    for line in stdout_output.splitlines():
        line = line.strip()
        if line:
            if re.match(r"^\w+.*:", line):
                current_device = line
            else:
                device_info[current_device].append(line)

    parsed_dict = dict(device_info)

    # print(parsed_dict)
    return parsed_dict


def cams_init():
    webcams = list_webcams()
    intcams = webcams[[cam for cam in webcams.keys() if 'Integrated' in cam][0]]
    briocams = webcams[[cam for cam in webcams.keys() if 'BRIO' in cam][0]]
    camsdict = {}

    print('cam1')
    cam1 = cv2.VideoCapture(briocams[0])
    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam1.set(cv2.CAP_PROP_FPS, 60)
    camsdict['brio'] = cam1

    # print('cam2')
    # cam2 = cv2.VideoCapture(briocams[2])
    # cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 340)
    # cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 340)
    # cam2.set(cv2.CAP_PROP_FPS, 30)
    # camsdict['brioBW'] = cam2 # regular BRIO cam hangs when BW cam is in use, same behavior in guvcview

    # print('cam3')
    # cam3 = cv2.VideoCapture(intcams[0])
    # cam3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cam3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cam3.set(cv2.CAP_PROP_FPS, 30)
    # camsdict['integrated'] = cam3

    # print('cam4')
    # cam4 = cv2.VideoCapture(intcams[2])
    # cam4.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cam4.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    # cam4.set(cv2.CAP_PROP_FPS, 30)
    # camsdict['integratedBW'] = cam4 # regular cam hangs when BW cam is in use, but ok in guvcview, bug in cv2?

    return camsdict


def cams_deinit(camsdict):
    for name, cam in camsdict.items():
        cam.release()


def cams_capture(cams):
    frames = {}
    for camname, cam in cams.items():
        ret, frame = cam.read()
        frames[camname] = frame
    return frames


def predict(frame):
    rgb = cv2.cvtColor(frame, 1, cv2.COLOR_BGR2RGB)
    faces = mediapipe_detect_faces(face_mesh, rgb, num_warmup=1, num_avg=3)
    if faces is None:
        return None, None
    X = faces[0][train_indices].ravel().reshape(1, -1)
    X = torch.tensor(X, dtype=torch.float32)

    imsize = np.array([frame.shape[1], frame.shape[0]])
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            prediction = model(X)
        predictions.append(prediction)
        m_cursor = (prediction[0].numpy() + 1) / 2 * imsize
        cv2.circle(frame, m_cursor.astype(int), 2, (255, 0, 0), -1)
    predictions = torch.stack(predictions)
    y = torch.mean(predictions, dim=0)
    m_cursor = (y[0].numpy() + 1) / 2 * imsize
    print('im', imsize, m_cursor, y)
    cv2.circle(frame, m_cursor.astype(int), 4, (255, 0, 0), -1)

    monsize = np.array([2560, 1440])
    cursor = (y[0].numpy() + 1) / 2 * monsize
    cursor = cursor.clip([0, 0], [2560, 1440])
    return cursor, faces


def get_paths(globs):
    glpaths = []
    for gl in globs:
        glpaths.extend(glob.glob(gl))
    return [str(p) for p in glpaths]


photo_globs = [
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T15:57:06.820873-continuous-ok/brio *.jpeg',
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T16:33:38.163179-3-ok/brio *-1 *.jpeg',
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-09T15:37:18.761700-first-spiral-ok/brio *-1 *.jpeg',
    '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/*/brio *.jpeg',
]
photo_paths = get_paths(photo_globs)


pyautogui.FAILSAFE = False

face_mesh = mp.solutions.face_mesh.FaceMesh(
    # static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


mpaths = sys.argv[1:]
models = [EyePositionPredictor.load_from_file(p) for p in mpaths]
scores = np.array([float(re.match(r'.* (0.\d+) .*', p)[1]) for p in mpaths])
print(f'{scores=}')
# min_perf = scores.min()
# max_perf = scores.max()
# normalized_perf = [(perf - min_perf) / (max_perf - min_perf) for perf in scores]
# weights = [1 - np for np in normalized_perf]
# sum_weights = sum(weights)
# normalized_weights = [weight / sum_weights for weight in weights]
# print("Normalized Weights:", normalized_weights)


numavg = 3
avgs = np.zeros(shape=(numavg, 2))


def detect_blink(face):
    blink_threshold = 0.35
    left_h = np.linalg.norm(face[386] - face[374])
    left_w = np.linalg.norm(face[362] - face[263])
    left_blink = left_h < blink_threshold * left_w
    right_h = np.linalg.norm(face[145] - face[159])
    right_w = np.linalg.norm(face[133] - face[33])
    right_blink = right_h < blink_threshold * right_w
    return left_blink, right_blink


def main():
    global avgs, numavg
    print('hello')

    cams = cams_init()

    # while True:
    #     frames = cams_capture(cams)
    #     frame = frames['brio']

    for filepath in photo_paths:
        frame = cv2.imread(filepath)

        cursor, faces = predict(frame)
        print(cursor)
        if cursor is not None:
            avgs = np.roll(avgs, -1, axis=0)
            avgs[-1] = cursor
            avg = avgs.mean(axis=0)
            print(avg)
            pyautogui.moveTo(*avg, 0.0, pyautogui.easeInOutQuad)
            left_blink, right_blink = detect_blink(faces[0])
            cv2.putText(frame, f"{'L' if left_blink else ' '} {'R' if right_blink else ''}",
                        (100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

        draw_landmarks(frame, faces)
        input()

    cams_deinit(cams)


main()
