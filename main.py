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
from modules.eye_position_predictor import EyePositionPredictor


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
    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam1.set(cv2.CAP_PROP_FPS, 30)
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


face_mesh = mp.solutions.face_mesh.FaceMesh(
    # static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
model = EyePositionPredictor.load_from_file('./data/model.pickle')


def predict(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb)
    faces = mp_landmarks_to_points(output.multi_face_landmarks)
    if faces is None:
        return None, None
    X = faces[0].ravel().reshape(1, -1)
    X = torch.tensor(X, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y = model(X)

    monsize = np.array([2560, 1440])
    print(y)
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
    '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-09T15:37:18.761700-first-spiral-ok/brio *.jpeg',
]
photo_paths = get_paths(photo_globs)


def main():
    print('hello')

    cams = cams_init()

    # while True:
    #     frames = cams_capture(cams)
    #     frame = frames['brio']

    pyautogui.FAILSAFE = False
    for filepath in photo_paths:
        frame = cv2.imread(filepath)

        cursor, faces = predict(frame)
        print(cursor)
        if cursor is not None:
            pyautogui.moveTo(*cursor)
        draw_landmarks(frame, faces)
        # input()

    cams_deinit(cams)


main()
