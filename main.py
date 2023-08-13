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
from modules.cursor_predictor import EyePositionPredictor, train_indices
from modules.mediapipe_detect_faces import mediapipe_detect_faces
from modules.predict_cursor import predict_cursor, cursor_to_pixelxy
from modules.webcam import list_webcams
from modules.detect_blink import detect_blink
from modules.get_paths import get_paths
from modules.webcam import list_webcams, cams_init, cams_capture


def draw_cursors(frame, cursor, cursors):
    imsize = np.array([frame.shape[1], frame.shape[0]])
    cv2.circle(frame, cursor_to_pixelxy(cursor, imsize).astype(int), 4, (255, 0, 0), -1)
    for cur in cursors:
        cv2.circle(frame, cursor_to_pixelxy(cur, imsize).astype(int), 2, (255, 0, 0), -1)


def draw(frame, cursor, cursors, faces):
    draw_landmarks(frame, faces)
    frame = cv2.flip(frame, 1)
    if cursor is not None:
        left_blink, right_blink = detect_blink(faces[0])
        cv2.putText(frame, f"{'L' if left_blink else ' '} {'R' if right_blink else ''}",
                    (100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        draw_cursors(frame, cursor, cursors)

    cv2.namedWindow('Fullscreen Image', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Fullscreen Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Fullscreen Image", frame)
    cv2.waitKey(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'{device=}')


photo_globs = [
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T15:57:06.820873-continuous-ok/brio *.jpeg',
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T16:33:38.163179-3-ok/brio *-1 *.jpeg',
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-09T15:37:18.761700-first-spiral-ok/brio *-1 *.jpeg',
    '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/*/brio *.jpeg',
]
photo_paths = get_paths(photo_globs)


pyautogui.FAILSAFE = False


mpaths = sys.argv[1:]
models = [EyePositionPredictor.load_from_file(p) for p in mpaths]
scores = np.array([float(re.match(r'.* (0.\d+) .*', p)[1]) for p in mpaths])
models = {model: 1 for model, score in zip(models, scores)}
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


def main():
    global avgs, numavg
    print('hello')

    cams = cams_init()
    monsize = np.array([2560, 1440])

    while True:
        frames = cams_capture(cams)
        frame = frames['brio']

    # for filepath in photo_paths:
    #     frame = cv2.imread(filepath)

        cursor, cursors, faces = predict_cursor(frame, models)
        print(cursor)
        if cursor is not None:
            avgs = np.roll(avgs, -1, axis=0)
            avgs[-1] = cursor
            avg = avgs.mean(axis=0)
            print(avg)
            pyautogui.moveTo(*cursor_to_pixelxy(avg, monsize), 0.0, pyautogui.easeInOutQuad)

        draw(frame, cursor, cursors, faces)
        # input()

    cams_deinit(cams)


main()
