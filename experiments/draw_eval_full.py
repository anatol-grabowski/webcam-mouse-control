from tqdm import tqdm
import re
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from modules.gaze_predictor import GazePredictor, train_indices  # noqa
from modules.mediapipe_detect_faces import mediapipe_detect_faces  # noqa
from modules.predict_cursor import predict_cursor, predict_ensemble, prepare_X, cursor_to_pixelxy, pixelxy_to_cursor  # noqa
from modules.webcam import list_webcams  # noqa
from modules.detect_blink import detect_blink  # noqa
from modules.get_paths import get_paths, get_xy_from_filename  # noqa
from modules.webcam import list_webcams, cams_init, cams_capture  # noqa
from modules.draw_landmarks import draw_landmarks  # noqa
from modules.mp_landmarks_to_points import mp_landmarks_to_points  # noqa
from modules.dataset import Dataset, dataset_filepath  # noqa


photo_globs = [
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T15:57:06.820873-continuous-ok/brio *.jpeg',
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T16:33:38.163179-3-ok/brio *-1 *.jpeg',
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-09T15:37:18.761700-first-spiral-ok/brio *-1 *.jpeg',
    '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/*/brio *.jpeg',
]
photo_paths = get_paths(photo_globs)

directory_boundaries = []
previous_directory = ''
for i, path in enumerate(photo_paths):
    current_directory = '/'.join(path.split('/')[:-1])  # Extract the directory part of the path
    if current_directory != previous_directory:
        directory_boundaries.append(i)
        previous_directory = current_directory


def on_press(key):
    global i, directory_boundaries, photo_paths
    if key == pynput.keyboard.Key.esc:
        sys.exit()

    if key == pynput.keyboard.Key.right:
        i += 1
        i = i % len(photo_paths)
        draw_eval()
    if key == pynput.keyboard.Key.left:
        i -= 1
        i = i % len(photo_paths)
        draw_eval()
    if key == pynput.keyboard.Key.page_down:
        next_boundaries = [boundary for boundary in directory_boundaries if boundary > i]
        i = next_boundaries[0] if len(next_boundaries) != 0 else 0
        i = i % len(photo_paths)
        draw_eval()
    if key == pynput.keyboard.Key.page_up:
        prev_boundaries = [boundary for boundary in directory_boundaries[::-1] if boundary < i]
        i = prev_boundaries[0] if len(prev_boundaries) != 0 else directory_boundaries[-1]
        i = i % len(photo_paths)
        draw_eval()


mpaths = sys.argv[1:]
models = [GazePredictor.load_from_file(p) for p in mpaths]
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

numavg = 1
avgs = np.zeros(shape=(numavg, 2))

i = 0

monsize = np.array([2560, 1440])

b = None
points = []

with open(dataset_filepath, 'rb') as file:
    dataset_list = pickle.load(file)

xs = []
for j in tqdm(range(0, len(dataset_list))):
    dp = dataset_list[j]
    face = dp['landmarks']
    x = prepare_X(face.reshape(1, *face.shape))
    xs.append(x[0])

X = torch.stack(xs)
y, *_ = predict_ensemble(models, X)
xy = dp['cursor_norm']
points = [[xy, cur.numpy()] for cur in y]


def draw_eval():
    global points, b

    frame = cv2.imread(photo_paths[i])
    imsize = np.array([frame.shape[1], frame.shape[0]])
    for xy, cur in points:
        xy = cursor_to_pixelxy(xy, imsize).astype(int)
        cv2.circle(frame, xy, 2, (0, 255, 0), -1)
        if cur is not None:
            cur = cursor_to_pixelxy(cur[0], imsize).astype(int)
            cv2.circle(frame, cur, 2, (255, 0, 0), -1)
            cv2.line(frame, cur, xy, (255, 0, 0), 1)

    cv2.namedWindow('Fullscreen Image', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Fullscreen Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Fullscreen Image", frame)
    cv2.waitKey(1)


# draw_eval()
print('ready')
with pynput.keyboard.Listener(on_press=on_press) as listener:
    listener.join()

cv2.destroyAllWindows()


def main():
    global avgs, numavg
    print('hello')

    monsize = np.array([2560, 1440])

    for filepath in photo_paths:
        frame = cv2.imread(filepath)
        xy = pixelxy_to_cursor(get_xy_from_filename(filepath), monsize)

        cursor, cursors, faces = predict_cursor(frame, models)
        cursor = cursor.reshape(2)
        cursors = cursors.reshape(-1, 2)
        if cursor is not None:
            avgs = np.roll(avgs, -1, axis=0)
            avgs[-1] = cursor
            avg = avgs.mean(axis=0)
            print('cursor', avg)

        render(frame, cursor, cursors, faces, xy)
        input()


# main()
