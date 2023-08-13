from modules.gaze_predictor import GazePredictor
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
from modules.webcam import list_webcams, cam_init
from modules.spiral import spiral
import sys
from screeninfo import get_monitors
from modules.gaze_predictor import GazePredictor, train_indices
from modules.mediapipe_detect_faces import mediapipe_detect_faces
from modules.draw_landmarks import draw_landmarks
from modules.predict_cursor import predict_cursor, cursor_to_pixelxy
from modules.webcam import list_webcams
from modules.detect_blink import detect_blink
from modules.get_paths import get_paths
from modules.webcam import list_webcams, cams_init, cams_capture, cam_init


def draw_cursors(frame, cursor, cursors):
    imsize = np.array([frame.shape[1], frame.shape[0]])
    cv2.circle(frame, cursor_to_pixelxy(cursor, imsize).astype(int), 4, (255, 0, 0), -1)
    for cur in cursors:
        cv2.circle(frame, cursor_to_pixelxy(cur, imsize).astype(int), 2, (255, 0, 0), -1)


def render(frame, cursor, cursors, faces):
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


def on_press(key):
    global cam, iso_date, i, pos, dirpath, monxy
    if key == pynput.keyboard.Key.enter:
        frames = {}
        x, y = pyautogui.position()
        x, y = x - monxy[0], y - monxy[1]
        t0 = time.time()
        time.sleep(0.1)
        for j in range(3):
            ret, frame = cam.read()
            filename = f'{dirpath}/{i+1}-{j+1} [{x} {y}] {int(time.time() * 1000)}.jpeg'
            frames[filename] = frame
        dt = time.time() - t0
        print(f'{dt*1000:.0f}')
        i += 1
        pyautogui.moveTo(*points[i % len(points)])
        for filename, frame in frames.items():
            cv2.imwrite(filename, frame)
            print('save', filename)


kb_listener = pynput.keyboard.Listener(on_press=on_press)
kb_listener.start()

camname = 'intg'
cam = cam_init(camname)
iso_date = datetime.datetime.now().isoformat()

monname = 'eDP-1'  # 'eDP-1' (integrated) or 'DP-3' (Dell)
mon = next((mon for mon in get_monitors() if mon.name == monname))
monsize = np.array([mon.width, mon.height])
monxy = np.array([mon.x, mon.y])
pyautogui.FAILSAFE = False
edge_offset = 5
steps = np.array([8, 5])
edge = np.array([edge_offset, edge_offset, monsize[0]-edge_offset, monsize[1]-edge_offset])
points = spiral(*edge, *steps)
dstep = np.array([edge[2] - edge[0], edge[3] - edge[1]]) / (steps + 1)
randomness = 0
r = np.random.randint(-dstep/2, dstep/2, size=[len(points), 2]) * randomness
points = (points + r).clip([0, 0], [monsize[0] - 3, monsize[1] - 4])
points += monxy
print(points.max(axis=0))

i = 0 if len(sys.argv) < 2 else int(sys.argv[1])
pyautogui.moveTo(*points[i % len(points)], 0.2, pyautogui.easeInOutQuad)

dirpath = f'./data/{iso_date}-{camname}-{steps[0]}x{steps[1]}{"r" if randomness != 0 else ""}'
os.mkdir(dirpath)

mpaths = sys.argv[2:]
models = [GazePredictor.load_from_file(p) for p in mpaths]
scores = np.array([float(re.match(r'.* (0.\d+) .*', p)[1]) for p in mpaths])
models = {model: 1 for model, score in zip(models, scores)}

while True:
    t0 = time.time()
    ret, frame = cam.read()
    dt = time.time() - t0
    # print(dt)

    if len(models) != 0:
        cursor, cursors, faces = predict_cursor(frame, models)
        if cursor is not None:
            cursor = cursor.reshape(2)
            cursors = cursors.reshape(-1, 2)
        render(frame, cursor, cursors, faces)
    else:
        cv2.namedWindow('Fullscreen Image', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Fullscreen Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Fullscreen Image", frame)
        cv2.waitKey(1)
