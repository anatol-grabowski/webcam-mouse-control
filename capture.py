import pathlib
from modules.gaze_predictor import GazePredictor
import re
import cv2
import mediapipe as mp
import pyautogui
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
from modules.predict_cursor import predict_cursor, cursor_to_pixelxy, pixelxy_to_cursor
from modules.webcam import list_webcams
from modules.detect_blink import detect_blink
from modules.get_paths import get_paths
import keyboard
from modules.webcam import list_webcams, cams_init, cams_capture, cam_init
from modules.dataset import Dataset


def draw_cursors(frame, cursor, cursors):
    imsize = np.array([frame.shape[1], frame.shape[0]])
    if cursor is not None:
        cv2.circle(frame, cursor_to_pixelxy(cursor, imsize).astype(int), 4, (255, 0, 0), -1)
        for k, cur in enumerate(cursors):
            col = (255, 0, 0) if k != len(cursors) - 1 else (0, 0, 255)
            cv2.circle(frame, cursor_to_pixelxy(cur, imsize).astype(int), 2, col, -1)

    xy = cursor_to_pixelxy(pixelxy_to_cursor(np.array(pyautogui.position()), monsize), imsize)
    color_capt = (0, 0, 0)
    cv2.circle(frame, xy.astype(int), 4, (255, 255, 255) if not is_capturing else color_capt)
    cv2.circle(frame, xy.astype(int), 3, (0, 255, 0))


def render(frame, cursor, cursors, faces):
    draw_landmarks(frame, faces)
    frame = cv2.flip(frame, 1)
    imsize = np.array([frame.shape[1], frame.shape[0]])
    if faces is not None:
        left_blink, right_blink = detect_blink(faces[0])
        cv2.putText(frame, f"{'L' if left_blink else ' '} {'R' if right_blink else ''}",
                    (100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    draw_cursors(frame, cursor, cursors)

    for xy, loss in path:
        intensity = int(np.clip(loss * 4 * 255, 20, 255))
        cv2.circle(frame, cursor_to_pixelxy(xy, imsize).astype(int), 2, (0, 0, 255, intensity))

    cv2.namedWindow('Fullscreen Image', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Fullscreen Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Fullscreen Image", frame)
    cv2.waitKey(1)


ralt_vk = 65027
fn_vk = 269025067


def on_press(key):
    global cam, i, pos, dirpath, monxy, should_exit
    if type(key) == pynput.keyboard.KeyCode and key.vk == ralt_vk:
        frames = {}
        x, y = np.array(pyautogui.position()) - monxy
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
            os.makedirs(dirpath, exist_ok=True)
            cv2.imwrite(filename, frame)
            print('save', filename)

    if type(key) == pynput.keyboard.KeyCode and key.vk == fn_vk:
        if len(dataset.datapoints) != 0:
            datapoint = dataset.datapoints.pop()
            path.pop()
            pyautogui.moveTo(*cursor_to_pixelxy(datapoint['position'], monsize))
            print('rm', len(dataset.datapoints))

    if key == pynput.keyboard.Key.esc:
        if len(dataset.datapoints) != 0:
            dataset_filepath = f'./data/datasets/{iso_date}-{camname} {len(dataset.datapoints)}.pickle'
            dataset.store(dataset_filepath)

            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            imsize = np.array([frame.shape[1], frame.shape[0]])
            for xy, loss in path:
                intensity = int(np.clip(loss * 4 * 255, 20, 255))
                cv2.circle(frame, cursor_to_pixelxy(xy, imsize).astype(int), 2, (0, 0, 255, intensity))
            im_filepath = dataset_filepath.replace('.pickle', '.jpeg')
            cv2.imwrite(im_filepath, frame)
        should_exit = True


def on_release(key):
    if type(key) == pynput.keyboard.KeyCode and (key.vk == ralt_vk or key.vk == fn_vk):
        print(f'release')


kb_listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
kb_listener.start()


is_capturing = False


def on_click(x, y, button, pressed):
    global is_capturing
    if button == pynput.mouse.Button.left:
        is_capturing = pressed


mouse_listener = pynput.mouse.Listener(on_click=on_click)
mouse_listener.start()


def capture(face, xy, frame, t0):
    global i
    i += 1
    x, y = xy
    filename = f'{dirpath}/{i} [{x} {y}] {int(t0 * 1000)}.jpeg'
    cur = pixelxy_to_cursor(xy, monsize)
    dataset.add_datapoint(filename, face, cur)
    if save_photos:
        os.makedirs(dirpath, exist_ok=True)
        cv2.imwrite(filename, frame)
        print('save', filename)


save_photos = True

camname = 'intg'
camname = 'brio'
cam = cam_init(camname)
if camname == 'intg':
    monname = 'eDP-1'  # 'eDP-1' (integrated) or 'DP-3' (Dell)
else:
    monname = 'DP-3'
mon = next((mon for mon in get_monitors() if mon.name == monname))
monsize = np.array([mon.width, mon.height])
monxy = np.array([mon.x, mon.y])

pyautogui.FAILSAFE = False
edge_offset = 5
steps = np.array([8, 5])
edge = np.array([edge_offset, edge_offset, monsize[0]-edge_offset, monsize[1]-edge_offset])
points = spiral(*edge, *steps)
dstep = np.array([edge[2] - edge[0], edge[3] - edge[1]]) / (steps + 1)
randomness = 1
r = np.random.randint(-dstep/2, dstep/2, size=[len(points), 2]) * randomness
points = (points + r).clip([0, 0], [monsize[0] - 3, monsize[1] - 4])
points += monxy
print(points.max(axis=0))

i = 0
# if len(sys.argv) >= 2:
#     int(sys.argv[1])
pyautogui.moveTo(*points[i % len(points)], 0.2, pyautogui.easeInOutQuad)

iso_date = datetime.datetime.now().isoformat().split('.')[0].replace(':', '_')
dirpath = f'./data/photos/{iso_date}-{camname}'
dataset = Dataset()

mpaths = sys.argv[1:]
models = [GazePredictor.load_from_file(p) for p in mpaths]
# scores = np.array([float(re.match(r'.* (0.\d+) .*', p)[1]) for p in mpaths])
scores = np.arange(len(models))
models = {model: 1 for model, score in zip(models, scores)}

should_exit = False

path = []

while True:
    if should_exit:
        kb_listener.stop()
        mouse_listener.stop()
        cv2.destroyAllWindows()
        sys.exit()

    t0 = time.time()
    ret, frame = cam.read()
    dt = time.time() - t0
    # print(dt)

    cursor, cursors, faces = predict_cursor(frame, models)
    if cursor is not None:
        cursor = cursor[0]
        cursors = cursors[:, 0]

    if is_capturing:
        xy = np.array(pyautogui.position()) - monxy
        capture(faces[0], xy, frame, t0)
        real_cur = pixelxy_to_cursor(xy, monsize)
        loss = ((cursor - real_cur) ** 2).mean() if cursor is not None else 0.3
        path.append([real_cur, loss])

    render(frame, cursor, cursors, faces)
