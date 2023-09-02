from modules.win_tools import wmctrl_r
import pathlib
from modules.gaze_predictor import GazePredictor
import re
import cv2
import mediapipe as mp
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
from modules.interpolate_points import interpolate_points


def draw_cursors(frame, cursor, cursors):
    imsize = np.array([frame.shape[1], frame.shape[0]])
    if cursor is not None:
        cv2.circle(frame, cursor_to_pixelxy(cursor, imsize).astype(int), 4, (255, 0, 0), -1)
        for k, cur in enumerate(cursors):
            col = (255, 0, 0) if k != len(cursors) - 1 else (0, 0, 255)
            cv2.circle(frame, cursor_to_pixelxy(cur, imsize).astype(int), 2, col, -1)

    xy = cursor_to_pixelxy(pixelxy_to_cursor(np.array(mouse_controller.position), monsize, monxy), imsize)
    color_capt = (0, 0, 0)
    cv2.circle(frame, xy.astype(int), 4, (255, 255, 255) if not is_capture else color_capt)
    cv2.circle(frame, xy.astype(int), 3, (0, 255, 0))


def render(frame, cursor, cursors, faces):
    global frame0, frame_ref
    draw_landmarks(frame, faces)
    frame = cv2.flip(frame, 1)
    if frame0 is not None:
        cv2.addWeighted(frame0, 0.3, frame, 0.7, 1, frame)
    elif frame_ref is not None:
        cv2.addWeighted(frame_ref, 0.3, frame, 0.7, 1, frame)

    imsize = np.array([frame.shape[1], frame.shape[0]])
    if faces is not None:
        left_blink, right_blink = detect_blink(faces[0], 0.3)
        cv2.putText(frame, f"{'L' if left_blink else ' '} {'R' if right_blink else ''}",
                    (100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    draw_cursors(frame, cursor, cursors)

    for xy in points[i % len(points): i % len(points) + 80]:
        framexy = cursor_to_pixelxy(pixelxy_to_cursor(xy, monsize, monxy), imsize)
        cv2.circle(frame, np.array(framexy).astype(int), 1, (0, 255, 0))

    for xy, loss in path[-50:]:
        intensity = int(np.clip(loss / 0.02 * 255, 0, 255))
        cv2.circle(frame, cursor_to_pixelxy(xy, imsize).astype(int), 2, (0, 0, intensity))

    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(winname, frame)
    cv2.waitKey(1)


ralt_vk = 65027
fn_vk = 269025067


def on_press(key):
    global cam, i, pos, dirpath, monxy, should_exit
    global is_capture, is_automove
    # if type(key) == pynput.keyboard.KeyCode and key.vk == ralt_vk:
    #     frames = {}
    #     x, y = np.array(pyautogui.position()) - monxy
    #     t0 = time.time()
    #     time.sleep(0.1)
    #     for j in range(3):
    #         ret, frame = cam.read()
    #         filename = f'{dirpath}/{i+1}-{j+1} [{x} {y}] {int(time.time() * 1000)}.jpeg'
    #         frames[filename] = frame
    #     dt = time.time() - t0
    #     print(f'{dt*1000:.0f}')
    #     i += 1
    #     pyautogui.moveTo(*points[i % len(points)])
    #     for filename, frame in frames.items():
    #         os.makedirs(dirpath, exist_ok=True)
    #         cv2.imwrite(filename, frame)
    #         print('save', filename)

    if type(key) == pynput.keyboard.KeyCode and key.vk == ralt_vk:
        is_capture = True
        is_automove = True

    if type(key) == pynput.keyboard.KeyCode and key.vk == fn_vk:
        if len(dataset.datapoints) != 0:
            datapoint = dataset.datapoints.pop()
            path.pop()
            i -= 1
            mouse_controller.postion = cursor_to_pixelxy(
                datapoint['position'], monsize)  # doesn't work from kb_listener thread?
            print('rm', len(dataset.datapoints))

    if key == pynput.keyboard.Key.esc:
        should_exit = True


def on_release(key):
    global is_capture, is_automove
    if type(key) == pynput.keyboard.KeyCode and key.vk == ralt_vk:
        is_capture = False
        is_automove = False


kb_listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
kb_listener.start()


is_capture = False
is_automove = False


def on_click(x, y, button, pressed):
    global is_capture
    if button == pynput.mouse.Button.left:
        is_capture = pressed


mouse_listener = pynput.mouse.Listener(on_click=on_click)
# mouse_listener.start()
mouse_controller = pynput.mouse.Controller()


def capture(face, xy, frame, t0):
    global i, should_exit
    i += 1
    x, y = xy
    filename = f'{dirpath}/{i}.jpg'
    cur = pixelxy_to_cursor(xy, monsize, monxy)
    dataset.add_datapoint(filename, face, cur)
    if save_photos:
        os.makedirs(dirpath, exist_ok=True)
        cv2.imwrite(filename, frame)
        print('save', filename)


save_photos = True

camname = sys.argv[1]
cam = cam_init(camname)
if camname == 'intg':
    monname = 'eDP-1'  # 'eDP-1' (integrated) or 'DP-3' (Dell)
else:
    monname = 'DP-3'
mon = next((mon for mon in get_monitors() if mon.name == monname))
monsize = np.array([mon.width, mon.height])
monxy = np.array([mon.x, mon.y])

winname = 'Capture-xy'

edge_offset = 7
steps = np.array([6, 3])
randomness = 1
edge = np.array([edge_offset, edge_offset, monsize[0]-edge_offset, monsize[1]-edge_offset])
points = spiral(*edge, *steps)
dstep = np.array([edge[2] - edge[0], edge[3] - edge[1]]) / (steps + 1)
r = np.random.randint(-dstep/2, dstep/2, size=[len(points), 2]) * randomness
points = (points + r).clip([0, 0], [monsize[0] - 3, monsize[1] - 4])
points = interpolate_points(points, 10)
fl = np.random.randint(0, 4)
if fl == 1:
    points = np.array(points) * [-1, 1] + [monsize[0], 0]
if fl == 2:
    points = np.array(points) * [1, -1] + [0, monsize[1]]
    points = np.array(points) * [-1, 1] + [monsize[0], 0]
if fl == 3:
    points = np.array(points) * [1, -1] + [0, monsize[1]]
points += monxy
print(monxy, points.min(axis=0), points.max(axis=0))

i = 0
# if len(sys.argv) >= 2:
#     int(sys.argv[1])
mouse_controller.position = points[i % len(points)]

iso_date = datetime.datetime.now().isoformat().split('.')[0].replace(':', '_')
dirpath = f'./data/photos/{iso_date}-{camname}-synth'
dataset = Dataset()

mpaths = sys.argv[2:]
models = [GazePredictor.load_from_file(p) for p in mpaths]
# scores = np.array([float(re.match(r'.* (0.\d+) .*', p)[1]) for p in mpaths])
scores = np.arange(len(models))
models = {model: 1 for model, score in zip(models, scores)}

should_exit = False

path = []

frame_ref = None
# frame_ref = cv2.imread('/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/datasets/intg-static/2023-08-17T12_54_19-intg 979 frame0.jpeg')  # noqa
# frame_ref = cv2.imread('/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/datasets/brio/2023-08-17T15_23_58-brio 1180 frame0.jpeg')  # noqa

frame0 = None
ret, frame = cam.read()
render(frame, None, None, None)
time.sleep(0.1)
wmctrl_r(winname, *monxy)

while not should_exit:
    t0 = time.time()
    xy_true = np.array(mouse_controller.position)
    cur_true = pixelxy_to_cursor(xy_true, monsize, monxy)

    ret, frame = cam.read()
    # print(frame)
    cursor, cursors, faces = predict_cursor(frame, models)
    if cursor is not None:
        cursor = cursor[0]
        cursors = cursors[:, 0]

    loss = ((cursor - cur_true) ** 2).mean() if cursor is not None else 1
    if faces is not None:
        face = faces[0]
        # print(loss)
        if is_capture:
            if frame0 is None:
                frame0 = cv2.flip(frame, 1)
            left_blink, right_blink = detect_blink(face, 0.25)
            if not (left_blink and right_blink):
                capture(face, xy_true, frame, t0)
                path.append([cur_true, loss])

    render(frame, cursor, cursors, faces)

    if is_automove:
        mouse_controller.position = points[i % len(points)]
    dt = time.time() - t0
    # print(dt)


if len(dataset.datapoints) != 0:
    dataset_filepath = f'./data/datasets/{iso_date}-{camname} {len(dataset.datapoints)}.pickle'
    dataset.store(dataset_filepath)

    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    imsize = np.array([frame.shape[1], frame.shape[0]])
    for xy, loss in path:
        intensity = int(np.clip(loss / 0.02 * 255, 0, 255))
        cv2.circle(frame, cursor_to_pixelxy(xy, imsize).astype(int), 2, (0, 0, intensity))
    mean_loss = np.array([l for xy, l in path]).mean()
    im_filepath = dataset_filepath.replace('.pickle', f' {mean_loss:.3f}.jpeg')
    cv2.imwrite(im_filepath, frame)
    cv2.imwrite(dataset_filepath.replace('.pickle', ' frame0.jpeg'), frame0)

mean_cur = np.array([dp['position'] for dp in dataset.datapoints]).mean()
print(f'{mean_cur=}')
mean_loss = np.array([l for xy, l in path]).mean()
print(f'{mean_loss=}')

kb_listener.stop()
mouse_listener.stop()
cv2.destroyAllWindows()
sys.exit()
