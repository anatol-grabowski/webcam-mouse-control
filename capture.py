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


def track():
    # cam = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    screen_w, screen_h = pyautogui.size()

    while True:
        _, frame = cam2.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape
        if landmark_points:
            landmarks = landmark_points[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))
                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    # pyautogui.moveTo(screen_x, screen_y)
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))
            # if (left[0].y - left[1].y) < 0.004:
            #     pyautogui.click()

            #     pyautogui.sleep(1)
        cv2.imshow('Eye Controlled Mouse', frame)
        cv2.waitKey(1)


def on_click(x, y, button, pressed):
    should_handle = pressed and y < 1439
    # if not should_handle:
    # return

    print("Mouse clicked", x, y, button, pressed)
    # _, frame1 = cam1.read()
    # _, frame2 = cam2.read()
    # print(frame1.empty())
    # cv2.imwrite(f'./data/{time.time()} re {x} {y}.jpg', frame1)
    # cv2.imwrite(f'./data/{time.time()} ir {x} {y}.jpg', frame2)


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

    print('cam3')
    cam3 = cv2.VideoCapture(intcams[0])
    cam3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam3.set(cv2.CAP_PROP_FPS, 30)
    camsdict['integrated'] = cam3

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


frames = []


def cams_capture(cams, iso_date, pos):
    t0_iter = time.time()
    t = time.time()
    for i in range(3):
        for camname, cam in cams.items():
            x, y = pos
            # print(x, y)
            t0 = time.time()
            ret, frame = cam.read()
            dt = time.time() - t0
            filename = f'./data/{iso_date}/{camname} {(t * 1000):.0f}-{i+1} [{x} {y}] {dt*1000:.0f}.jpeg'
            cv2.imwrite(filename, frame)
    dt_iter = time.time() - t0_iter
    print(dt_iter)


def spiral(xmin, ymin, xmax, ymax, xsteps, ysteps):
    points_list = []
    x, y = 0, 0
    num_points = (xsteps + 1) * (ysteps + 1)
    dir = 'right'
    x0, y0, x1, y1 = 0, 0, xsteps, ysteps

    while len(points_list) < num_points:
        points_list.append([x, y])
        if dir == 'right':
            x += 1
            if x == x1:
                y0 += 1
                dir = 'down'
            continue
        if dir == 'down':
            y += 1
            if y == y1:
                x1 -= 1
                dir = 'left'
            continue
        if dir == 'left':
            x -= 1
            if x == x0:
                y1 -= 1
                dir = 'up'
            continue
        if dir == 'up':
            y -= 1
            if y == y0:
                x0 += 1
                dir = 'right'
            continue

    dx = (xmax - xmin) / xsteps
    dy = (ymax - ymin) / ysteps
    points = np.array([xmin, ymin]) + np.array(points_list) * np.array([dx, dy])
    return points


def main():
    print('hello')

    iso_date = datetime.datetime.now().isoformat()
    os.mkdir(f'./data/{iso_date}')

    cams = cams_init()

    edge_offset = 5
    points = spiral(edge_offset, edge_offset, 2560-edge_offset, 1440-edge_offset, 16, 10)
    i = 0

    while True:
        step = points[i % len(points)]
        i += 1
        pyautogui.moveTo(*step)
        input()
        cams_capture(cams, iso_date=iso_date, pos=pyautogui.position())

    cams_deinit(cams)


main()
