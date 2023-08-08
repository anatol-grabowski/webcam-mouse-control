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

    completed_process = subprocess.run('v4l2-ctl --list-devices 2>/dev/null',
                                       shell=True, stdout=subprocess.PIPE, text=True)

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


webcams = list_webcams()
briocams = webcams[[cam for cam in webcams.keys() if 'BRIO' in cam][0]]
print('cam2')
cam2 = cv2.VideoCapture(briocams[2])
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 340)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 340)
cam2.set(cv2.CAP_PROP_FPS, 30)


buff = np.zeros(10, dtype=np.int8)
i = 0
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

while True:
    ret, frame = cam2.read()
    if not ret:
        continue
    buff = np.roll(buff, -1)
    buff[-1] = frame.mean()
    buffmean = buff.mean()
    print('ret', ret, buff[-1], buffmean)
    i += 1
    if i < len(buff) or buff[-1] < buffmean:
        continue
    frame = cv2.flip(frame, 1)
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        # if (left[0].y - left[1].y) < 0.004:
        #     pyautogui.click()

        #     pyautogui.sleep(1)
    cv2.imshow('Eye Controlled Mouse', frame)

    # cv2.imshow('Eye Controlled Mouse', frame)
    # cv2.imwrite(f'./data/{time.time()}.jpeg', frame)
    cv2.waitKey(1)
