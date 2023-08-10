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
import sys

sys.path.append("../modules")
from list_webcams import list_webcams  # noqa


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

    return camsdict


def on_press(key):
    if key == pynput.keyboard.Key.enter:
        frames = {}
        t0 = time.time()
        for camname, cam in cams.items():
            for i in range(3):
                ret, frame = cam.read()
                filename = f'./data/{iso_date}/{camname} {t0*1000:.0f} {i}.jpeg'
                frames[filename] = frame
        dt = time.time() - t0
        print(dt)
        for filename, frame in frames.items():
            cv2.imwrite(filename, frame)
            print('save', filename)


kb_listener = pynput.keyboard.Listener(on_press=on_press)
kb_listener.start()


cams = cams_init()
cam = cams['brio']
iso_date = datetime.datetime.now().isoformat()
os.mkdir(f'./data/{iso_date}')
i = 0
while True:
    if i == 1:
        print('ready')
    for camname, cam in cams.items():
        t0 = time.time()
        ret, frame = cam.read()
        dt = time.time() - t0
        if camname == 'brio':
            cv2.imshow('cam', frame)
        print(dt)
    cv2.waitKey(1)
    i += 1
