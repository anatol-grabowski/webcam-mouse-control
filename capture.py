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
from modules.list_webcams import list_webcams
from modules.spiral import spiral


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
    # having multiple cams enabled slows down and delays capture

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


def on_press(key):
    global cams, iso_date, i, pos
    if key == pynput.keyboard.Key.enter:
        frames = {}
        x, y = pyautogui.position()
        t0 = time.time()
        time.sleep(0.1)
        for camname, cam in cams.items():
            for j in range(3):
                ret, frame = cam.read()
                filename = f'./data/{iso_date}/{camname} {i}-{j} [{x} {y}] {int(time.time() * 1000)}.jpeg'
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

cams = cams_init()
iso_date = datetime.datetime.now().isoformat()
os.mkdir(f'./data/{iso_date}')

edge_offset = 10
points = spiral(edge_offset, edge_offset, 2560-edge_offset, 1440-edge_offset, 15, 9)
i = 0
pyautogui.moveTo(*points[i % len(points)])

while True:
    for camname, cam in cams.items():
        t0 = time.time()
        ret, frame = cam.read()
        dt = time.time() - t0
        if camname == 'brio':
            cv2.imshow('cam', frame)
        # print(dt)
    cv2.waitKey(1)
