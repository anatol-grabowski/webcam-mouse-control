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

face_mesh = mp.solutions.face_mesh.FaceMesh(
    # static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def predict(model, X):
    ''' return torch tensor '''
    model.eval()
    with torch.no_grad():
        y = model(X)
    return y


def predict_ensemble(models, X):
    ''' return torch tensor '''
    ys = []
    for model, weight in models.items():
        prediction = predict(model, X)
        ys.append(prediction)
    ys = torch.stack(ys)
    weights = torch.tensor([w for w in models.values()])
    y = torch.mean(ys * weights.view(-1, 1, 1), dim=0)
    return y, ys


def prepare_X(faces):
    X = faces[:, train_indices].reshape(len(faces), len(train_indices)*2)
    X = torch.tensor(X, dtype=torch.float32)
    return X


def predict_cursor(frame, models):
    '''cursor - contains cursor positions in interval [-1, 1] for each face'''
    rgb = cv2.cvtColor(frame, 1, cv2.COLOR_BGR2RGB)
    faces = mediapipe_detect_faces(face_mesh, rgb, num_warmup=1, num_avg=3)
    if faces is None:
        return None, None, None
    X = prepare_X(faces)
    y, ys = predict_ensemble(models, X)
    cursor = y.numpy()
    cursors = ys.numpy()
    return cursor, cursors, faces


def cursor_to_pixelxy(cursor, monsize):
    xy = (cursor + 1) / 2 * monsize
    return xy


def pixelxy_to_cursor(xy, monsize):
    cursor = xy / monsize * 2 - 1
    return cursor
