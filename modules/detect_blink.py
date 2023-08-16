import numpy as np


def detect_blink(face, blink_threshold=0.25):
    left_h = np.linalg.norm(face[386] - face[374])
    left_w = np.linalg.norm(face[362] - face[263])
    left_blink = left_h < blink_threshold * left_w
    right_h = np.linalg.norm(face[145] - face[159])
    right_w = np.linalg.norm(face[133] - face[33])
    right_blink = right_h < blink_threshold * right_w
    return left_blink, right_blink
