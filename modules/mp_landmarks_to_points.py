import numpy as np


def mp_landmarks_to_points(multi_face_landmarks):
    if multi_face_landmarks is None:
        return None
    points_list = [[[p.x, p.y] for p in face.landmark] for face in multi_face_landmarks]
    points = np.array(points_list)
    return points
