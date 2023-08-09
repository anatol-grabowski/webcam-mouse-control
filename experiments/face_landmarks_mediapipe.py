import dlib
import cv2
import mediapipe as mp
import pynput
import glob
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

face_mesh = mp.solutions.face_mesh.FaceMesh(
    # static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

photo_directories = ["/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T16:33:38.163179-3-ok"]

landmarks_list = []
photo_paths = []

for directory in photo_directories:
    photo_paths.extend(glob.glob(f"{directory}/brio *-1 *.jpeg"))  # Replace with appropriate pattern


def mp_landmarks_to_points(multi_face_landmarks):
    if multi_face_landmarks is None:
        return None
    points_list = [[[p.x, p.y] for p in face.landmark] for face in multi_face_landmarks]
    points = np.array(points_list)
    return points


def mp_detect_faces(rgb, num_warmup=3, num_avg=3):
    for i in range(num_warmup):
        output = face_mesh.process(rgb)
    output = face_mesh.process(rgb)
    if output is None:
        return None
    faces = mp_landmarks_to_points(output.multi_face_landmarks)
    for i in range(num_avg - 1):
        output = face_mesh.process(rgb)
        if output is None:
            return None
        faces += mp_landmarks_to_points(output.multi_face_landmarks)
    faces /= num_avg
    return faces


def detect_landmarks(photo_path):
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    img = cv2.imread(str(photo_path))
    img_h, img_w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mp_detect_faces(rgb)
    print(len(faces) if faces is not None else None)
    if faces is None:
        return img

    for points in faces:
        points = np.multiply(points, [img_w, img_h]).astype(int)
        # print(mesh_points.shape)
        # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
        # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(points[RIGHT_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        cv2.circle(img, center_left, int(l_radius), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(img, center_right, int(r_radius), (0, 255, 0), 1, cv2.LINE_AA)

        for x, y in points:
            # cv2.circle(img, (x, y), 3, (0, 255, 0))
            cv2.circle(img, (x, y), 1, (255, 255, 0), -1)
        cv2.circle(img, center_right, 1, (0, 255, 0), 1)

    return img


current_photo_index = 0


def on_key_press(key):
    global current_photo_index

    try:
        if key == pynput.keyboard.Key.left:
            current_photo_index = max(current_photo_index - 1, 0)
        elif key == pynput.keyboard.Key.right:
            current_photo_index = min(current_photo_index + 1, len(photo_paths) - 1)

        photo_path = photo_paths[current_photo_index]
        img_with_landmarks = detect_landmarks(photo_path)
        cv2.imshow("Photo with Landmarks", img_with_landmarks)
        cv2.waitKey(1)

        if key == pynput.keyboard.Key.esc:
            cv2.destroyAllWindows()
            return False

    except Exception as e:
        print(e)


with pynput.keyboard.Listener(on_press=on_key_press) as listener:
    print('hello')
    listener.join()

cv2.destroyAllWindows()
