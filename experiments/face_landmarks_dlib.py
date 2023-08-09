import dlib
import cv2
import pynput
import glob

# Load the face detection and landmark detection models from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/anatoly/_tot/model/shape_predictor_68_face_landmarks.dat")

# Specify the directories containing the photos
photo_directories = ["/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T16:33:38.163179-3-ok"]

# Initialize lists to store landmarks and photo paths
landmarks_list = []
photo_paths = []

# Collect all photo paths from the specified directories
for directory in photo_directories:
    photo_paths.extend(glob.glob(f"{directory}/*.jpeg"))  # Replace with appropriate pattern

# Function to detect landmarks in a photo and append to the list


def detect_landmarks(photo_path):
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
    img = cv2.imread(str(photo_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    print(len(faces))

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_list.append(landmarks)

        for point in landmarks.parts():
            cv2.circle(img, (point.x, point.y), 1, (0, 255, 0), -1)
        for id in [*LEFT_EYE_POINTS, *RIGHT_EYE_POINTS]:
            point = landmarks.parts()[id]
            cv2.circle(img, (point.x, point.y), 2, (0, 255, 0), -1)

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
    listener.join()

cv2.destroyAllWindows()
