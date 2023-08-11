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


landmarks_list = []
photo_paths = []

photo_paths.extend(glob.glob(f"/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/*/brio *-1 *.jpeg"))


top_grad_indices = [334, 471, 468, 293, 55, 109, 296, 472, 10, 282, 65, 336, 470, 107, 469, 66, 295, 338, 67, 300, 276, 283, 285, 103, 105, 52, 243, 476, 63, 297, 53, 221, 193, 189, 477, 372, 473, 46, 54, 133, 383, 353, 413, 245, 365, 244, 475, 337, 108, 332, 151, 397, 233, 417, 21, 342, 247, 466, 226, 288, 398, 435, 367, 379, 190, 130, 186, 364, 113, 441, 287, 124, 112, 388, 57, 70, 68, 463, 359, 225, 467, 212, 361, 435, 104, 445, 410, 340, 474, 69, 183, 255, 222, 322, 446, 299, 165, 155, 54, 442, 265, 92, 185, 263, 21, 216, 76, 341, 359, 387, 82, 362, 391, 401, 224, 176, 362, 128, 273, 433, 261, 422, 358, 43, 414, 414, 103, 40, 104, 465, 367, 346, 464, 257, 436, 78, 290, 384, 455, 355, 272, 184, 249, 382, 191, 284, 331, 81, 434, 370, 62, 33, 269, 397, 413, 444, 378, 149, 25, 169, 67, 443, 333, 463, 365, 249, 71, 430, 262, 135, 376, 80, 398, 206, 202, 400, 148, 372, 214, 258, 407, 432, 162, 394, 35, 223, 431, 61, 394, 347, 288, 279, 434, 326, 315, 401, 289, 415, 167, 386, 68, 97, 39, 426, 136, 69, 247, 393, 395, 357, 24, 364, 150, 416, 366, 402, 448, 328, 360, 286, 98, 406, 134, 75, 268, 460, 467, 452, 326, 127, 396, 141, 228, 207, 23, 328, 439, 216, 152, 226, 128, 95, 74, 72, 327, 232, 305, 73, 255, 172, 446, 144, 59, 65, 341, 107, 225, 100, 211, 66, 26, 164, 331, 403, 196, 339, 52, 457, 231, 129, 411, 204, 63, 239, 423, 235, 263, 210, 259, 83, 9, 424, 416, 461, 25, 113, 361, 77, 146, 260, 143, 460, 408, 312, 187, 420, 41, 378, 230, 88, 19, 70, 391, 156, 31, 369, 121, 440, 164, 110, 87, 8, 358, 323, 86, 162, 314, 197, 152, 89, 302, 399, 175, 168, 424, 385, 363, 223, 377, 0, 142, 351, 7, 310, 43, 408, 423, 12, 213, 105, 20, 303, 38, 185, 48, 110, 409, 22, 366, 13, 26, 138, 245, 462, 47, 441, 319, 377, 324, 170, 4, 96, 60, 102, 405, 120, 456, 71, 447, 91, 242, 50, 235, 354, 17, 458, 335, 20, 281, 14, 309, 221, 148, 360, 163, 122, 392, 327, 317, 356, 352, 275, 430, 142, 411, 265, 442, 121, 154, 89, 166, 438, 42, 375, 279, 307, 240, 459, 182, 440, 464, 219, 90, 156, 312, 85, 14, 146, 313, 192, 145, 352, 32, 381, 458, 12, 278, 329, 306, 99, 46, 427, 422, 60, 77, 222, 56, 205, 108, 86, 396, 141, 133, 294, 292, 289, 291, 318, 415, 453, 406, 429, 37, 119, 124, 181, 274, 455, 130, 217, 465, 238, 2, 379, 115, 84, 200, 237, 232, 339, 439, 139, 51, 45, 433, 317,
                    308, 189, 291, 285, 219, 251, 64, 370, 106, 412, 44, 280, 45, 30, 198, 53, 147, 2, 218, 244, 304, 242, 256, 266, 404, 153, 272, 445, 240, 173, 278, 174, 237, 321, 175, 393, 402, 325, 178, 38, 371, 250, 74, 234, 181, 431, 395, 447, 290, 294, 298, 94, 58, 40, 215, 248, 318, 344, 218, 27, 311, 271, 241, 203, 316, 292, 186, 427, 363, 400, 320, 109, 55, 102, 179, 371, 200, 1, 250, 302, 199, 228, 16, 13, 437, 1, 5, 307, 79, 101, 118, 97, 337, 220, 194, 324, 140, 207, 209, 419, 437, 432, 323, 57, 304, 30, 316, 56, 322, 72, 194, 11, 243, 106, 167, 5, 36, 187, 457, 403, 59, 127, 11, 94, 201, 353, 267, 155, 273, 344, 454, 390, 99, 48, 39, 180, 380, 137, 267, 87, 35, 3, 92, 281, 125, 287, 325, 171, 15, 180, 306, 117, 354, 49, 462, 177, 208, 436, 7, 320, 6, 199, 29, 115, 425, 277, 392, 147, 205, 345, 64, 229, 182, 438, 173, 343, 236, 425, 409, 90, 195, 134, 461, 114, 61, 206, 209, 271, 93, 91, 426, 453, 286, 211, 126, 174, 376, 383, 188, 111, 28, 224, 129, 305, 212, 443, 417, 444, 36, 139, 311, 274, 421, 241, 266, 269, 410, 117, 448, 179, 151, 15, 429, 330, 76, 95, 342, 193, 336, 239, 120, 270, 44, 75, 82, 373, 374, 418, 170, 350, 132, 143, 123, 190, 131, 268, 309, 157, 50, 184, 335, 449, 303, 98, 166, 375, 73, 85, 233, 178, 369, 158, 418, 321, 407, 118, 119, 404, 31, 195, 140, 112, 421, 295, 198, 80, 343, 405, 234, 62, 49, 83, 101, 310, 42, 382, 351, 260, 37, 428, 345, 126, 123, 262, 428, 16, 131, 451, 280, 171, 96, 32, 238, 183, 165, 213, 33, 19, 159, 204, 79, 338, 349, 246, 18, 275, 314, 355, 34, 454, 125, 300, 350, 191, 10, 34, 161, 163, 399, 111, 41, 340, 160, 29, 420, 236, 459, 208, 201, 466, 203, 253, 330, 176, 456, 84, 116, 227, 17, 270, 264, 348, 357, 229, 210, 261, 308, 6, 368, 202, 3, 315, 452, 252, 227, 264, 319, 356, 116, 4, 192, 81, 168, 196, 217, 313, 296, 248, 412, 88, 254, 259, 390, 220, 100, 389, 47, 197, 277, 0, 449, 149, 246, 78, 258, 18, 283, 329, 150, 132, 136, 334, 122, 301, 138, 215, 114, 254, 299, 169, 231, 297, 368, 293, 214, 51, 154, 28, 419, 22, 276, 93, 188, 230, 172, 24, 27, 9, 349, 256, 282, 450, 135, 137, 58, 177, 450, 346, 333, 301, 347, 389, 384, 348, 257, 144, 23, 373, 8, 451, 153, 381, 161, 298, 388, 251, 284, 332, 253, 252, 157, 380, 145, 374, 385, 160, 387, 386, 158, 159, 469, 468, 470, 472, 471, 475, 474, 473, 476, 477]
top_grad_indices = np.array(top_grad_indices)


train_indices = [
    21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251,  # forehead
    108, 151, 337,  # forehead lower
    143, 156, 70, 63, 105, 66, 107,  # brow right outer
    336, 296, 334, 293, 300, 383, 372,  # brow left outer
    124, 46, 53, 52, 65, 55, 193,  # brow right middle
    285, 295, 282, 283, 276, 353, 417,  # brow left middle
    226, 247, 246, 221,  # around right eye
    446, 467, 466, 441,  # around left eye
    189, 190, 173, 133, 243, 244, 245, 233,  # right z
    413, 414, 398, 362, 463, 464, 465, 153,  # left z
    58, 172, 136, 150,  # right cheek
    288, 397, 365, 379,  # left cheek
    468, 469, 470, 471, 472,  # right iris
    473, 474, 475, 476, 477,  # left iris
]
train_indices = np.array(train_indices)
print(len(train_indices))


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
        for x, y in points[train_indices]:
            # cv2.circle(img, (x, y), 3, (0, 255, 0))
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

        cv2.circle(img, points[p_num], 3, (0, 255, 0), -1)
        importance = np.where(top_grad_indices == p_num)[0]
        cv2.putText(img, f'#{p_num}, {importance}', points[p_num], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=2)

    return img


p_num = 0
current_photo_index = 0


def on_key_press(key):
    global current_photo_index, p_num

    try:
        if key == pynput.keyboard.Key.left:
            current_photo_index = max(current_photo_index - 1, 0)
        elif key == pynput.keyboard.Key.right:
            current_photo_index = min(current_photo_index + 1, len(photo_paths) - 1)
        if key == pynput.keyboard.Key.up:
            p_num += 1
        if key == pynput.keyboard.Key.down:
            p_num -= 1

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
