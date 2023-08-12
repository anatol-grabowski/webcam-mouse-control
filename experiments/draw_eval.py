
import torch
import glob
import sys
import mediapipe as mp
from matplotlib import pyplot as plt

sys.path.append("../modules")
from eye_position_predictor import EyePositionPredictor, train_indices  # noqa
from dataset import Dataset  # noqa


def get_paths(globs):
    glpaths = []
    for gl in globs:
        glpaths.extend(glob.glob(gl))
    return [str(p) for p in glpaths]


photo_globs = [
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T15:57:06.820873-continuous-ok/brio *.jpeg',
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-08T16:33:38.163179-3-ok/brio *-1 *.jpeg',
    # '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/2023-08-09T15:37:18.761700-first-spiral-ok/brio *-1 *.jpeg',
    '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/*/brio *.jpeg',
]
photo_paths = get_paths(photo_globs)


face_mesh = mp.solutions.face_mesh.FaceMesh(
    # static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


model = EyePositionPredictor.load_from_file(sys.argv[1])
X, y = Dataset.read_dataset()
X_tensor = torch.tensor(X, dtype=torch.float32)
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor)

points1 = y.reshape(-1, 2)
points2 = y_pred.numpy().reshape(-1, 2)


plt.figure(figsize=(8, 6))

for p1, p2 in zip(points1, points2):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], marker='o')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Segments between Points')
plt.grid(True)
plt.show()
