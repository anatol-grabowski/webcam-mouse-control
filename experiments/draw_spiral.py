# import pyautogui
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from modules.spiral import spiral  # noqa
from modules.interpolate_points import interpolate_points  # noqa

# Example usage
edge_offset = 10
monsize = np.array([2560, 1440])
steps = np.array([8, 5])
edge = np.array([edge_offset, edge_offset, monsize[0]-edge_offset, monsize[1]-edge_offset])
points = spiral(*edge, *steps)
dstep = np.array([edge[2] - edge[0], edge[3] - edge[1]]) / (steps + 1)
r = np.random.randint(-dstep/2, dstep/2, size=[len(points), 2])
points = (points + r).clip([0, 0], monsize - 2)
points = points * [-1, 1] + [monsize[0], 0]
points = np.array(interpolate_points(points, 50))
print(points.max(axis=0))
# for x, y in points:
#     pyautogui.moveTo(x, y)
#     time.sleep(0.01)

plt.plot(points[:, 0], points[:, 1], marker='o')
plt.show()
