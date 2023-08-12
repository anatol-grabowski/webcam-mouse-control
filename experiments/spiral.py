# import pyautogui
import sys
import time
import matplotlib.pyplot as plt
import numpy as np


def spiral(xmin, ymin, xmax, ymax, xsteps, ysteps):
    points_list = []
    x, y = 0, 0
    num_points = (xsteps + 1) * (ysteps + 1)
    dir = 'right'
    x0, y0, x1, y1 = 0, 0, xsteps, ysteps

    while len(points_list) < num_points:
        points_list.append([x, y])
        if dir == 'right':
            x += 1
            if x == x1:
                y0 += 1
                dir = 'down'
            continue
        if dir == 'down':
            y += 1
            if y == y1:
                x1 -= 1
                dir = 'left'
            continue
        if dir == 'left':
            x -= 1
            if x == x0:
                y1 -= 1
                dir = 'up'
            continue
        if dir == 'up':
            y -= 1
            if y == y0:
                x0 += 1
                dir = 'right'
            continue

    dx = (xmax - xmin) / xsteps
    dy = (ymax - ymin) / ysteps
    points = np.array([xmin, ymin]) + np.array(points_list) * np.array([dx, dy])
    return points


# Example usage
edge_offset = 10
monsize = np.array([2560, 1440])
steps = np.array([8, 5])
edge = np.array([edge_offset, edge_offset, monsize[0]-edge_offset, monsize[1]-edge_offset])
points = spiral(*edge, *steps)
dstep = np.array([edge[2] - edge[0], edge[3] - edge[1]]) / (steps + 1)
r = np.random.randint(-dstep/2, dstep/2, size=[len(points), 2])
points = (points + r).clip([0, 0], monsize - 2)
print(points.max(axis=0))
# for x, y in points:
#     pyautogui.moveTo(x, y)
#     time.sleep(0.01)

plt.plot(points[:, 0], points[:, 1], marker='o')
plt.show()
