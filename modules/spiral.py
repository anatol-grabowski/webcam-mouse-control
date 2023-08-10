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
