import math


def interpolate_points(points, maxstep):
    if len(points) <= 1:
        return points

    interpolated_coords = [points[0]]

    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if distance > maxstep:
            num_interpolations = math.ceil(distance / maxstep)
            dx = (x2 - x1) / num_interpolations
            dy = (y2 - y1) / num_interpolations

            for j in range(1, num_interpolations):
                new_x = x1 + j * dx
                new_y = y1 + j * dy
                interpolated_coords.append([new_x, new_y])

        interpolated_coords.append(points[i])

    return interpolated_coords


# # Example usage
# coordinates = [[0, 0], [3, 4], [6, 8], [9, 12]]
# maxstep = 2
# interpolated = interpolate_points(coordinates, maxstep)
# print(interpolated)
