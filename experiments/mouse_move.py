import pyautogui
import time
import numpy as np

# Set the source and destination coordinates


def calc_step(src, dst, max_offset=3):
    direction = dst - src
    distance = np.linalg.norm(direction)

    if distance == 0:
        return src  # Avoid division by zero

    normalized_direction = direction / distance

    offset_distance = min(max_offset, distance)
    step = src + normalized_direction * offset_distance

    # Ensure step has integer coordinates and doesn't overshoot dst
    step = np.round(step).astype(int)
    # print('st', src, dst, distance, normalized_direction, offset_distance, step)

    return step


monsize = (2560, 1440)
while True:
    target = np.random.randint(low=[0, 0], high=monsize, size=(2))
    while True:
        is_in_target = (target == np.array(pyautogui.position())).all()
        if is_in_target:
            break
        step = calc_step(np.array(pyautogui.position()), target)
        pyautogui.moveTo(*step)
        time.sleep(1 / 120)  # Pause to simulate 60 steps per second

move_smoothly(1000, 1000)
