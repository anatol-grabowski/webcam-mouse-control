import time
import numpy as np
import pygame
import pygame.camera
import cv2
import sys
sys.path.append("../modules")
from list_webcams import list_webcams  # noqa


webcams = list_webcams()
intcams = webcams[[cam for cam in webcams.keys() if 'Integrated' in cam][0]]
briocams = webcams[[cam for cam in webcams.keys() if 'BRIO' in cam][0]]

pygame.camera.init()
print(briocams)
cam1 = pygame.camera.Camera(briocams[0], (1280, 720))
cam1.start()

# cam2 = pygame.camera.Camera(intcams[0], (1280, 720))
# cam2.start()

while True:
    # img = cam2.get_image()
    t0 = time.time()
    img = cam1.get_image()
    pygame_image_string = pygame.image.tostring(img, 'RGB')
    cv2_image_array = np.frombuffer(pygame_image_string, dtype=np.uint8)
    cv2_image_array = cv2_image_array.reshape((img.get_height(), img.get_width(), 3))
    bgr = cv2.cvtColor(cv2_image_array, cv2.COLOR_RGB2BGR)
    dt = time.time() - t0
    print(dt)
    cv2.imshow('cam', bgr)
    cv2.waitKey(1)


pygame.image.save(img, "filename.jpg")
