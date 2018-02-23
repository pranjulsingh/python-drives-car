# -*- coding: utf-8 -*-
from keyboard_io import key_check
from PIL import ImageGrab
import numpy as np
import time
import cv2

training_data = []
while True:
    time.sleep(0.1)
    key = key_check()
    screen = np.array(ImageGrab.grab(bbox=(0,40,1000,800)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen,(400,320))
    training_data.append([screen, key])
    if len(training_data)==1000:
        np.save("dataset/training_data.npy", training_data)
        break

print("Dataset Saved")
