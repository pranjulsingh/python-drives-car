# -*- coding: utf-8 -*-
from keyboard_io import key_check
from PIL import ImageGrab
import numpy as np
import time
import cv2

training_data = []
k=0
time.sleep(10)
print("start recording")
while True:
    time.sleep(0.2)
    key = key_check()
    screen = np.array(ImageGrab.grab(bbox=(0,40,1000,800)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    screen = cv2.resize(screen,(400,320))
    training_data.append([screen, key])
    if len(training_data)%200==0:
        k+=1
        print(str(len(training_data)*k)+' frames captured')
        np.save('dataset/training_data_pr_'+str(k)+'.npy', training_data)
        print('Saved File: dataset/training_data_pr'+str(k)+'.npy')
        training_data = []
    if (len(training_data)*k)==100000:
        break

print("Dataset Saved")
