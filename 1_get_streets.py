# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time

training_data = np.load("dataset/training_data_pr_1.npy")

#img = training_data[10][0]
#ak = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.bilateralFilter(img,29,115,115)
#cv2.imshow('Original',img)
for data in training_data:
    img = data[0]
    img1=np.zeros((320,400))
    for w in range(320):
        for h in range(400):
            if not((abs(img[w][h][0]-img[w][h][1])>210) and (abs(img[w][h][0]-img[w][h][2])>210) and (abs(img[w][h][2]-img[w][h][1])>210)):
                img1[w][h] = 255
    cv2.imshow('video',img1)
    print(data[1])
    time.sleep(0.2)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

