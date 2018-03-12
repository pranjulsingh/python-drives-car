# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time

training_data = np.load("dataset/training_data_pr_4.npy")
lower = np.array([60,60,60], dtype = "uint8")
upper = np.array([150,150,150], dtype = "uint8")

#img = training_data[10][0]
#ak = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.bilateralFilter(img,29,115,115)
#cv2.imshow('Original',img)
for data in training_data:
    #img = cv2.bilateralFilter(data[0],19,75,75)
    #mask = cv2.inRange(img, lower, upper)
    #output = cv2.bitwise_and(img, img, mask = mask)
    sobelY = cv2.Sobel(data[0], cv2.CV_64F, 0, 1)
    sobelY = np.uint8(np.absolute(sobelY))
    cv2.imshow('video',sobelY)
    print(data[1])
    time.sleep(0.1)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

