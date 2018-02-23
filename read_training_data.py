# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time

training_data = np.load("dataset/training_data.npy")
for data in training_data:
    time.sleep(0.1)
    #cv2.imshow("Training Data",training_data[i][0])
    cv2.imshow('video',data[0])
    print(data[1])
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

print(len(training_data))
print("Data End")
