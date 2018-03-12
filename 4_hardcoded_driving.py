import numpy as np
import cv2
import time
from numpy import ones,vstack
from PIL import ImageGrab
from numpy.linalg import lstsq
from statistics import mean
from directkeys import PressKey, ReleaseKey, W, A, S, D

training_data = np.load("dataset/training_data_pr_1.npy")

def roi(img, vertices):
    
    #blank mask:
    mask = np.zeros_like(img)   
    
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, 255)
    
    #returning the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked


def draw_lanes(img, lines, color=[220, 80, 0], thickness=3):

    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker 
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []  
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]]
        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = {}

        for idx,i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]

                # Calculating our new, and improved, xs
                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2], lane1_id, lane2_id
    except Exception as e:
        print(str(e))


def process_img(image):
    original_image = image
    # edge detection
    processed_img =  cv2.Canny(image, threshold1 = 50, threshold2=150)
    
    processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
    
    vertices = np.array([[10,200],[170,100],[230,100],[390,200],[390,280],[10,280],
                         ], np.int32)

    processed_img = roi(processed_img, [vertices])

    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #                                     rho   theta   thresh  min length, max gap:        
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 50, 20, 35)
    m1 = 0
    m2 = 0
    try:
        l1, l2, m1, m2 = draw_lanes(original_image,lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [240,20,0], 5)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [240,20,0], 5)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                
                
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return processed_img,original_image, m1, m2

def Forward():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def Left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)

def Right():
    PressKey(W)
    ReleaseKey(A)
    PressKey(D)

def Stop():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

time.sleep(5)
while True:
    #sobelY = cv2.Sobel(data[0], cv2.CV_64F, 0, 1)
    #screen = np.uint8(np.absolute(sobelY))
    #screen = cv2.bilateralFilter(screen,19,75,75)
    screen = np.array(ImageGrab.grab(bbox=(0,40,1000,800)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    screen = cv2.resize(screen,(400,320))
    #time.sleep(0.1)
    new_screen,original_image, m1, m2 = process_img(screen)
    cv2.imshow('window', new_screen)
    cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    if m1<0 and m2<0:
        Right()
        time.sleep(0.15)
        Stop()
        print('W D')
    elif m1>0 and m2>0:
        Left()
        time.sleep(0.15)
        Stop()
        print('W A')
    else:
        Forward()
        time.sleep(0.1)
        print('W')
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

