# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt

def filterLargeBoxes(boxArr):
    arr = boxArr[:]
    for i in range(len(arr)-1, -1, -1):
        x1, y1, w1, h1 = arr[i]
        for j in range(len(arr)-1, -1, -1):
            x2, y2, w2, h2 = arr[j]
            if (x2 < x1 and y2 < y1 and x2+w2 > x1+w1 and y2+h2 > y1+h1):
                arr = np.delete(arr, j, 0)
    return arr

def getY(line, x):
    if line is not None:
        rho, theta = line[0]
        slope = np.tan(theta-np.pi/2)
        x0 = np.cos(theta)*rho
        y0 = np.sin(theta)*rho
        b = y0 - x0 * slope
        return slope*x + b
    return 0

# remove extraneous lines return tuple of (horizontal line, vertical line)
def trim_lines(lines, avg_vert, avg_horiz):
    # y0 (perp line from origin) - x0 * slope
    # slope = float(y2 - y1) / float(x2 - x1)
    vert_lines = []
    horiz_lines = []
    # separate into horizontal and vertical lines
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            slope = float(y2-y1)/float(x2-x1)

            if abs(slope) < .1:
                horiz_lines.append(line)
            elif abs(slope) > .5:
                vert_lines.append(line)

    # grab the most extreme vertical line and highest horizontal line
    horiz_lines.sort(key=lambda line: get_y_intercept(line))

    horiz = horiz_lines[0] if len(horiz_lines) > 0 else avg_horiz
    vert = vert_lines[0] if len(vert_lines) > 0 else avg_vert
    
    return horiz, vert

def get_y_intercept(line):
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        slope = float(y2-y1)/float(x2-x1)
        intercept = y1 - x1 * slope

        return intercept

cap = cv2.VideoCapture('assets/video/trimmed_example.mp4')
i = 0
j = 0

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
trim_upper_factor = .9
trim_lower_factor = .25
out = cv2.VideoWriter('output.avi',fourcc, 30, (960,int(540 * trim_upper_factor - 540 * trim_lower_factor)))

avg_horiz = None
avg_vert = None

while cap.isOpened():
 
    i += 1
    ret, img = cap.read()
    if ret:
        # find the middle of the image
        if img is None:
            print('Img is none')

        img = imutils.resize(img, width=min(960, img.shape[1]))
        y_upper = int(img.shape[0] * .9)
        y_lower = int(img.shape[0] * .25)
        img = img[y_lower : y_upper, 0 : img.shape[1]]
        #img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 255)
        edges = cv2.Canny(blurred, 0, 150)

        lines = cv2.HoughLines(edges, 1, np.pi/180, 235)
        if lines is not None:
            # for line in lines:
            #     for rho, theta in line:
            #         a = np.cos(theta)
            #         b = np.sin(theta)
            #         x0 = a*abs(rho)
            #         y0 = b*abs(rho)
            #         x1 = int(x0 + 1000*(-b))
            #         y1 = int(y0 + 1000*(a))
            #         x2 = int(x0 - 1000*(-b))
            #         y2 = int(y0 - 1000*(a))
            #         slope = (y2-y1)/(x2-x1)
            #         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

            horiz, vert = trim_lines(lines, None, None)
            if avg_horiz is not None and horiz is None:
                horiz = avg_horiz
            if horiz is not None:
                if avg_horiz is None or (get_y_intercept(horiz) < 1.07 * get_y_intercept(avg_horiz) and get_y_intercept(horiz) > .93 * get_y_intercept(avg_horiz)):
                    avg_horiz = horiz

                if (get_y_intercept(horiz) > 1.07 * get_y_intercept(avg_horiz) or get_y_intercept(horiz) <.93 * get_y_intercept(avg_horiz)):
                    horiz = avg_horiz

                for rho, theta in horiz:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    slope = (y2-y1)/(x2-x1)
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
            
            if avg_vert is not None and vert is None:
                vert = avg_vert

            if vert is not None:
                avg_vert = vert
                for rho, theta in vert:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    slope = (y2-y1)/(x2-x1)
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        img = imutils.resize(img, width=min(960, img.shape[1]))
        orig = img.copy()

        (rects, weights) = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
        rects = filterLargeBoxes(rects)
        rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        for (xA,yA,xB,yB) in pick:
            if (yB > getY(avg_horiz, xA) and yB > getY(avg_horiz, xB)) and (yB > getY(avg_vert, xA) and yB > getY(avg_vert, xB)):
                cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

        out.write(img)
        j += 1
    else:
        break
                
cap.release()
out.release()
cv2.destroyAllWindows()
