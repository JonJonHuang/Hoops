# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt

def minMaxIndex(arr):
    min = abs(arr[0])
    max = abs(arr[0])
    minIndex = 0
    maxIndex = 0
    for i in range(0, len(arr)):
        val = abs(arr[i])
        if val < min:
            min = val
            minIndex = i
        if val >= max:
            max = val
            maxIndex = i
    return (minIndex, maxIndex)

def maxIndex(arr):
    max = arr[0]

def getY(m, b, x):
    return m*x + b

def getX(m, b, y):
    return (y-b)/m

img = cv2.flip(cv2.imread('assets/images/still.png'), 1)
img = imutils.resize(img, width=min(960, img.shape[1]))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 255)
edges = cv2.Canny(blurred, 25, 150)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('edges.png', edges)

hist = cv2.calcHist([hsv], [0], None, [179], [0,179])
color = ('b', 'g', 'r')

hsv_lower_bound = np.array([0, 100, 100])
hsv_upper_bound = np.array([30, 255, 255])

mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask= mask)

gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 255)
edges = cv2.Canny(blurred, 25, 150)

cv2.imshow('edges', edges)
#cv2.imshow('res', res)
cv2.imshow('mask', mask)

cv2.waitKey(0)

lines = cv2.HoughLines(edges, 1, np.pi/180, 235)
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

cv2.imwrite('houghlines3.jpg',img)
