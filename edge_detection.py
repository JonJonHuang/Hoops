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

img = cv2.flip(cv2.imread('assets/images/still_cropped.png'), 1)
img = imutils.resize(img, width=min(960, img.shape[1]))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 255)
edges = cv2.Canny(blurred, 25, 150)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('edges.png', edges)

hist = cv2.calcHist([hsv],[0],None,[179],[0,179])

lines = cv2.HoughLines(edges, 1, np.pi/180, 235)
bs = []
slopes = []
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

        bs.append(y0)
        slope = (y2-y1)/(x2-x1)
        slopes.append(slope)

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

minSlopeIndex, maxSlopeIndex = minMaxIndex(slopes)
minSlope = lines[minSlopeIndex]
maxSlope = lines[maxSlopeIndex]
minB = bs[minSlopeIndex]
maxB = bs[maxSlopeIndex]

for x in range(0, img.shape[1]):
    for y in range(0, img.shape[0]):
        if x < getX(maxSlopeLine)

cv2.imwrite('houghlines3.jpg',img)
