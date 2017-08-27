# import the necessary packages
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

def getY(m, b, x):
    return m*x + b

def getX(m, b, y):
    return (y-b)/m

def getCourtLineInfo(img):
    #img = imutils.resize(img, width=min(960, img.shape[1]))
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

            slope = float(y2-y1)/float(x2-x1)
            slopes.append(slope)
            b = y0 - (x0 * slope)
            bs.append(b)

            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

    minSlopeIndex, maxSlopeIndex = minMaxIndex(slopes)
    return (slopes[minSlopeIndex], bs[minSlopeIndex], slopes[maxSlopeIndex], bs[maxSlopeIndex])

if __name__ == "__main__":
    img, minSlope, minB, maxSlope, maxB = getCourtLineInfo('assets/images/still_cropped.png')

    for x in range(0, img.shape[1]-1):
        for y in range(0, img.shape[0]-1):
            if y < getY(minSlope, minB, x) or y < getY(maxSlope, maxB, x):
                img[y][x][0] = 0
                img[y][x][1] = 0
                img[y][x][2] = 0

    cv2.imwrite('houghlines3.jpg', cv2.flip(img, 1))
