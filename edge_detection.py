# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt

img = cv2.flip(cv2.imread('assets/images/still_cropped.png'), 1)
img = imutils.resize(img, width=min(960, img.shape[1]))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 255)
edges = cv2.Canny(blurred, 25, 150)
hsv = hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('edges.png', edges)

hist = cv2.calcHist([hsv[0]],[0],None,[179],[0,179])

lines = cv2.HoughLines(edges, 1, np.pi/180, 225)
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
