import cv2
import numpy as np

original = cv2.imread('assets/images/still.png', 1)
hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
low = (90, 50, 50)
high = (140, 255, 255)

mask = cv2.inRange(hsv, low, high)

cv2.imshow('original', hsv)
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
