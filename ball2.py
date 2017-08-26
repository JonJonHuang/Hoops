import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    greenLower = np.array([134, 30, 45])
    greenUpper = np.array([148, 46, 35])

    mask = cv2.inRange(hsv, greenLower, greenUpper)

    res = cv2.bitwise_and(frame, frame, mask= mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitkey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
