import numpy as np
import cv2

img = cv2.imread('assets/images/soccer_screenshot.jpg', 0)

cv2.imshow('Messi', img)
cv2.waitKey(0)
cv2.destroyAllWindows()