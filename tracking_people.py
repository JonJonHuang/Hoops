# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from edge_detection import getCourtLineInfo
from edge_detection import getY

def filterLargeBoxes(boxArr):
    arr = boxArr[:]
    for i in range(len(arr)-1, -1, -1):
        x1, y1, w1, h1 = arr[i]
        for j in range(len(arr)-1, -1, -1):
            x2, y2, w2, h2 = arr[j]
            if (x2 < x1 and y2 < y1 and x2+w2 > x1+w1 and y2+h2 > y1+h1):
                arr = np.delete(arr, j, 0)
    return arr


if __name__ == "__main__":
    imagePath = 'assets/images/still_cropped.png'
    image = cv2.flip(cv2.imread(imagePath), 1)
    image = imutils.resize(image, width=min(960, image.shape[1]))

    minSlope, minB, maxSlope, maxB = getCourtLineInfo(image)

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = imutils.resize(image, width=min(1280, image.shape[1]))
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = filterLargeBoxes(rects)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        if ((y+h) > getY(minSlope, minB, x) and (y+h) > getY(minSlope, minB, x+w)) and ((y+h) > getY(maxSlope, maxB, x) and (y+h) > getY(maxSlope, maxB, x+w)):
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        if (yB > getY(minSlope, minB, xA) and yB > getY(minSlope, minB, xB)) and (yB > getY(maxSlope, maxB, xA) and yB > getY(maxSlope, maxB, xB)):
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))

    # show the output images
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)