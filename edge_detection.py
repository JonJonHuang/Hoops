# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt

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
        slopes.append(slope)
        b = y0 - (x0 * slope)
        bs.append(b)

                if (get_y_intercept(horiz) > 1.07 * get_y_intercept(avg_horiz) or get_y_intercept(horiz) <.93 * get_y_intercept(avg_horiz)):
                    horiz = avg_horiz

minSlopeIndex, maxSlopeIndex = minMaxIndex(slopes)
minSlope = slopes[minSlopeIndex]
maxSlope = slopes[maxSlopeIndex]
minB = bs[minSlopeIndex]
maxB = bs[maxSlopeIndex]

height = img.shape[0]
for x in range(0, img.shape[1]-1):
    for y in range(0, img.shape[0]-1):
        if y < getY(minSlope, minB, x) or y < getY(maxSlope, maxB, x):
            img[y][x][0] = 0
            img[y][x][1] = 0
            img[y][x][2] = 0

            out.write(img)
            j += 1
    else:
        break
                
cap.release()
out.release()
cv2.destroyAllWindows()
