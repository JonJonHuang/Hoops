from collections import deque
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to video file")
ap.add_argument("-b", "--buffer", type=int, default=80, help="max buffer size")
args = vars(ap.parse_args())
