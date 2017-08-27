# import numpy as np
# import cv2
# import imutils
# from matplotlib import pyplot as plt

# temp1 = cv2.imread('assets/images/top_down.png',0) # queryImage
# temp2 = cv2.imread('assets/images/still.png',0)    # trainImage

# # Initiate SIFT detector
# orb = cv2.ORB_create()

# # find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)

# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# # Match descriptors.
# matches = bf.match(des1,des2)

# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)

# # Draw first 10 matches.
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)

# plt.imshow(img3),plt.show()

import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

temp1 = cv2.imread('assets/images/top_down.png',0) # queryImage
temp2 = cv2.imread('assets/images/still.png',0)    # trainImage

img1 = imutils.resize(temp1, width=min(1280, temp1.shape[1]))
img2 = imutils.resize(temp2, width=min(1280, temp2.shape[1]))

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 2)
search_params = dict(checks=100)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

# Match descriptors.
# matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    print '{}, {}'.format(m.distance, n.distance)
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

# Draw first 10 matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches, None, **draw_params)

plt.imshow(img3),plt.show()
