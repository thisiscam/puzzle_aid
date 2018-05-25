import cv2
import numpy as np

import matplotlib as mpl
mpl.use('WebAgg')

from matplotlib import pyplot as plt

# Import our game board
canvas = cv2.imread('./data/puzzle.jpg')
# Import our piece (we are going to use a clump for now)
piece = cv2.imread('./data/p2.png')

# Pre-process the piece
def identify_contour(piece, threshold_low=100, threshold_high=200):
    """Identify the contour around the piece"""
    piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY) # better in grayscale
    piece_gray = cv2.GaussianBlur(piece_gray, (5,5),0)
    ret, piece_thres = cv2.threshold(piece_gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    image, contours, heirarchy = cv2.findContours(piece_thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key = cv2.contourArea)
    # contour_sorted = np.argsort(map(cv2.contourArea, contours))
    return c

def get_bounding_rect(contour):
    """Return the bounding rectangle given a contour"""
    x,y,w,h = cv2.boundingRect(contour)
    return x, y, w, h

def get_contour_mask(piece, contour):
    mask = np.zeros(piece.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask

# Get the contours
contour = identify_contour(piece.copy())

# Get a bounding box around the piece
x, y, w, h = get_bounding_rect(contour)
contour_mask = get_contour_mask(piece, contour)
cropped_piece = piece.copy()[y:y+h, x:x+w]
cropped_mask = contour_mask.copy()[y:y+h, x:x+w]

# Initiate SIFT detector
sift =  cv2.xfeatures2d.SIFT_create()

img1 = cropped_piece.copy() # queryImage
img2 = canvas.copy() # trainImage

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, cropped_mask)
kp2, des2 = sift.detectAndCompute(img2,None)

import pdb; pdb.set_trace()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
#     good.append(m)
    if m.distance < 0.7*n.distance:
        good.append(m)


MIN_MATCH_COUNT = 10

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    d,h,w = img1.shape[::-1]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imwrite('./output/solution.jpg', img3)