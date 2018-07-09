import cv2
import numpy as np
import math

import matplotlib as mpl
mpl.use('WebAgg')

from matplotlib import pyplot as plt


def binary_thres(piece, threshold_low=240, threshold_high=255):
    piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY) # better in grayscale
    piece_thres = cv2.adaptiveThreshold(piece_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 3)
    return piece_thres

def adaptive_thres(piece):
    piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY) # better in grayscale
    piece_gray = cv2.GaussianBlur(piece_gray, (51, 51), 0)
    piece_thres = cv2.adaptiveThreshold(piece_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 501, 3)
    return piece_thres

# Pre-process the piece
def identify_contours(piece, thres=adaptive_thres):
    """Identify the contour around the piece"""
    # piece = cv2.fastNlMeansDenoisingColored(piece,None,10,10,7,21)
    piece_thres = thres(piece)
    # piece_thres = cv2.medianBlur(piece_thres, 51)
    image, contours, heirarchy = cv2.findContours(piece_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    assert len(contours) > 0

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    area = cv2.contourArea(contours[0][0])
    pieces_contours = []
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < 0.4 * area:
            break
        else:
            pieces_contours.append(cnt)
            area = cnt_area

    contours = pieces_contours

    def merge_contours(contours):
        areas = list(map(cv2.contourArea, contours))
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append(np.array([cX, cY]))

        for i in range(1, len(contours)):
            for j in range(i):
                if i != j:
                    r1 = np.sqrt(areas[i] / np.pi)
                    r2 = np.sqrt(areas[j] / np.pi)
                    d = np.linalg.norm(centers[i] - centers[j])
                    if r1 + r2 > d:
                        new_cnt = cv2.convexHull(np.vstack([contours[i], contours[j]]))
                        contours.pop(i)
                        contours.pop(j)
                        contours.append(new_cnt)
                        merge_contours(contours)
    
    merge_contours(contours)
    return pieces_contours

def get_contour_mask(piece, contour):
    mask = np.zeros(piece.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask

def segment_pieces(pieces, thres=adaptive_thres):
    # Get the contours
    contours = identify_contours(pieces.copy(), thres)

    ret = []
    for contour in contours:
        # Get a bounding box around the piece
        x, y, w, h = cv2.boundingRect(contour)
        contour_mask = get_contour_mask(pieces, contour)
        cropped_piece = pieces.copy()[y:y+h, x:x+w]
        cropped_mask = contour_mask.copy()[y:y+h, x:x+w]
        plt.imshow(cropped_mask)
        plt.show()
        ret.append((cropped_piece, cropped_mask))

    return contours, ret

def init_sift(sift, whole_img):
    # Initiate SIFT detector
    img2 = whole_img.copy()
    kp2, des2 = sift.detectAndCompute(img2, None)
    return img2, kp2, des2

def matcher(piece, cropped_mask, sift, kp2, des2):
    img1 = piece.copy()
    cropped_mask = cropped_mask.copy()

    kp1, des1 = sift.detectAndCompute(img1, cropped_mask)

    # find the keypoints and descriptors with SIFT
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    return good, img1, kp1, des1

MIN_MATCH_COUNT = 10

def draw_matches(good, img1, kp1, des1, img2, kp2, des2):

    if len(good)>=MIN_MATCH_COUNT:
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

    return cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

def draw_matches_whole(goods, kp1s, kp2, pieces_img, whole_img, contours):
    # stich images
    height = max(pieces_img.shape[0], whole_img.shape[0])
    width = pieces_img.shape[1] + whole_img.shape[1]
    output = np.zeros((height,width,3))
    pieces_img_w = pieces_img.shape[1]
    
    x = 0
    for image in [pieces_img, whole_img]:
        h,w,d = image.shape
        output[0:h,x:x+w] = image
        x += w


    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 3
    lineType               = cv2.LINE_AA
    i = 0
    for good, kp1, contour in zip(goods, kp1s, contours):
        if len(good) < MIN_MATCH_COUNT:
            continue
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
        src_center = np.average(src_pts, axis=0)
        dst_center = np.average(dst_pts, axis=0)
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        src_center += np.array([x1, y1])
        dst_center += np.array([pieces_img_w, 0])
        src_center = (src_center[0], src_center[1])
        dst_center = (dst_center[0], dst_center[1])
        # cv2.line(output, (src_center[0], src_center[1]), (dst_center[0], dst_center[1]), (0, 255, 0), thickness=3, lineType=8)
        cv2.putText(output, str(i), src_center, font, fontScale, (0,255,0), 10, lineType)
        cv2.putText(output, str(i), dst_center, font, fontScale, (0,255,0), 10, lineType)
        i += 1
    return output

import argparse, os
from contour_match import find_curvatures

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--whole', help='entire puzzle image')
    parser.add_argument('--pieces', help='white background image of all the query pieces')
    parser.add_argument('--output', default='output', help='output directory')
    args = parser.parse_args()

    whole_img = cv2.imread(args.whole)
    pieces_img = cv2.imread(args.pieces)

    sift =  cv2.xfeatures2d.SIFT_create()

    print "Init sift"
    img2, kp2, des2 = init_sift(sift, whole_img)

    print "Segment pieces"
    contours, pieces = segment_pieces(pieces_img, binary_thres)
    import pdb; pdb.set_trace()
    curvatures = find_curvatures(contours[0])

    print "Identified {} piece(s)".format(len(pieces))

    base_filename = os.path.basename(args.pieces)

    kp1s, goods = [], []
    for i, (piece, mask) in enumerate(pieces):
        print "Matching {0}".format(i)
        good, img1, kp1, des1 = matcher(piece, mask, sift, kp2, des2)
        # solution = draw_matches(good, img1, kp1, des1, img2, kp2, des2)
        # cv2.imwrite(os.path.join(args.output, base_filename + '-sol-{i}.png'.format(i=i)), solution)
        kp1s.append(kp1)
        goods.append(good)

    output = draw_matches_whole(goods, kp1s, kp2, pieces_img, whole_img, contours)
    cv2.imwrite(os.path.join(args.output, base_filename + '-sol.png'), output)
