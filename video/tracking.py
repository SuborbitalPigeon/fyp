import cv2
import numpy as np
from matplotlib import pyplot as plt

class Tracking:
    def __init__(self):
        self._orb = cv2.ORB_create()

#        FLANN_INDEX_LSH = 6
#        index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
#        search_params = dict(checks = 50)
#        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = target
        self._targetkps, self._targetdes = self._orb.detectAndCompute(target, None)

    def find_homography(self, image):
        kps, des = self._orb.detectAndCompute(image, None)
        matches = self._matcher.knnMatch(self._targetdes, des, 2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) < 4:
            return # Not enough good matches

        apts = np.float32([self._targetkps[m.queryIdx].pt for m in good])
        bpts = np.float32([kps[m.trainIdx].pt for m in good])

        H, mask = cv2.findHomography(apts, bpts, cv2.RANSAC, 3.0)
        return H

if __name__ == '__main__':
    tracking = Tracking()

    img = cv2.imread('images/image00001.png', cv2.IMREAD_GRAYSCALE)
    roi = np.array([[550, 375], [780, 550]])
    obj = np.copy(img[roi[0,1]:roi[1, 1], roi[0, 0]:roi[1, 0]])

    tracking.define_target(obj)
    H = tracking.find_homography(img)

    h,w = img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,H)

    img = cv2.polylines(img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    plt.imshow(img)
    plt.show()
