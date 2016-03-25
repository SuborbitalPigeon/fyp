import cv2
import numpy as np

class Tracking:
    def __init__(self):
        self._det = cv2.AgastFeatureDetector_create(threshold=30)
        self._desc = cv2.ORB_create()

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
        search_params = dict(checks = 50)
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = target
        self._targetkps = self._det.detect(target, None)
        self._targetkps, self._targetdes = self._desc.compute(target, self._targetkps)

    def find_homography(self, image):
        kps = self._det.detect(image, None)
        kps, des = self._desc.compute(image, kps)
        matches = self._matcher.knnMatch(self._targetdes, des, 2)

        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)

        if len(good) < 4:
            return # Not enough good matches

        apts = np.float32([self._targetkps[m.queryIdx].pt for m in good])
        bpts = np.float32([kps[m.trainIdx].pt for m in good])

        H, mask = cv2.findHomography(apts, bpts, cv2.RANSAC, 3.0)
        return H
