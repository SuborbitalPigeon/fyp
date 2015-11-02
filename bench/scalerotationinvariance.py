#!/usr/bin/env python

from __future__ import division

import os
import re

import cv2
import numpy as np
from scipy.spatial import distance

from benchmark import Benchmark

THRESHOLD = 10

#TODO look at https://github.com/Itseez/opencv/blob/master/modules/features2d/src/evaluation.cpp
#TODO only consider keypoints within the transformed image plane

class ScaleRotationInvariance(Benchmark):
    def __init__(self, dirs, fileexts):
        super(ScaleRotationInvariance, self).__init__(dirs, fileexts)
        self.kps = {}

    @staticmethod
    def _transform_point(p, h):
        p = np.vstack((p, 1))  # (x, y, 1)^T
        d = np.dot(h, p)       # h * p
        d = (d / d[2])[0:2] # Divide rows 1 and 2 by 3, return only these rows
        return np.transpose(d)

    def run_test(self, detector, descriptor, label):
        pattern = re.compile('(\w+)/img(\d).(\w+)')

        for file in self.files:
            match = pattern.match(file)
            (dir, num, ext) = match.groups()

            image = cv2.imread(file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            (keypoints, descriptors) = self.get_keypoints(image, detector, descriptor)

            if num is '1':
                basepts = [point.pt for point in keypoints]
            else:
                tpts = [] # transformed base keypoints

                mat = np.loadtxt(os.path.join(dir, 'H1to{}p'.format(num)))

                # This image's keypoints
                pts = [point.pt for point in keypoints]
                # The base image's keypoints, projection required
                for point in basepts:
                    p = np.array([[point[0]], [point[1]]])
                    tpts.append(self._transform_point(p, mat))

                tpts = np.vstack(tpts)
                pts = np.vstack(pts)
                dist = distance.cdist(pts, tpts)
                count = np.sum(np.any(dist < THRESHOLD, axis=1))
                print("Image {} correspondence: {:.2f} %".format(num, (count / len(dist) * 100)))

    def show_plots(self):
        raise NotImplementedError("Not implemented yet")

    def save_data(self):
        raise NotImplementedError("Not implemented yet")

if __name__ == '__main__':
    dirs = ['bark']
    bench = ScaleRotationInvariance(dirs, 'ppm')

    bench.run_tests()
