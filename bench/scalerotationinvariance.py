#!/usr/bin/env python

from __future__ import division

import csv
import os
import re

import cv2
import numpy as np
from scipy.spatial import distance

from benchmark import Benchmark

THRESHOLD = 10

class ScaleRotationInvariance(Benchmark):
    def __init__(self, dirs, fileexts):
        super(ScaleRotationInvariance, self).__init__(dirs, fileexts)

        self.data = {}

    @staticmethod
    def _transform_point(p, h):
        p = np.vstack((p, 1))  # (x, y, 1)^T
        d = np.dot(h, p)       # h * p
        d = (d / d[2])[0:2] # Divide rows 1 and 2 by 3, return only these rows
        return np.transpose(d)[0]

    @staticmethod
    def _point_in_image(pt, image):
        # Checks the image at this point, and returns True if this is totally black
        (x, y) = pt

        try:
            ret = np.any(image[x][y] == np.array([0, 0, 0]))
        except IndexError: # if point is outside bounds of image
            return False

        return ret

    def run_test(self, detector, descriptor, label):
        kps = []
        ckps = []
        pattern = re.compile('(\w+)/img(\d).(\w+)')

        for file in self.files:
            match = pattern.match(file)
            (dir, num, ext) = match.groups()

            image = cv2.imread(file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            (keypoints, descriptors) = self.get_keypoints(image, detector, descriptor)

            if num is '1':
                basepts = [point.pt for point in keypoints]
                baseimg = image
            else:
                pts = [] # current image's keypoints
                tpts = [] # transformed base keypoints

                size = image.shape[1::-1]
                mat = np.loadtxt(os.path.join(dir, 'H1to{}p'.format(num)))
                rimage = cv2.warpPerspective(baseimg, mat, size) # for filtering keypoints

                # This image's keypoints
                for point in keypoints:
                    if self._point_in_image(point.pt, rimage):
                        pts.append(point.pt)

                # The base image's keypoints, projection required
                for point in basepts:
                    p = np.array([[point[0]], [point[1]]])
                    tp = self._transform_point(p, mat)
                    if self._point_in_image(tp, rimage):
                        tpts.append(tp)

                tpts = np.vstack(tpts)
                pts = np.vstack(pts)
                dist = distance.cdist(pts, tpts)
                kps.append(len(dist)) # total evaulated keypoints
                ckps.append(np.sum(np.any(dist < THRESHOLD, axis=1))) # corresponding keypoints

        self.data[label] = np.true_divide(ckps, kps)

    def show_plots(self):
        raise NotImplementedError("Not implemented yet")

    def save_data(self):
        with open('scale.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(self.data.keys())
            writer.writerows(zip(*self.data.values()))

if __name__ == '__main__':
    dirs = ['boat']
    bench = ScaleRotationInvariance(dirs, 'pgm')

    bench.run_tests()
    bench.save_data()
