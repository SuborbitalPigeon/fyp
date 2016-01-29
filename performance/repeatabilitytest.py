#!/usr/bin/env python3

import csv
from collections import OrderedDict
import os
from os.path import join
import re

import cv2
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.spatial import distance

from performancetest import PerformanceTest

THRESHOLD = 10


class RepeatabilityTest(PerformanceTest):
    def __init__(self, dirs, fileexts):
        super().__init__(dirs, fileexts)

        self.data = OrderedDict()

    @staticmethod
    def _transform_point(p, h):
        # Takes a row vector, and returns a column vector
        p = np.vstack((p, 1))  # Converts to homogenous coords
        d = np.dot(h, p)       # h * p
        d = (d / d[2])[0:2]    # Converts from homogenous coords
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

    def run_test(self, label, detector):
        kps = []
        ckps = []
        pattern = re.compile('(\w+)/img(\d).(\w+)')

        for file in self.files:
            match = pattern.match(file)
            (dir, num, ext) = match.groups()

            image = cv2.imread(file, 0)
            keypoints = self.get_keypoints(image, detector)

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

                if len(pts) == 0:
                    continue

                tpts = np.vstack(tpts)
                pts = np.vstack(pts)
                dist = distance.cdist(pts, tpts).min(axis=1)

                kps.append(len(dist)) # total evaluated keypoints
                ckps.append(np.sum(dist < THRESHOLD)) # corresponding keypoints

        if len(kps) > 0:
            self.data[label] = np.true_divide(ckps, kps)

    @staticmethod
    def _percent_format(y, position):
        s = str(y * 100)
        return s + '%'

    def show_plots(self):
        ytick = FuncFormatter(self._percent_format)
        fnames = [f.split('/')[1] for f in self.files[1:]] # filenames

        # One graph per detector
        for detector in self.detectors:
            plt.figure()

            for key, val in self.data.items():
                if key is detector:
                    plt.plot(val)

            plt.title("Detector = {}".format(detector))
            plt.xticks(np.arange(len(fnames)), fnames)
            plt.xlabel("Image")
            plt.gca().yaxis.set_major_formatter(ytick)
            plt.ylabel("Repeatability")
            plt.ylim(0, 1) # 0 % -- 100 %
            plt.draw()
            plt.savefig(join("results", "repeatability", detector.lower() + ".pdf"))

        #plt.show()

    def save_data(self):
        with open(join('results', 'scale.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(list(self.data.keys()))
            writer.writerows(zip(*self.data.values()))

if __name__ == '__main__':
    dirs = PerformanceTest.get_dirs_from_argv()
    test = RepeatabilityTest(dirs, ('pgm', 'ppm'))

    test.run_tests_only_detector()
    test.show_plots()
    test.save_data()
