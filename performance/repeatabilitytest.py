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
        (x, y) = pt

        try:
            ret = image[x][y] != 0 # If mask rectangle is visible
        except IndexError:
            return False # Outside the mask rectangle image boundaries

        return ret

    def run_tests(self):
        count = 0
        for detector in self.detectors:
            count += 1
            print("Running test {}/{} - {}".format(count, len(self.detectors), detector))

            det = self.create_detector_descriptor(detector)
            self.run_test(detector, det)

    def run_test(self, label, detector):
        repeatability = []
        pattern = re.compile('(\w+)/img(\d).(\w+)')

        for file in self.files:
            match = pattern.match(file)
            (dir, num, ext) = match.groups()

            image = cv2.imread(file, 0)
            keypoints = self.get_keypoints(image, detector)

            if num is '1':
                basepts = [point.pt for point in keypoints]
            else:
                pts = [] # current image's keypoints
                tpts = [] # transformed base keypoints

                mat = np.loadtxt(os.path.join(dir, 'H1to{}p'.format(num)))

                # Create rectangle which is warped for purposes of boundary checking
                rectangle = np.empty(image.shape, np.uint8)
                rectangle.fill(255)
                rimage = cv2.warpPerspective(rectangle, mat, image.shape[1::-1])

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

                repeatability.append(np.sum(dist < THRESHOLD) / len(tpts))

            self.data[label] = repeatability

    @staticmethod
    def _percent_format(y, position):
        s = str(y * 100)
        return s + '%'

    def show_plots(self):
        ytick = FuncFormatter(self._percent_format)
        fnames = [f.split('/')[1] for f in self.files[1:]] # filenames

        # TODO Linestyle changes?
        colour = iter(plt.cm.jet(np.linspace(0,1,len(self.data))))

        for key, val in self.data.items():
            c = next(colour)
            plt.plot(val, label=key, c=c)

        plt.title("10-Repeatability")
        plt.xticks(np.arange(len(fnames)), fnames)
        plt.xlabel("Image")
        plt.gca().yaxis.set_major_formatter(ytick)
        plt.ylabel("10-Repeatability")
        plt.ylim(0, 1) # 0 % -- 100 %
        plt.legend(loc='best', framealpha=0.5)
        plt.draw()
        plt.savefig(join("results", "repeatability.pdf"))

    def save_data(self):
        with open(join('results', 'repeat.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(list(self.data.keys()))
            writer.writerows(zip(*self.data.values()))

if __name__ == '__main__':
    dirs = PerformanceTest.get_dirs_from_argv()
    test = RepeatabilityTest(dirs, ('pgm', 'ppm'))

    test.run_tests()
    test.show_plots()
    test.save_data()
