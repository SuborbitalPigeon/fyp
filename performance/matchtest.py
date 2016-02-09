#!/usr/bin/env python3

from os.path import isfile, join
import re
import sys

import cv2
from matplotlib import pyplot as plt
import numpy as np

from performancetest import PerformanceTest


class MatchTest(PerformanceTest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        testname = kwargs['testimg']

        m = re.match('(\w+)/img(\d).(\w+)', testname)
        (dir, num, ext) = m.groups()
        basename = join(dir, 'img' + '1.' + ext)

        self.baseimg = cv2.imread(basename)
        self.baseimg = cv2.cvtColor(self.baseimg, cv2.COLOR_BGR2GRAY)
        self.timg = cv2.imread(testname)
        self.timg = cv2.cvtColor(self.timg, cv2.COLOR_BGR2GRAY)
        self.h = np.loadtxt(join(dir, 'H1to' + num + 'p'))

        self.recall = {}
        self.precision = {}

    def run_tests(self):
        count = 0
        det = cv2.ORB_create()

        for descriptor in self.descriptors:
            count += 1
            label = "{}".format(descriptor)
            print("Running test {}/{}  - {}".format(count, len(self.descriptors), descriptor))

            desc = self.create_descriptor(descriptor, 'ORB')
            if desc == None:
                print("Invalid combination - ORB/{}".format(descriptor))
                continue

            self.run_test(label, det, desc)

    def run_test(self, label, detector, descriptor):
        basekps = self.get_keypoints(self.baseimg, detector)
        basekps, basedes = self.get_descriptors(self.baseimg, basekps, descriptor)

        kps = self.get_keypoints(self.timg, detector)
        kps, des = self.get_descriptors(self.timg, kps, descriptor)

        mask = np.empty(self.baseimg.shape, np.uint8)
        mask.fill(255)
        mask = cv2.warpPerspective(mask, self.h, self.baseimg.shape[1::-1])

        if label == 'SIFT' or 'SURF':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des, basedes)
        dists = [m.distance for m in matches]
        lower = np.min(dists)
        upper = np.max(dists)
        recall = []
        precision = []

        for t in np.linspace(lower, upper, 20):
            overlaps = []
            corresponding = 0

            for m in matches:
                basekp = basekps[m.trainIdx]
                tkp = kps[m.queryIdx]
                basekp = self.transform_point(basekp, self.h)
                if self.point_in_image(basekp, mask):
                    corresponding += 1
                    if m.distance < t:
                        overlaps.append(cv2.KeyPoint_overlap(basekp, tkp))

            if len(overlaps) is 0:
                continue

            correct = np.sum(np.asarray(overlaps) > 0.5)
            wrong = np.sum(np.asarray(overlaps) <= 0.5)

            recall.append(correct / corresponding)
            precision.append(wrong / len(overlaps))

        plt.plot(precision, recall, 'x')
        plt.xlabel("1 - precision")
        plt.xlim(xmin=0, xmax=1)
        plt.ylim(ymin=0, ymax=1)
        plt.ylabel("recall")
        plt.show()

    def show_plots(self):
        pass

    def save_data(self):
        pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("No file")
    if isfile(sys.argv[1]):
        img = sys.argv[1]
    test = MatchTest(testimg=img)
    test.run_tests()
    test.show_plots()
    test.save_data()
