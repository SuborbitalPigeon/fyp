#!/usr/bin/env python3

import csv
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

        self.mask = self.create_mask(self.baseimg.shape, self.h)

    def run_tests(self):
        count = 0
        det = cv2.ORB_create(nfeatures=5000)

        for descriptor in self.descriptors:
            count += 1
            label = "{}".format(descriptor)
            print("Running test {}/{}  - {}".format(count, len(self.descriptors), descriptor))

            desc = self.create_descriptor(descriptor, 'ORB')
            if desc == None:
                print("Invalid combination - ORB/{}".format(descriptor))
                continue

            self.run_test(label, det, desc)

    def _get_overlap(self, kp1, kp2):
        """ Get overlap between two keypoints

        Parameters
        ----------
        kp1: cv2.KeyPoint
            the first keypoint.
        kp2: cv2.KeyPoint
            the second keypoint.

        Returns
        ------
        overlap: float
            The overlap percentage
        """
        shape = self.baseimg.shape
        H = self.h

        img1 = np.zeros(shape, np.uint8)
        cv2.circle(img1, (int(kp1.pt[0]), int(kp1.pt[1])), int(kp1.size / 2), 255, -1, cv2.LINE_AA)
        img1 = cv2.warpPerspective(img1, H, self.baseimg.shape[1::-1])
        ret, img1 = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)

        img2 = np.zeros(shape, np.uint8)
        cv2.circle(img2, (int(kp2.pt[0]), int(kp2.pt[1])), int(kp2.size / 2), 255, -1, cv2.LINE_AA)
        ret, img2 = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)

        union = cv2.bitwise_or(img1, img2)
        intersection = cv2.bitwise_and(img1, img2)
        return np.sum(intersection) / np.sum(union)

    def run_test(self, label, detector, descriptor):
        basekps = self.get_keypoints(self.baseimg, detector)
        basekps, basedes = self.get_descriptors(self.baseimg, basekps, descriptor)

        kps = self.get_keypoints(self.timg, detector)
        kps, des = self.get_descriptors(self.timg, kps, descriptor)

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
        corresponding = []

        for m in matches:
            basekp = basekps[m.trainIdx]
            tbasekp = self.transform_point(basekp, self.h)
            if self.point_in_image(tbasekp, self.mask):
                if self._get_overlap(basekp, kps[m.queryIdx]) > 0.2:
                    corresponding.append(m)

        for t in np.linspace(lower, upper, 20):
            tp = 0
            fp = 0

            for m in matches:
                if m.distance < t:
                    if m in corresponding:
                        tp += 1
                    else:
                        fp += 1

            if tp == 0 and fp == 0:
                tp = 1

            precision.append(fp / (tp + fp))
            recall.append(tp / len(corresponding))

        self.recall[label] = recall
        self.precision[label] = precision

    def show_plots(self):
        for key in self.precision.keys():
            plt.plot(self.precision[key], self.recall[key], label=key)

        plt.xlabel("1-precision")
        plt.xlim(xmin=0, xmax=1)
        plt.ylabel("recall")
        plt.ylim(ymin=0, ymax=1)
        plt.legend(loc='best')
        plt.draw()
        plt.savefig(join("results", "precisionrecall.pdf"))

    def save_data(self):
        with open(join('results', 'precision.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(list(self.precision.keys()))
            writer.writerows(zip(*self.precision.values()))

        with open(join('results', 'recall.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(list(self.recall.keys()))
            writer.writerows(zip(*self.recall.values()))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("No file")
    if isfile(sys.argv[1]):
        img = sys.argv[1]
    test = MatchTest(testimg=img)
    test.run_tests()
    test.show_plots()
    test.save_data()
