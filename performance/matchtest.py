#!/usr/bin/env python3

import csv
from itertools import chain
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

        self.recall = {}
        self.precision = {}

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

    def _get_overlap(self, kp1, kp2, h):
        """ Get overlap between two keypoints

        Parameters
        ----------
        kp1: cv2.KeyPoint
            the first keypoint (transformed by homography)
        kp2: cv2.KeyPoint
            the second keypoint.
        h: array_like
            the homography matrix

        Returns
        ------
        overlap: float
            The overlap percentage
        """
        shape = self._baseimg.shape

        img1 = np.zeros(shape, np.uint8)
        cv2.circle(img1, (int(kp1.pt[0]), int(kp1.pt[1])), int(kp1.size / 2), 255, -1, cv2.LINE_AA)
        img1 = cv2.warpPerspective(img1, h, self._baseimg.shape[1::-1])
        ret, img1 = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)

        img2 = np.zeros(shape, np.uint8)
        cv2.circle(img2, (int(kp2.pt[0]), int(kp2.pt[1])), int(kp2.size / 2), 255, -1, cv2.LINE_AA)
        ret, img2 = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)

        union = cv2.bitwise_or(img1, img2)
        intersection = cv2.bitwise_and(img1, img2)
        return np.sum(intersection) / np.sum(union)

    def run_test(self, label, detector, descriptor):
        pattern = re.compile('(\w+)/img(\d).(\w+)')
        matches = []
        corresponding = []

        if label == 'SIFT' or 'SURF':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for file in self.files:
            match = pattern.match(file)
            (dir, num, ext) = match.groups()
            print("Processing file {}".format(num))

            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            kps = self.get_keypoints(img, detector)
            kps, des = self.get_descriptors(img, kps, descriptor)

            # First image in the sequence, store kps and descriptors for matching
            if num == '1':
                self._baseimg = img
                basekps = kps
                basedes = des
                continue

            # Require homography and mask for image comparison
            h = np.loadtxt(join(dir, 'H1to{}p'.format(num)))
            mask = self.create_mask(img.shape, h)

            thismatches = bf.match(des, basedes)
            matches.extend(thismatches)

            # Find close match pairs and designate corresponding
            for m in thismatches:
                basekp = basekps[m.trainIdx]
                kp = kps[m.queryIdx]
                tbasekp = self.transform_point(basekp, h)
                if self.point_in_image(tbasekp, mask):
                    if self._get_overlap(basekp, kp, h) > 0.2:
                        corresponding.append(m)

        dists = [m.distance for m in matches]
        lower = np.min(dists)
        upper = np.max(dists)
        recall = []
        precision = []

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

            precision.append(1 - (tp / (tp + fp)))
            recall.append(tp / len(corresponding))

        self.recall[label] = recall
        self.precision[label] = precision

    def show_plots(self):
        for key in self.precision.keys():
            plt.scatter(self.precision[key], self.recall[key], label=key, cmap=plt.get_cmap("jet"))

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
    dirs = PerformanceTest.get_dirs_from_argv()
    test = MatchTest(dirs=dirs, filexts=('pgm', 'ppm'))

    test.run_tests()
    test.show_plots()
    test.save_data()
