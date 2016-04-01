#!/usr/bin/env python3

import csv
from os.path import join
import re

import cv2
from matplotlib import pyplot as plt
import numpy as np

from performancetest import PerformanceTest


class PrecisionRecall(PerformanceTest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.recall = {}
        self.precision = {}

    def run_tests(self):
        count = 0
        det = cv2.AKAZE_create()

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
        pattern = re.compile('(\w+)/img(\d).(\w+)')
        matches = []
        corresponding = []

        for file in self.files:
            match = pattern.match(file)
            (dir, num, ext) = match.groups()
            print("Processing file {}".format(num))

            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            kps = self.get_keypoints(img, detector)
            kps, des = self.get_descriptors(img, kps, descriptor)

            if des.dtype == np.float32:
                bf = cv2.BFMatcher(cv2.NORM_L2)
            else:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)

            # First image in the sequence, store kps and descriptors for matching
            if num == '1':
                self._baseimg = img
                basekps = kps
                basedes = des
                continue

            # Require homography and mask for image comparison
            hi = np.linalg.inv(np.loadtxt(join(dir, 'H1to{}p'.format(num))))
            mask = self.create_mask(img.shape, hi)

            thismatches = bf.match(des, basedes)
            matches.extend(thismatches)

            # Find close match pairs and designate corresponding
            for m in thismatches:
                basekp = basekps[m.trainIdx]
                kp = kps[m.queryIdx]
                tkp = self.transform_point(kp.pt, hi)
                if self.point_in_image(tkp, mask):
                    if self.get_overlap_error(basekp, kp, hi, self._baseimg.shape) < 0.4:
                        corresponding.append(m)

        dists = [m.distance for m in matches]
        recall = []
        precision = []

        for t in np.linspace(np.min(dists), np.max(dists), 20):
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
    dirs = PerformanceTest.get_dirs_from_argv()
    test = PrecisionRecall(dirs=dirs, filexts=('pgm', 'ppm'))

    test.run_tests()
    test.show_plots()
    test.save_data()
