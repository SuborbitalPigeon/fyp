from collections import OrderedDict
import csv
from os.path import isfile, join
import re
import sys

import cv2
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from performancetest import PerformanceTest


class PrecisionRecall(PerformanceTest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        testname = kwargs['testimg']
        m = re.match('(\w+)/img(\d).(\w+)', testname)
        (dir, num, ext) = m.groups()
        basename = join(dir, 'img1.' + ext)

        self.det = cv2.xfeatures2d.SURF_create()

        self.img1 = cv2.imread(basename, cv2.IMREAD_GRAYSCALE)
        self.img2 = cv2.imread(testname, cv2.IMREAD_GRAYSCALE)

        self.h = np.loadtxt(join(dir, 'H1to' + num + 'p'))
        self.hi = np.linalg.inv(self.h)

        self.kp1 = self.det.detect(self.img1, None)
        self.kp2 = self.det.detect(self.img2, None)

        self.precision = OrderedDict()
        self.recall = OrderedDict()

    def run_test(self, desc):
        des = self.create_descriptor(desc, 'SURF')
        if des is None:
            return

        print("Testing {}".format(desc))

        kp1, des1 = des.compute(self.img1, self.kp1)
        kp2, des2 = des.compute(self.img2, self.kp2)
        mask = self.create_mask(self.img1.shape, self.hi)

        if des1.dtype == np.float32:
            bf = cv2.BFMatcher(cv2.NORM_L2)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des1, des2)

        corresponding = []
        for m in matches:
            query = kp1[m.queryIdx]
            train = kp2[m.trainIdx]
            transformed = self.transform_point(train.pt, self.hi)

            # Remove points outside the common image area
            if self.point_in_image(query.pt, mask) is False:
                matches.remove(m)
                continue
            if self.point_in_image(transformed, mask) is False:
                matches.remove(m)
                continue

            if self.get_overlap_error(query, train, self.hi, self.img1.shape) < 0.4:
                corresponding.append(m)

        if len(corresponding) == 0:
            return

        precision = []
        recall = []
        dists = [m.distance for m in matches]
        for t in np.linspace(np.min(dists), np.max(dists)):
            tp = fp = 0

            for m in [mat for mat in matches if mat.distance < t]:
                if m in corresponding:
                    tp += 1
                else:
                    fp += 1

            if (tp + fp) == 0:
                continue

            precision.append(1 - (tp / (tp + fp)))
            recall.append(tp / len(corresponding))

        self.precision[desc] = precision
        self.recall[desc] = recall

    def run_tests(self):
        for desc in self.descriptors:
            self.run_test(desc)

    @staticmethod
    def _percent_format(y, position):
        s = str(y * 100)
        return s + '%'

    def show_plots(self):
        for key in self.precision.keys():
            plt.plot(self.precision[key], self.recall[key], '+', label=key)

        plt.xlabel("1-precision")
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.ylabel("recall")
        tick = FuncFormatter(self._percent_format)
        plt.gca().xaxis.set_major_formatter(tick)
        plt.gca().yaxis.set_major_formatter(tick)
        plt.legend(loc='best')

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
        f = sys.argv[1]

    test = PrecisionRecall(testimg=f)
    test.run_tests()
    test.show_plots()
    test.save_data()
