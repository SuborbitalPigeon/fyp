from collections import OrderedDict
import csv
from os.path import join

import cv2
from matplotlib import pyplot as plt
import numpy as np

from performancetest import PerformanceTest

IMG1FILE = "graf/img1.ppm"
IMG2FILE = "graf/img3.ppm"
HFILE = "graf/H1to3p"


class PrecisionRecall(PerformanceTest):
    def __init__(self):
        self.det = cv2.xfeatures2d.SURF_create()

        self.precision = OrderedDict()
        self.recall = OrderedDict()

    def run_tests(self):
        img1 = cv2.imread(IMG1FILE, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(IMG2FILE, cv2.IMREAD_GRAYSCALE)

        h = np.loadtxt(HFILE)
        hi = np.linalg.inv(h)

        des = cv2.ORB_create()

        kp1 = self.det.detect(img1, None)
        kp1, des1 = des.compute(img1, kp1)
        kp2 = self.det.detect(img2, None)
        kp2, des2 = des.compute(img2, kp2)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.match(des1, des2)

        corresponding = []
        for m in matches:
            query = kp1[m.queryIdx]
            train = kp2[m.trainIdx]
            if self.get_overlap_error(query, train, hi, img1.shape) < 0.4:
                corresponding.append(m)

        # This is ludicrously low
        print("Corresponding:", len(corresponding))

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

        self.precision['ORB'] = precision
        self.recall['ORB'] = recall

    def show_plots(self):
        for key in self.precision.keys():
            plt.plot(self.precision[key], self.recall[key], 'r+')

        plt.xlabel("1-precision")
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.ylabel("recall")

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
    test = PrecisionRecall()
    test.run_tests()
    test.show_plots()
    test.save_data()
