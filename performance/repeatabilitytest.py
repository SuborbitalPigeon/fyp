#!/usr/bin/env python3

import csv
from collections import OrderedDict
from os.path import join
import re

import cv2
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.spatial import distance

from performancetest import PerformanceTest

THRESHOLD = 2


class RepeatabilityTest(PerformanceTest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.common = OrderedDict()
        self.baset = OrderedDict()

    def run_tests(self):
        count = 0
        for detector in self.detectors:
            count += 1
            print("Running test {}/{} - {}".format(count, len(self.detectors), detector))

            det = self.create_detector(detector)
            self.run_test(detector, det)

    def run_test(self, label, detector):
        common = []
        baset = []
        pattern = re.compile('(\w+)/img(\d).(\w+)')

        for file in self.files:
            match = pattern.match(file)
            (dir, num, ext) = match.groups()

            print("Processing file {}".format(num))
            image = cv2.imread(file, 0)
            keypoints = self.get_keypoints(image, detector)

            if num is '1':
                basepts = keypoints
                common.append(len(basepts))
                baset.append(len(basepts))
                continue

            pts = [] # current image's keypoints
            tpts = [] # transformed base keypoints

            mat = np.loadtxt(join(dir, 'H1to{}p'.format(num)))
            mask = self.create_mask(image.shape, mat)

            pts = np.vstack([point.pt for point in keypoints if self.point_in_image(point.pt, mask)])

            # The base image's keypoints, projection required
            for point in basepts:
                tp = self.transform_point(point.pt, mat)
                if self.point_in_image(tp, mask):
                    tpts.append(tp)
            tpts = np.vstack(tpts)

            if len(pts) > len(tpts):
                dist = distance.cdist(pts, tpts).min(axis=0)
            else:
                dist = distance.cdist(pts, tpts).min(axis=1)

            common.append(len(tpts))
            baset.append(np.sum(dist < THRESHOLD))

        self.common[label] = common
        self.baset[label] = baset

    @staticmethod
    def _percent_format(y, position):
        s = str(y * 100)
        return s + '%'

    def show_plots(self):
        ytick = FuncFormatter(self._percent_format)
        fnames = [f.split('/')[1] for f in self.files[1:]] # filenames

        for (ckey, cval), (tkey, tval) in zip(self.common.items(), self.baset.items()):
            plt.plot(np.divide(tval[1:], cval[1:]), label=ckey)

        plt.title("2-Repeatability")
        plt.xticks(np.arange(len(fnames)), fnames)
        plt.xlabel("Image")
        plt.gca().yaxis.set_major_formatter(ytick)
        plt.ylim(0, 1) # 0 % -- 100 %
        plt.legend(loc='best', framealpha=0.5)
        plt.draw()
        plt.savefig(join("results", "repeatability.pdf"))

    def save_data(self):
        with open(join('results', 'repeat-common.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(list(self.common.keys()))
            writer.writerows(zip(*self.common.values()))

        with open(join('results', 'repeat-baset.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(list(self.baset.keys()))
            writer.writerows(zip(*self.baset.values()))

if __name__ == '__main__':
    dirs = PerformanceTest.get_dirs_from_argv()
    test = RepeatabilityTest(dirs=dirs, filexts=('pgm', 'ppm'))
    test.run_tests()
    test.show_plots()
    test.save_data()
