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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data = OrderedDict()

    def run_tests(self):
        count = 0
        for detector in self.detectors:
            count += 1
            print("Running test {}/{} - {}".format(count, len(self.detectors), detector))

            det = self.create_detector(detector)
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
                basepts = keypoints
            else:
                pts = [] # current image's keypoints
                tpts = [] # transformed base keypoints

                mat = np.loadtxt(os.path.join(dir, 'H1to{}p'.format(num)))
                mask = self.create_mask(image.shape, mat)

                # This image's keypoints
                for point in keypoints:
                    if self.point_in_image(point, mask):
                        pts.append(point.pt)

                # The base image's keypoints, projection required
                for point in basepts:
                    tp = self.transform_point(point , mat)
                    if self.point_in_image(tp, mask):
                        tpts.append(tp.pt)

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
    test = RepeatabilityTest(dirs=dirs, filexts=('pgm', 'ppm'))
    test.run_tests()
    test.show_plots()
    test.save_data()
