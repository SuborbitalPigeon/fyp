#!/usr/bin/env python3

import csv
from collections import OrderedDict
import itertools
from os.path import isdir, join
import sys
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

from performancetest import PerformanceTest

class SpeedTest(PerformanceTest):
    def __init__(self, dirs, fileexts):
        """ Benchmark concerned with the raw speed of combinations of detector and descriptor.

        Parameters
        ----------
        dirs : List[str]
            A list of directories to scan.
        filexts : Tuple[str]
            A tuple containing the file extensions to allow for test images.

        """
        super().__init__(dirs, fileexts)
        self.times = OrderedDict()
        self.nkps = OrderedDict()

    def run_test(self, detector, descriptor, label):
        times = []
        nkps = []

        for file in self.files:
            image = cv2.imread(file, 0)
            start = time.clock()
            (keypoints, descriptors) = self.get_keypoints(image, detector, descriptor)
            end = time.clock()

            times.append(1 / (end - start))
            nkps.append(len(keypoints))

        self.times[label] = np.array(times)
        self.nkps[label] = np.array(nkps)

    def show_plots(self):
        pass
        # TODO: reimplment graphing

    def save_data(self):
        # FPS CSV
        with open(join('results', 'fps.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(list(self.times.keys()))
            writer.writerows(zip(*self.times.values()))

        # Number of keypoints CSV
        with open(join('results', 'nkps.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(list(self.nkps.keys()))
            writer.writerows(zip(*self.nkps.values()))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("No directories given")

    dirs = [dir for dir in sys.argv[1:] if isdir(dir)]
    test = SpeedTest(dirs, ('pgm', 'ppm'))

    test.run_tests()
    test.show_plots()
    test.save_data()
