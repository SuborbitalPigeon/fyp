#!/usr/bin/env python3

import csv
from os.path import join
from time import process_time

import cv2
from cv2 import xfeatures2d
from matplotlib import pyplot as plt
import numpy as np

from performancetest import PerformanceTest

class CombinedSpeedTest(PerformanceTest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._algos = [cv2.AKAZE_create()]
        self._algos.append(cv2.KAZE_create())
        self._algos.append(cv2.ORB_create())
        self._algos.append(xfeatures2d.SIFT_create())
        self._algos.append(xfeatures2d.SURF_create())

        self._times = {}
        self._nkps = {}

        self._images = [cv2.imread(image) for image in self.files]

    def run_tests(self):
        for algo in self._algos:
            label = str(algo).split()[0][1:] # Take only the algo name, and remove <
            print("Running test {}".format(label))
            times = []
            nkps = []

            for image in self._images:
                start = process_time()
                kps, des = algo.detectAndCompute(image, None)
                end = process_time()
                times.append((end-start)*1000)
                nkps.append(len(kps))

            self._times[label] = np.array(times)
            self._nkps[label] = np.array(nkps)

    def save_data(self):
        # FPS CSV
        with open(join('results', 'combinedspeed.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(list(self._times.keys()))
            writer.writerows(zip(*self._times.values()))

        # Number of keypoints CSV
        with open(join('results', 'combinednkps.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(list(self._nkps.keys()))
            writer.writerows(zip(*self._nkps.values()))

    def show_plots(self):
        plt.figure()
        labels = list(self._times.keys())
        data = list(self._times.values())
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.title("Combined speed test")
        plt.xticks(rotation=45)
        plt.xlabel("Algorithm")
        plt.ylabel("Time taken / ms")
        plt.draw()
        plt.savefig(join("results", "combinedspeed.pdf"))

if __name__ == '__main__':
    dirs = PerformanceTest.get_dirs_from_argv()
    test = CombinedSpeedTest(dirs=dirs, filexts=('pgm', 'ppm'))

    test.run_tests()
    test.save_data()
    test.show_plots()
