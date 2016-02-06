#!/usr/bin/env python3

import csv
from collections import OrderedDict
import itertools
from os.path import join
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

    def run_tests(self):
        count = 0
        for detector, descriptor in itertools.product(self.detectors, self.descriptors):
            count += 1
            label = "{}/{}".format(detector, descriptor)
            print("Running test {}/{}  - {}/{}".format(count, len(self.detectors) * len(self.descriptors), detector, descriptor))

            det, desc = self.create_detector_descriptor(detector, descriptor)
            if det == None or desc == None:
                print("Invalid combination - {}/{}".format(detector, descriptor))
                continue

            self.run_test(label, det, desc)

    def run_test(self, label, detector, descriptor):
        times = []
        nkps = []

        for file in self.files:
            image = cv2.imread(file, 0)
            start = time.clock()
            keypoints = self.get_keypoints(image, detector)
            (keypoints, descriptors) = self.get_descriptors(image, descriptor)
            end = time.clock()

            times.append(1 / (end - start))
            nkps.append(len(keypoints))

        self.times[label] = np.array(times)
        self.nkps[label] = np.array(nkps)

    def show_plots(self):
        # One graph per detector
        for detector in self.detectors:
            graphdict = OrderedDict()
            for key, val in self.times.items():
                if key.startswith(detector):
                    graphdict[key] = val

            labels = [l.split('/')[1] for l in graphdict.keys()] # only descriptor bit
            data = list(graphdict.values())
            plt.figure()
            plt.boxplot(data, labels=labels, showmeans=True)
            plt.title("Detector = {}".format(detector))
            plt.xticks(rotation=45)
            plt.xlabel("Descriptor")
            plt.ylabel("FPS")
            plt.draw()
            plt.savefig(join("results", "fps", detector.lower() + ".pdf"))

        #plt.show()

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
    dirs = PerformanceTest.get_dirs_from_argv()
    test = SpeedTest(dirs, ('pgm', 'ppm'))

    test.run_tests()
    test.show_plots()
    test.save_data()
