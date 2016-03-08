#!/usr/bin/env python3

import csv
from collections import OrderedDict
import itertools
from os.path import join
from time import process_time

import cv2
from matplotlib import pyplot as plt
import numpy as np

from performancetest import PerformanceTest


class SpeedTest(PerformanceTest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.times = OrderedDict()
        self.nkps = OrderedDict()

        self.images = [cv2.imread(image, 0) for image in self.files]

    def run_tests(self):
        count = 0
        for detector, descriptor in itertools.product(self.detectors, self.descriptors):
            count += 1
            label = "{}/{}".format(detector, descriptor)
            print("Running test {}/{}  - {}/{}".format(count, len(self.detectors) * len(self.descriptors), detector, descriptor))

            det = self.create_detector(detector)
            desc = self.create_descriptor(descriptor, detector)
            if desc == None:
                print("Invalid combination - {}/{}".format(detector, descriptor))
                continue

            self.run_test(label, det, desc)

    def run_test(self, label, detector, descriptor):
        times = []
        nkps = []

        for image in self.images:
            start = process_time()
            keypoints = self.get_keypoints(image, detector)
            (keypoints, descriptors) = self.get_descriptors(image, keypoints, descriptor)
            end = process_time()

            times.append((end - start) * 1000) # Milliseconds
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
            plt.ylabel("Time taken / ms")
            plt.draw()
            plt.savefig(join("results", "speed", detector.lower() + ".pdf"))

        #plt.show()

    def save_data(self):
        # FPS CSV
        with open(join('results', 'speed.csv'), 'w') as f:
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
    test = SpeedTest(dirs=dirs, filexts=('pgm', 'ppm'))

    test.run_tests()
    test.show_plots()
    test.save_data()
