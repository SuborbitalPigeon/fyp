#!/usr/bin/env python

from __future__ import division

import csv
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

from benchmark import Benchmark

# removed SIFT and SURF from both lists
detectors = ['FAST', 'STAR', 'ORB', 'BRISK', 'MSER', 'GFTT', 'HARRIS', 'Dense', 'SimpleBlob']
descriptors = ['BRIEF', 'BRISK', 'ORB', 'FREAK']

class SpeedBenchmark(Benchmark):
    def __init__(self, dirs, fileexts):
        super(SpeedBenchmark, self).__init__(dirs, fileexts)
        self.times = {}
        self.nkps = {}

    def create_detector_descriptor(self, detector, descriptor):
        if detector not in detectors:
            raise ValueError("Unsupported detector")
        if descriptor not in descriptors:
            raise ValueError("Unsupported descriptor")

        det = cv2.FeatureDetector_create(detector)
        desc = cv2.DescriptorExtractor_create(descriptor)
        return (det, desc)

    def run_test(self, detector, descriptor):
        times = []
        nkps = []

        for file in self.files:
            image = cv2.imread(file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            start = time.clock()
            keypoints = detector.detect(image)
            (keypoints, descriptors) = descriptor.compute(image, keypoints)
            end = time.clock()

            times.append(1 / (end - start))
            nkps.append(len(keypoints))

        return (np.array(times), np.array(nkps))

    def run_tests(self):
        count = 0
        for detector in detectors:
            for descriptor in descriptors:
                count += 1
                name = "{}/{}".format(detector, descriptor)
                print("Running test {}/{}: {}".format(count, len(detectors) * len(descriptors), name))

                det, desc = self.create_detector_descriptor(detector, descriptor)
                self.times[name], self.nkps[name] = self.run_test(det, desc)

                # FPS
                (mean, stdev) = self.get_mean_stdev(self.times[name])
                print("FPS - mean: {:.2f} Hz, stdev: {:.2f} Hz".format(mean, stdev))

                # Keypoints
                (mean, stdev) = self.get_mean_stdev(self.nkps[name])
                print("Keypoints - mean: {:.2f}, stdev: {:.2f}".format(mean, stdev))


    def show_plots(self):
        # FPS plot
        plt.boxplot(self.times.values(), labels=self.times.keys())
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=45)
        plt.title("Frame rate")
        plt.ylabel("Frame rate / FPS")
        plt.show()

        # Number of keypoints plot
        plt.figure()
        plt.boxplot(self.nkps.values(), labels=self.nkps.keys())
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=45)
        plt.title("Keypoints")
        plt.ylabel("Keypoints")
        plt.yscale('log')
        plt.show()

    def save_data(self):
        # FPS CSV
        with open('fps.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(self.times.keys())
            writer.writerows(zip(*self.times.values()))

        # Number of keypoints CSV
        with open('nkps.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(self.nkps.keys())
            writer.writerows(zip(*self.nkps.values()))

    @staticmethod
    def get_mean_stdev(data):
        mean = np.mean(data)
        stdev = np.std(data)
        return (mean, stdev)

if __name__ == '__main__':
    dirs = ['bark', 'bikes', 'boat', 'graf', 'leuven', 'trees', 'ubc', 'wall']
    bench = SpeedBenchmark(dirs, ('pgm', 'ppm'))

    bench.run_tests()
    bench.show_plots()
    bench.save_data()
