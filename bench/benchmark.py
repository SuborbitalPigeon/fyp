#!/usr/bin/env python

from __future__ import division

import csv
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

from fileutils import get_images_from_dirs

# removed SIFT and SURF from both lists
detectors = ['FAST', 'STAR', 'ORB', 'BRISK', 'MSER', 'GFTT', 'HARRIS', 'Dense', 'SimpleBlob']
descriptors = ['BRIEF', 'BRISK', 'ORB', 'FREAK']

class Benchmark:
    def __init__(self, detector, descriptor):
        if detector not in detectors:
            raise ValueError("Unsupported detector")
        if descriptor not in descriptors: 
            raise ValueError("Unsupported descriptor")

        self.detector = cv2.FeatureDetector_create(detector)
        self.descriptor = cv2.DescriptorExtractor_create(descriptor)
        self.times = []
        self.nkps = []

    def get_descriptors(self, frame):
        start = time.clock()
        keypoints = self.detector.detect(frame)
        (keypoints, descriptors) = self.descriptor.compute(frame, keypoints)
        end = time.clock()

        self.times.append(1 / (end - start))
        self.nkps.append(len(keypoints))

        return (keypoints, descriptors)

    def run_tests(self, filenames):
        for file in filenames:
            image = cv2.imread(file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            (kps, descriptors) = self.get_descriptors(image)
        return (self.times, self.nkps)

def get_mean_stdev(data):
    mean = np.mean(data)
    stdev = np.std(data)
    return (mean, stdev)

if __name__ == '__main__':
    times = {}
    nkps = {} # number of keypoints

    count = 0
    for detector in detectors:
        for descriptor in descriptors:
            count += 1
            name = "{}/{}".format(detector, descriptor)
            print("Running test {}/{}: {}".format(count, len(detectors) * len(descriptors), name))

            bench = Benchmark(detector, descriptor)
            dirs = ['bark', 'bikes', 'boat', 'graf', 'leuven', 'trees', 'ubc', 'wall']
            files = get_images_from_dirs(dirs)

            times[name], nkps[name] = bench.run_tests(files)

            # FPS
            (mean, stdev) = get_mean_stdev(times[name])
            print("FPS - mean: {:.2f} Hz, stdev: {:.2f} Hz".format(mean, stdev))

            # Keypoints
            (mean, stdev) = get_mean_stdev(nkps[name])
            print("Keypoints - mean: {:.2f}, stdev: {:.2f}".format(mean, stdev))

    # FPS plot
    plt.boxplot(times.values())
    plt.title("Frame rate")
    # replace this with "labels" keyword parameter if possible
    plt.xticks(range(1, len(times.keys()) + 1), times.keys(), rotation=45)
    plt.ylabel("Frame rate / FPS")
    plt.show(block=False)

    # Number of keypoints plot
    plt.figure()
    plt.boxplot(nkps.values(), labels=times.keys())
    plt.title("Keypoints")
    # replace this with "labels" keyword parameter if possible
    plt.xticks(range(1, len(times.keys()) + 1), times.keys(), rotation=45)
    plt.ylabel("Keypoints")
    plt.yscale('log')
    plt.show()

    # FPS CSV
    with open('fps.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(times.keys())
        writer.writerows(zip(*times.values()))

    # Number of keypoints CSV
    with open('nkps.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(nkps.keys())
        writer.writerows(zip(*nkps.values()))
