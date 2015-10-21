#!/usr/bin/env python

from __future__ import division
import csv
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

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

	def get_descriptors(self, frame):
		start = time.clock()
		keypoints = self.detector.detect(frame)
		(keypoints, descriptors) = self.descriptor.compute(frame, keypoints)
		end = time.clock()

		self.times.append(1 / (end - start))

		return (keypoints, descriptors)

	def run_tests(self, filename):
		cap = cv2.VideoCapture(filename)
		while True:
			(ret, frame) = cap.read()
			if ret == False:
				break

			grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to greyscale
			self.get_descriptors(grey)
		return self.times
			

if __name__ == '__main__':
	times = {}

	for detector in detectors:
		for descriptor in descriptors:
			name = "{}/{}".format(detector, descriptor)
			
			print("Running {}".format(name))
			bench = Benchmark(detector, descriptor)
			times[name] = bench.run_tests('test.mov')

			mean = np.mean(times[name])
			stdev = np.std(times[name])
			print("Mean: {:.2f} Hz, stdev: {:.2f} Hz".format(mean, stdev))

	# Boxplots
	plt.boxplot(times.values())
	# replace this with "labels" keyword parameter if possible
	plt.xticks(range(1, len(times.keys()) + 1), times.keys())
	plt.ylabel("Frame rate / FPS")
	plt.show()

	# CSV
	with open('data.csv', 'wb') as f:
		writer = csv.writer(f)
		writer.writerow(times.keys())
		writer.writerows(zip(*times.values()))
