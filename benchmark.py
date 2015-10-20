#!/usr/bin/env python

import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

class Benchmark:
	def __init__(self, detector, descriptor):
		if detector not in ["FAST", "STAR", "ORB", "BRISK", "MSER",
                                    "GFTT", "HARRIS", "Dense", "SimpleBlob"]: # removed SIFT and SURF
			raise ValueError("Unsupported detector")
		if descriptor not in ["BRIEF", "BRISK", "ORB", "FREAK"]: # removed SIFT and SURF
			raise ValueError("Unsupported descriptor")

		self.detector = cv2.FeatureDetector_create(detector)
		self.descriptor = cv2.DescriptorExtractor_create(descriptor)
		self.times = []

	def get_descriptors(self, frame):
		start = time.clock()
		keypoints = self.detector.detect(frame)
		(keypoints, descriptors) = self.descriptor.compute(frame, keypoints)
		end = time.clock()

		self.times.append((end - start) * 1000) # convert to milliseconds

		return (keypoints, descriptors)

if __name__ == "__main__":
	detector = 'SimpleBlob'
	descriptor = 'FREAK'

	bench = Benchmark(detector, descriptor)
	cap = cv2.VideoCapture('test.mov')

	while True:
		(ret, frame) = cap.read()
		if ret == False:
			break

		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to greyscale
		bench.get_descriptors(grey)

	times = bench.times
	plt.boxplot(times, vert=False)
	plt.xlabel('Time / ms')
	plt.ylabel('')
	plt.title('{}/{}'.format(detector, descriptor))
	plt.show()
