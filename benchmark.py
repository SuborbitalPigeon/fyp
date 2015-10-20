import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

class Benchmark:
	def __init__(self):
		self.detector = cv2.ORB()
		self.descriptor = cv2.ORB()

		self.times = []

	def get_descriptors(self, frame):
		start = time.clock()
		keypoints = self.detector.detect(frame)
		(keypoints, descriptors) = self.descriptor.compute(frame, keypoints)
		end = time.clock()

		self.times.append(end - start)

		return (keypoints, descriptors)

if __name__ == "__main__":
	bench = Benchmark()
	cap = cv2.VideoCapture('test.mov')

	while True:
		(ret, frame) = cap.read()
		bench.get_descriptors(frame)
		if ret == False:
			break

	times = bench.times
	plt.boxplot(times, vert=False)
	plt.show()
