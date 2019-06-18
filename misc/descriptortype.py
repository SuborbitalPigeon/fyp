#!/usr/bin/env python3

import cv2
from cv2 import xfeatures2d

cv2.ocl.setUseOpenCL(False)

file = '../performance/boat/img1.pgm'
img = cv2.imread(file)

akaze = cv2.AKAZE_create()
points = akaze.detect(img)

descriptors = [akaze]
descriptors.append(cv2.BRISK_create())
descriptors.append(cv2.KAZE_create())
descriptors.append(cv2.ORB_create())
descriptors.append(xfeatures2d.BriefDescriptorExtractor_create())
descriptors.append(xfeatures2d.DAISY_create())
descriptors.append(xfeatures2d.FREAK_create())
descriptors.append(xfeatures2d.LATCH_create())
descriptors.append(xfeatures2d.LUCID_create(1 ,1))

for descriptor in descriptors:
    des = descriptor.compute(img, points)[1]
    print("Algorithm: {}, size: {}, type: {}".format(descriptor, des[0].size, des[0].dtype))
