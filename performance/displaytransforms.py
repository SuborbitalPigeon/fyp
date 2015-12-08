#!/usr/bin/env python3

import os
import re
import sys

import cv2
import numpy as np

def show_image(dir, file, num):
    basename = "img{}".format(num)
    cv2.namedWindow(basename)
    image = cv2.imread(os.path.join(dir, file))
    cv2.imshow(basename, image)

    # Base image has no transformation required
    if num is '1':
        return

    name = "img{}transform".format(num)
    cv2.namedWindow(name)
    image = cv2.imread(os.path.join(dir, file.replace(num, "1"))) # read 'base' file
    m = np.loadtxt(os.path.join(dir, 'H1to{}p'.format(num)))
    size = image.shape[1::-1]
    warped = cv2.warpPerspective(image, m, size)
    cv2.imshow(name, warped)

try:
    dir = sys.argv[1]
except:
    print("Pass a directory to display")
    sys.exit(-1)

pattern = re.compile('img(\d).(\w+)')

for file in os.listdir(dir):
    if file.endswith(('pgm', 'ppm')):
        match = pattern.match(file)
        (num, ext) = match.groups()
        
        show_image(dir, file, num)

cv2.waitKey(0)
