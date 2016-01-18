#!/usr/bin/env python3

import os
import re
import sys

import cv2
import numpy as np

def write_image(dir, file, num):
    # Base image has no transformation required
    if num is '1':
        return

    wfile = os.path.join('transformed', dir, file)
    wfile = wfile.replace('pgm', 'png')
    wfile = wfile.replace('ppm', 'png')

    image = cv2.imread(os.path.join(dir, file.replace(num, "1"))) # read 'base' file
    m = np.loadtxt(os.path.join(dir, 'H1to{}p'.format(num)))
    size = image.shape[1::-1]
    warped = cv2.warpPerspective(image, m, size)

    cv2.imwrite(wfile, warped)

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
        
        write_image(dir, file, num)

cv2.waitKey(0)
