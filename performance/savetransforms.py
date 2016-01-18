#!/usr/bin/env python3

from os import listdir
from os.path import isdir, join
import re
import sys

import cv2
import numpy as np

def write_image(dir, num, ext):
    wfile = join('transformed', dir, 'img' + num + '.png')

    base = cv2.imread(join(dir, 'img1.' + ext)) # read 'base' file

    if num is '1':
        cv2.imwrite(wfile, base)
    else:
        m = np.loadtxt(join(dir, 'H1to{}p'.format(num)))
        size = base.shape[1::-1]
        warped = cv2.warpPerspective(base, m, size)
        cv2.imwrite(wfile, warped)

try:
    dirs = [dir for dir in sys.argv[1:] if isdir(dir)]
except:
    print("Pass a directory to display")
    sys.exit(-1)

pattern = re.compile('(\w+)/img(\d).(\w+)')

files = [join(dir, file) for dir in dirs for file in listdir(dir) if file.endswith(('pgm', 'ppm'))]
for file in files:
    match = pattern.match(file)
    (dir, num, ext) = match.groups()
    write_image(dir, num, ext)

cv2.waitKey(0)
