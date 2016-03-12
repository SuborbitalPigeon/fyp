#!/usr/bin/env python3

from os import listdir
from os.path import isdir, join
import re
import sys

import cv2
import numpy as np

def write_image(file, image):
    # Reduce to 240px wide (20mm at 300 dpi)
    height, width = image.shape[:-1]
    height = int(height / (width / 240))
    resized = cv2.resize(image, (240, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(file, resized, params=[cv2.IMWRITE_JPEG_QUALITY, 75])

def process_images(dir, num, ext):
    bfile = join('transformed', dir, 'img' + num + '.jpg')
    wfile = join('transformed', dir, 'img' + num + 't.jpg')

    base = cv2.imread(join(dir, 'img1.' + ext)) # read 'base' file

    if num is '1':
        write_image(bfile, base)
    else:
        image = cv2.imread(join(dir, 'img' + num + '.' + ext))
        write_image(bfile, image)

        m = np.loadtxt(join(dir, 'H1to{}p'.format(num)))
        size = base.shape[1::-1]
        warped = cv2.warpPerspective(base, m, size)
        write_image(wfile, warped)

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
    process_images(dir, num, ext)

cv2.waitKey(0)
