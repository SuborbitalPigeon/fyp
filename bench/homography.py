#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def file_to_matrix(filename):
    """ Opens and parses a homography file and converts to a matrix.

    Parameters
    ----------
    filename : str
        The file to parse.

    Returns
    -------
    h : ndarray
        The homography matrix.

    """
    with open(filename) as f:
        data = f.readlines()

    data = [line.split() for line in data]
    return np.array(data[:3], dtype=float) # strip the last newline

def transform_point(p, h):
    """ Transforms a 2D point according to a homography matrix.

    Similar to OpenCV's warpPerspective function, but operates on a single point.

    Parameters
    ----------
    p : ndarray
        A 2D point.
    h : ndarray
        A homography matrix.

    Returns
    -------
    pt : ndarray
        The point after transformation.

    """
    p = np.vstack((p, 1))  # (x, y, 1)^T
    d = np.dot(h, p)       # h * p
    return (d / d[2])[0:2] # Divide rows 1 and 2 by 3, return only these rows

if __name__ == '__main__':
    mat = file_to_matrix('bark/H1to2p')

    coor = np.array([[400], [200]])
    tcoor = transform_point(coor, mat)

    cv2.namedWindow("img")
    cv2.namedWindow("img2")
    cv2.namedWindow("img2h")

    img = cv2.imread('bark/img1.ppm')
    cv2.circle(img, (coor[0], coor[1]), 20, (0, 0, 255), 5)
    cv2.imshow("img", img)

    img2 = cv2.imread('bark/img2.ppm')
    cv2.imshow("img2", img2)

    img2h = cv2.warpPerspective(img, mat, (800, 600))
    cv2.circle(img2h, (tcoor[0], tcoor[1]), 25, (255, 0, 0), 5)
    cv2.imshow("img2h", img2h)

    cv2.waitKey(0)
