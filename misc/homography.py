#!/usr/bin/env python3

import cv2
import numpy as np


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
    mat = np.loadtxt('bark/H1to2p')

    coor = np.array([[200], [400]])
    tcoor = transform_point(coor, mat)
    print("Original point: {}, transformed point: {}".format(coor.T, tcoor.T))

    cv2.namedWindow("baseimg")
    cv2.namedWindow("img2")
    cv2.namedWindow("baseimg2")

    img = cv2.imread('bark/img1.ppm')
    cv2.circle(img, (coor[0], coor[1]), 20, (0, 0, 255), 5)
    cv2.imshow("baseimg", img)

    img2 = cv2.imread('bark/img2.ppm')
    cv2.imshow("img2", img2)

    size = img.shape[1::-1]
    img2h = cv2.warpPerspective(img, mat, size)
    cv2.circle(img2h, (tcoor[0], tcoor[1]), 25, (255, 0, 0), 5)
    cv2.imshow("baseimg2", img2h)

    cv2.waitKey(0)
