import os
import re

import cv2
import numpy as np


image_re = re.compile('img(\d).\w+')


def ensure_path(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def transform_point(p, h):
    """
    Transform a point using a homography matrix.

    Parameters
    ----------
    p: tuple
        The point to transform.
    h: array_like
        The homography matrix.

    Returns
    -------
    tuple
        The point's new location.
    """
    p = np.reshape((p + (1,)), (-1, 1))
    d = h @ p
    return tuple((d / d[2])[:2].reshape(1, -1)[0])


def point_in_mask(p, mask):
    """
    Test if a point is in an area created by create_mask().

    Parameters
    ----------
    p: array_like
        A point.
    mask: np.ndarray
        A mask created with create_mask().

    Returns
    -------
    bool
        Whether or not the point is in the image represented by the mask.
    """
    p = np.rint(p).astype(int)
    x, y = p[0], p[1]

    if x < 0 or y < 0:
        return False

    try:
        return mask.item(x, y) != 0
    except IndexError:
        return False


def create_mask(shape, h):
    """
    Create an image mask for a transformation.

    Parameters
    ----------
    shape: tuple
        The shape of the image's ndarray.
    h: array_like
        The homography matrix to transform the mask by.

    Returns
    np.ndarray
        A mask which represents the image after the homography has been applied.
    """
    mask = np.full(shape, 255, np.uint8)
    mask = cv2.warpPerspective(mask, h, shape[1::-1])
    return (cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY))[1]


def h_for_file(path):
    """
    Get the homography matrix for a given image file.

    Parameters
    ----------
    path: str
        The path for the image.

    Returns
    -------
    np.ndarray, optional
        The homography matrix, if any
    """
    d, f = os.path.split(path)
    match = image_re.match(f)
    num = match.group(1)

    if int(num) != 1:
        hfile = os.path.join(d, "H1to{}p".format(num))
        return np.loadtxt(hfile)
    else:
        return None
