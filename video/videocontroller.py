#!/usr/bin/env python3

FILE_PATTERN = "images/image%05d.png"

import cv2
from matplotlib import pyplot as plt

class VideoController(object):
    def __init__(self):
        super().__init__()

        self._cap = cv2.VideoCapture(FILE_PATTERN)
        self._paused = False

    @property
    def paused(self):
        return self._paused

    @paused.setter
    def paused(self, value):
        self._paused = value

    @property
    def frame(self):
        if self.paused == True:
            return self._saved_frame
        else:
            ret, img = self._cap.read()
            self._saved_frame = img
            return img

if __name__ == '__main__':
    controller = VideoController()
    frame = controller.frame
    plt.imshow(frame)
    plt.show()
