FILE_PATTERN = "images/image%05d.png"

import cv2
from matplotlib import pyplot as plt

class VideoController(object):
    def __init__(self):
        super().__init__()

        self._cap = cv2.VideoCapture(FILE_PATTERN)
        self._paused = True
        ret, self._saved_frame = self._cap.read()

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
            if ret == False:
                return None
            self._saved_frame = img
            return img
