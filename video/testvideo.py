#!/usr/bin/env python3

import cv2
import numpy as np

from videocontroller import VideoController

class UserInterface(object):
    def __init__(self):
        super().__init__()

        self._controller = VideoController()
        self._shown_frame = self._controller.frame

        self._initial_point = (0, 0)
        self._selecting = False
        self._roi = []

        cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('output', self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        # Only allow ROI selection while paused
        if self._controller.paused:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._selecting = True
                self._initial_point = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE:
                if self._selecting:
                    self._shown_frame = np.copy(self._controller.frame)
                    cv2.rectangle(self._shown_frame, self._initial_point, (x, y), (0,255,0), 1)

            elif event == cv2.EVENT_LBUTTONUP:
                self._selecting = False
                self._roi = [self._initial_point, (x, y)]
                print(self._roi)

    def run(self):
        while True:

            if self._controller.paused:
                cv2.imshow('output', self._shown_frame)
            else:
                frame = self._controller.frame
                if frame == None:
                    break
                cv2.imshow('output', frame)

            c = cv2.waitKey(40) # 40 ms = 25 fps
            if c == ord('p'):
                self._controller.paused = not self._controller.paused
                if self._controller.paused:
                    self._shown_frame = np.copy(self._controller.frame)

if __name__ == '__main__':
    interface = UserInterface()
    interface.run()    
