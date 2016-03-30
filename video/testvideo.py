#!/usr/bin/env python3

import cv2
import numpy as np

from tracking import Tracking
from videocontroller import VideoController

class UserInterface(object):
    def __init__(self):
        super().__init__()

        self._controller = VideoController()
        self._shown_frame = self._controller.frame

        self._initial_point = (0, 0)
        self._selecting = False
        self._tracking = Tracking()

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
                    self._shown_frame = cv2.cvtColor(self._shown_frame, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(self._shown_frame, self._initial_point, (x, y), (0,255,0), 1)

            elif event == cv2.EVENT_LBUTTONUP:
                self._selecting = False
                tl = np.min((self._initial_point, (x, y)), axis=0) # top left
                br = np.max((self._initial_point, (x, y)), axis=0) # bottom right
                roi = np.copy(self._controller.frame[tl[1]:br[1], tl[0]:br[0]])
                self._tracking.target = roi

    def run(self):
        while True:

            if self._controller.paused:
                cv2.imshow('output', self._shown_frame)
            else:
                frame = self._controller.frame
                if frame == None:
                    self._tracking.perf.save_data()
                    break

                H = self._tracking.find_homography(frame)
                if H is not None:
                    h, w = self._tracking.target.shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, H)

                    img = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                    cv2.imshow('output', img)
                else:
                    cv2.imshow('output', frame)

            c = cv2.waitKey(1)
            if c == ord('p'):
                self._controller.paused = not self._controller.paused
                if self._controller.paused:
                    self._shown_frame = np.copy(self._controller.frame)

if __name__ == '__main__':
    interface = UserInterface()
    interface.run()    
