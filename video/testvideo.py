#!/usr/bin/env python3

import cv2

from videocontroller import VideoController

controller = VideoController()
cv2.namedWindow('output', cv2.WINDOW_NORMAL)

while True:
    frame = controller.frame
    cv2.imshow('output', frame)
    cv2.waitKey(40) # 40 ms = 25 fps
