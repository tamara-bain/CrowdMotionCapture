#!/usr/bin/env python

import numpy as np
import cv2


def draw_flow(img, flow, step=16):
    # Get image width and height
    h, w = img.shape[:2]

    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)

    x = np.int32(x)
    y = np.int32(y)

    fx, fy = flow[y,x].T
    fx = -fx
    fy = -fy

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # Grab first frame
    ret, prev = cap.read()

    # Convert to grey scale
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        # Grab second frame
        ret, img = cap.read()

        # Convert to grey scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 20, 3, 5, 1.2, 0)
        prevgray = gray

        cv2.imshow('flow', draw_flow(gray, flow))

        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()
