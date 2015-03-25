#!/usr/bin/env python

import numpy as np
import cv2
import sys

ran = 0
dx,dy,tfx,tfy = 0,0,0,0
duration = 0.98

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    x = np.int32(x)
    y = np.int32(y)

    fx, fy = flow[y,x].T
        
    fx = -fx
    fy = -fy
    global ran , dx , dy, tfx, tfy
    if ran == 0:
        ran = 1
        dx = abs(fx)
        dy = abs(fy)
        tfx = fx
        tfy = fy
    else:
        dx = (abs(dx) + abs(fx))*duration
        dy = (abs(dy) + abs(fy))*duration
        tfx = (tfx + fx)*duration
        tfy = (tfy + fy)*duration
        
    lines = np.vstack([x, y, x+tfx, y+tfy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    lines = list(zip(lines, list(range(0,3600))))
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for line, i in lines:
        x, y = line[0]
        
        density = dx[i] + dy[i]
        
        # change color based on density
        color = (0, 0, 255)
        
        if density < 30:
            continue
            
        if density < 40:
            color = (0, 255, 0)
        elif density < 60:
            color = (0, 255, 255)
            
        cv2.polylines(vis, [line], 0, color)
        cv2.circle(vis, tuple(line[0]), 1, color, -1)
         
    return vis


if __name__ == '__main__':

    # Get video file if given
    # Else open default camera
    if len(sys.argv) < 2:
        cap = cv2.VideoCapture(-1)
    else:
        videoPath = sys.argv[1]
        cap = cv2.VideoCapture(videoPath)

    #cap = cv2.VideoCapture('../../TestHUB2-small.mp4')
    
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
