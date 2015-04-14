#!/usr/bin/env python
import numpy as np
import cv2
import sys

threshold = 20
block_size = 16

density = None

def draw_flow(img, prev, step=16):
    global density
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    x = np.int32(x)
    y = np.int32(y)
    
    
    td = np.int16(img) - np.int16(prev)
    
    sx = int(w/block_size)
    sy = int(h/block_size)
    
    if density == None:
        density = np.zeros((sy, sx))
    
    mask = np.zeros((h, w, 3))
    
    for i in range(sx):
        for j in range(sy):
            if i*block_size+block_size > w:
                continue
            
            if j*block_size+block_size > h:
                continue
            
            b = td[j*block_size:j*block_size+block_size, i*block_size:i*block_size+block_size]
            thresh = abs(b) > threshold
            
            d = 0
            for ki in range(thresh.shape[1]):
                for kj in range(thresh.shape[0]):
                    if thresh[ki][kj]:
                        d += 1
                    
            density[j][i] += d/(block_size*block_size)
            
            mask[j*block_size:j*block_size+block_size, i*block_size:i*block_size+block_size, 0] = 0
            mask[j*block_size:j*block_size+block_size, i*block_size:i*block_size+block_size, 1] = 0
            mask[j*block_size:j*block_size+block_size, i*block_size:i*block_size+block_size, 2] = 255
  
            if density[j][i] < 0.02:
                mask[j*block_size:j*block_size+block_size, i*block_size:i*block_size+block_size, 0] = 0
                mask[j*block_size:j*block_size+block_size, i*block_size:i*block_size+block_size, 1] = 255
                mask[j*block_size:j*block_size+block_size, i*block_size:i*block_size+block_size, 2] = 0
            elif density[j][i] < 0.5:
                mask[j*block_size:j*block_size+block_size, i*block_size:i*block_size+block_size, 0] = 0
                mask[j*block_size:j*block_size+block_size, i*block_size:i*block_size+block_size, 1] = 255
                mask[j*block_size:j*block_size+block_size, i*block_size:i*block_size+block_size, 2] = 255
        
            density[j][i] -= 0.1
            
            if (density[j][i] > 1):
                density[j][i] = 1
                
            if (density[j][i] < 0):
                density[j][i] = 0
    return np.uint8(mask)

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
        #flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 20, 3, 5, 1.2, 0)
        d_img = draw_flow(gray, prevgray)
        cv2.imshow('de', cv2.add(np.uint8(0.6*img), np.uint8(0.4*d_img)))
        prevgray = gray


        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()
