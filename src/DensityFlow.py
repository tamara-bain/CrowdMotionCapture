#!/usr/bin/env python

##############################################################################
# DensityFlow.py
# Code Written By: Michael Feist, Maciej Ogrocki, and Tamara Bain
#
# Displays a density map over the video.
#
# To run:
# python DensityFlow.py [OPTIONS]
#
# For Help:
# python DensityFlow.py --help
##############################################################################

import argparse
import sys

import numpy as np
import cv2

threshold = 20
block_size = 16

density = None

parser = argparse.ArgumentParser(
        prog='Rectification', 
        usage='python %(prog)s.py [options]')
parser.add_argument(
    '--video', 
    type=str, 
    help='path to input video')
parser.add_argument(
    '--blockSize', 
    type=int, 
    help='size of blocks used for density.')

def setColor(block, color):
    block[:, :, 0] = color[0]
    block[:, :, 1] = color[1]
    block[:, :, 2] = color[2]

    return block

def drawDensity(img, prev, step=16):
    global density
    h, w = img.shape[:2]

    td = np.int16(img) - np.int16(prev)
    
    sx = int(w/step)
    sy = int(h/step)
    
    if density == None:
        density = np.zeros((sy, sx))
    
    mask = np.zeros((h, w, 3))
    
    for i in range(sx):
        for j in range(sy):
            if i*step+step > w:
                continue
            
            if j*step+step > h:
                continue
            
            b = td[j*step:j*step+step, i*step:i*step+step]
            thresh = abs(b) > threshold
            
            d = 0
            for ki in range(thresh.shape[1]):
                for kj in range(thresh.shape[0]):
                    if thresh[ki][kj]:
                        d += 1
                    
            density[j][i] += 0.5*d/(step*step)
            
            if density[j][i] < 0.02:
                setColor(
                    mask[j*step:j*step+step, i*step:i*step+step, :], 
                    (0, 255, 0))
            elif density[j][i] < 0.5:
                v = 1.0 - 2.*(density[j][i] - 0.52)

                if v > 1.0:
                    v = 1.0
                if v < 0.0:
                    v = 0.0

                setColor(
                    mask[j*step:j*step+step, i*step:i*step+step, :], 
                    (0, 255, np.uint8(v*255)))
            else:
                v = 1.0 - 2.*(density[j][i] - 0.5)
                
                if v > 1.0:
                    v = 1.0
                if v < 0.0:
                    v = 0.0

                setColor(
                    mask[j*step:j*step+step, i*step:i*step+step, :], 
                    (0, np.uint8(v*255), 255))
        
            density[j][i] -= 0.05
            
            if (density[j][i] > 1):
                density[j][i] = 1
                
            if (density[j][i] < 0):
                density[j][i] = 0
    return np.uint8(mask)

if __name__ == '__main__':
    # Parse Arguments
    args = parser.parse_args(sys.argv[1:])

    # Get video file if given
    # Else open default camera
    if args.video != None:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(-1)

    if args.blockSize != None:
        block_size = args.blockSize

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

        # Draw density
        d_img = drawDensity(gray, prevgray, block_size)
        cv2.imshow('de', cv2.add(np.uint8(0.6*img), np.uint8(0.4*d_img)))
        prevgray = gray

        # Handle keyboard input
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()
