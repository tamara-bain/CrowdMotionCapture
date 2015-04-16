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

save_video = False
videoPath = ""

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
parser.add_argument(
    '--saveFrames', 
    type=str, 
    help='path to save images.')


# Helper function to set the color for each element in a matrix
def setColor(block, color):
    block[:, :, 0] = color[0]
    block[:, :, 1] = color[1]
    block[:, :, 2] = color[2]

    return block

# Calculate and draw the density map
def drawDensity(img, prev, step=16):
    global density

    # Get the size of the image
    h, w = img.shape[:2]

    # Calculate the temporal difference
    td = np.int16(img) - np.int16(prev)
    
    # Calculate the size of the density map
    sx = int(w/step)
    sy = int(h/step)
    
    # If not all ready created then create the density map
    if density == None:
        density = np.zeros((sy, sx))
    
    # Create the mask to draw the density
    mask = np.zeros((h, w, 3))
    
    # Loop over image and calculate density
    for i in range(sx):
        # If out of bounds then skip
        if i*step+step > w:
            continue

        for j in range(sy):
            # If out of bounds then skip
            if j*step+step > h:
                continue
            
            # Select a block of the temporal difference
            b = td[j*step:j*step+step, i*step:i*step+step]

            # Create threshold image from block
            thresh = abs(b) > threshold
            
            # Count non zero elements in threshold image
            d = np.count_nonzero(thresh)
            
            # Calculate the percent of change
            density[j][i] += 0.5*d/(step*step)
            
            # Using calculated density draw the density map
            if density[j][i] < 0.5:
                v = 2.*density[j][i]

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
        
            # Decrease density over time
            density[j][i] -= 0.05
            
            # Bound density between 0 and 1
            if (density[j][i] > 1):
                density[j][i] = 1
                
            if (density[j][i] < 0):
                density[j][i] = 0

    # Return mask
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

    if args.saveFrames != None:
        save_video = True
        videoPath = args.saveFrames
    
    # Grab first frame
    ret, prev = cap.read()
   
    # Convert to gray scale
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Frame number
    frame_num = 0

    while True:
        # Grab second frame
        ret, img = cap.read()

        # If failed to get frame then exit loop
        if not ret:
            break

        # Convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Draw density
        d_img = drawDensity(gray, prevgray, block_size)
        combined = cv2.add(np.uint8(0.6*img), np.uint8(0.4*d_img))
        cv2.imshow('Density', combined)
        prevgray = gray

        # If save video true then write frame to specified location
        if save_video:
             cv2.imwrite('{}{:0>5d}.bmp'.format(videoPath, frame_num), combined)

        # Handle keyboard input
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break

        # Update frame count
        frame_num += 1

    # Clean Up View
    cv2.destroyAllWindows()
