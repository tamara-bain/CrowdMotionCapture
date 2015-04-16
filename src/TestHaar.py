##############################################################################
# TestHaar.py
# Code Written By: Michael Feist and Tamara Bain
#
# This is a very simple test scipt to to test our trained cascade.
#
# To run:
# python TestHaar.py <video file>
##############################################################################


import numpy as np
import cv2
import sys

help = 'Usage: python TestHaar.py <video file>'

if __name__ == '__main__':
    print(help)
    videoPath = ""
    
    # Get video file if given
    # Else open default camera
    if len(sys.argv) < 2:
        cap = cv2.VideoCapture(-1)
    else:
        videoPath = sys.argv[1]
        cap = cv2.VideoCapture(videoPath)


    cascade = cv2.CascadeClassifier('../haarcascade_16000_positives/cascade.xml')

    while(1):
        # Grab new frame
        ret, frame = cap.read()

        # Check if read was successful
        if not ret:
            break

        # Walk1 and Meet_WalkTogether need to be rotated for detection to work
        if "Walk" in  videoPath:
            rows, cols, channels = frame.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
            frame = cv2.warpAffine(frame,M,(cols,rows))

        # Classifier settings for Test videos
        classifier_dict = {
            "341-46_l" : lambda: cascade.detectMultiScale(frame, 1.01, 5, maxSize=(50,50)),
            "HUB" : lambda: cascade.detectMultiScale(frame, 1.3, 15, maxSize=(100,100)),
            "LRT" : lambda: cascade.detectMultiScale(frame, 1.03, 15, maxSize=(100,100)),
            "Walk" : lambda: cascade.detectMultiScale(frame, 1.3, 10, maxSize=(100,100))
        }

        # default value
        people = cascade.detectMultiScale(frame, 1.2, 5)
        for key, item in classifier_dict.iteritems():
            if key in videoPath:
                people = item()

        for i in range(len(people)):
            (x,y,w,h) = people[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        cv2.imshow('Frame', frame)

        k = cv2.waitKey(7) & 0xff
        if k == 27:
            break
