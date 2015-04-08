import numpy as np
import cv2
import sys

help = 'Usage: python TestHaar.py <video file>'

if __name__ == '__main__':
    print(help)

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

        people = cascade.detectMultiScale(frame, 1.2, 5)

        for i in range(len(people)):
            (x,y,w,h) = people[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv2.imshow('Frame', frame)

        k = cv2.waitKey(7) & 0xff
        if k == 27:
            break
