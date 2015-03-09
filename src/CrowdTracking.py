import numpy as np
import cv2
import math
import sys

help = 'Usage: python3 CrowdTracking.py <video file>'

# Parameters
maxCorners = 100
qualityLevel = 0.05
minDistance = 20
blockSize = 7

# Track Buffer Length
track_len = 10

# Threshold
threshold = 50
distance = 5
distance2 = 2

# Find New Points
newPoints_on = True
tracks_on = True

def findGoodPoints(p0, p1, tracks, st):
    # Select good points
    for i,(new,old) in enumerate(zip(p1,p0)):
        if (len(tracks) > track_len):
            if (len(tracks[-track_len]) > i):
                a,b = new.ravel()
                c,d = tracks[-track_len][i].ravel()

                dx = c-a
                dy = d-b

                dx2 = dx*dx
                dy2 = dy*dy

                diff = math.sqrt(dx2 + dy2)

                if (diff < distance2):
                    st[i] = 0

    return st

def getNewPoints(img, img_old, points):
    t = np.int16(frame_gray) - np.int16(old_gray)

    #mask2 = np.ones_like(frame_gray)

    mask2 = abs(t) > threshold
    mask2 = mask2.astype('uint8')

    # Find good features to detect
    p_tmp = cv2.goodFeaturesToTrack(old_gray, mask = mask2, **feature_params)

    if points is None:
        return p_tmp

    if p_tmp != None:
        tmp = None
        for i,new in enumerate(p_tmp):
            found = False
            for j,old in enumerate(points):
                a,b = new.ravel()
                c,d = old.ravel()

                dx = c-a
                dy = d-b

                dx2 = dx*dx
                dy2 = dy*dy

                diff = math.sqrt(dx2 + dy2)

                if (diff < distance):
                    found = True
                    break;

            if not found:
                if tmp == None:
                    tmp = np.array([np.copy(new)])
                else:
                    tmp = np.concatenate((tmp, np.array([new])))

        if tmp != None:
            if len(tmp) > 0:
                points = np.concatenate((points, tmp))

    return points

def getDistance(a, b, c, d):
    dx = c-a
    dy = d-b

    dx2 = dx*dx
    dy2 = dy*dy

    return math.sqrt(dx2 + dy2)


if __name__ == '__main__':
    print(help)

    if len(sys.argv) < 2:
        cap = cv2.VideoCapture(-1)
    else:
        videoPath = sys.argv[1]
        cap = cv2.VideoCapture(videoPath)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = maxCorners,
                          qualityLevel = qualityLevel,
                          minDistance = minDistance,
                          blockSize = blockSize)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | 
                                 cv2.TERM_CRITERIA_COUNT, 
                                 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(2*maxCorners,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = getNewPoints(frame_gray, old_gray, None)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Store tracks
    tracks = []

    while(1):
        ret, frame = cap.read()

        if not ret:
            break;

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if newPoints_on:
            p0 = getNewPoints(frame_gray, old_gray, p0)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, 
                                               frame_gray, 
                                               p0, 
                                               None, 
                                               **lk_params)

        if p1 != None:
            #st = findGoodPoints(p0, p1, tracks, st)

            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                if tracks_on:
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            exit()
        if k == 10:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        tracks.append(good_new.reshape(-1,1,2))


    print("- Fixing Tracks")

    track_threshold = 50

    skip = []

    # Loop over tracks
    for i,points in enumerate(tracks):
        # Skip first track
        if i == 0:
            continue

        # Loop over previous track frame
        for j,point in enumerate(tracks[i-1]):
            if j in skip:
                continue

            print("Frame: ", i, " Point ", j)
            new_track = True
            if len(points) > j:
                new_track = False

                a,b = point.ravel()
                c,d = points[j].ravel()

                diff = getDistance(a, b, c, d)

                if (diff >= track_threshold):
                    new_track = True
                    for k,new_point in enumerate(points):
                        if k <= j:
                            continue

                        c,d = new_point.ravel()
                        diff = getDistance(a, b, c, d)
                        if (diff < track_threshold):
                            for l in range(i,len(tracks)):
                                if len(tracks[l]) - 1 < k:
                                    break

                                if len(tracks[l]) - 1 < j:
                                    break

                                tmp = tracks[l][k]
                                tracks[l][k] = tracks[l][j]
                                tracks[l][j] = tmp
                            new_track = False
                            break

            if new_track and i < len(tracks):
                print("- - New Track - -")
                for k in range(i,len(tracks)):
                    if len(tracks[k]) - 1 < j:
                        break

                    new_point = tracks[k][j]
                    tracks[k][j] = point
                    #skip.append(j)
                    #tracks[k] = np.concatenate((tracks[k], 
                    #                            np.array([new_point])))
                print(len(points))

    mask = np.zeros_like(old_frame)

    old_points = tracks[0]
    # draw the tracks
    for i,points in enumerate(tracks):
        for j,point in enumerate(old_points):
            if len(points) <= j:
                break
            a,b = point.ravel()
            c,d = points[j].ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[j].tolist(), 2)
        old_points = points

    img = cv2.add(np.uint8(0.5*old_frame), mask)
    cv2.imshow('frame',img)
    cv2.waitKey()

    cv2.destroyAllWindows()
    cap.release()
