import numpy as np
import cv2
import math
import sys

help = 'Usage: python3 CrowdTracking.py <video file>'

# Parameters
maxCorners = 100
qualityLevel = 0.05
minDistance = 50
blockSize = 7

# Track Buffer Length
track_len = 60

# Threshold
threshold = 50
distance = 2.5
distance2 = 20

# Find New Points
newPoints_on = True
tracks_on = False

def findGoodPoints(p0, p1, tracks, st):
    # Select good points
    if (len(tracks) > track_len):
        for i,(new,old) in enumerate(zip(p1,p0)):
            if (len(tracks[-track_len]) > i):
                a,b = new.ravel()
                c,d = tracks[-track_len][i].ravel()

                diff = getDistance(a, b, c, d)

                if (diff < distance2):
                    st[i] = 0

    return st

def getNewPoints(img, img_old, points):
    t = np.int16(frame_gray) - np.int16(old_gray)

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

                diff = getDistance(a, b, c, d)

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
    color = np.random.randint(0,255,(maxCorners,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = getNewPoints(frame_gray, old_gray, None)

    # Store tracks
    tracks = []
    tracks_index = []

    frame_count = 0
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

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        if p1 != None:
            st = findGoodPoints(p0, p1, tracks, st)

            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                cv = i % maxCorners
                frame = cv2.circle(frame,(a,b),5,color[cv].tolist(),-1)

            if tracks_on and len(tracks) > 2:
                upper = len(tracks)
                lower = 0
                if upper > 20:
                    lower = upper - 20

                number_of_points = len(tracks[-1])

                for i in range(lower+1, upper):
                    for j in range(number_of_points):
                        if j >= len(tracks[i-1]):
                            break

                        a,b = tracks[i][j].ravel()
                        c,d = tracks[i-1][j].ravel()
                        cv = i % maxCorners
                        mask = cv2.line(mask, (a,b),(c,d), color[cv].tolist(), 2)

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

        # Find add new points to tracks
        if len(tracks) < 1:
            tracks.append(p0)
            tracks_index = [i for i in range(len(tracks))]
        else:
            # Find Valid Tracks for New Points
            i = len(tracks)
            tracks.append(tracks[i-1])

            removed = 0
            for j in range(len(st)):
                index = j - removed
                if index not in tracks_index:
                    tracks_index.append(index)

                if st[j] == 0 and j <= max(tracks_index):
                    removed = removed + 1
                    try:
                        index = tracks_index.index(j)
                    except ValueError:
                        print("ValueError: no value ", j)
                        print(tracks_index)
                        exit()

                    tracks_index[index] = -1
                    for k in range(index+1,len(tracks_index)):
                        tracks_index[k] = tracks_index[k] - 1

            for j,new in enumerate(p0):
                index = tracks_index.index(j)
                if index < len(tracks[i]):
                    tracks[i][index] = new
                else:
                    tracks[i] = np.concatenate((tracks[i], np.array([new])))

            if len(tracks[i]) < len(tracks_index):
                while max(tracks_index) >= len(p0):
                    tracks_index.pop(-1)

            # Split up tracks for drawing
            if tracks_on:
                for i in range(len(tracks)-1):
                    n = len(tracks[i])
                    new_points = np.zeros(n)
                    for j,point in enumerate(tracks[i]):
                        a,b = point.ravel()
                        c,d = tracks[i+1][j].ravel()

                        diff = getDistance(a, b, c, d)

                        if diff > 20:
                            new_points[j] = 1
                            for k in range(i+1, len(tracks)):
                                tracks[k] = np.concatenate((tracks[k], 
                                    np.array([tracks[k][j]])))
                                tracks[k][j] = point

        frame_count = frame_count+1




##### IN DEVELOMENT ######

    # Clean up Tracks
    print(len(tracks[-1]))
    mask2 = np.zeros_like(old_frame)

    new_points = [0] * len(tracks[-1])

    # Split tracks that move more than 20 pixels between frames
    for i in range(len(tracks)-1):
        n = len(tracks[i])
        for j,point in enumerate(tracks[i]):
            if new_points[j] == 1:
                continue

            a,b = point.ravel()
            c,d = tracks[i+1][j].ravel()

            diff = getDistance(a, b, c, d)

            if diff > 20:
                new_points[j] = 1
                #new_points.append(0)
                for k in range(i+1, len(tracks)):
                    #tracks[k] = np.concatenate((tracks[k], np.array([tracks[k][j]])))
                    tracks[k][j] = point

    print(len(tracks[-1]))

    dead_points = [0] * len(tracks[-1])
    dead_points_time = [0] * len(tracks[-1])
    for i in range(1,len(tracks)):
        # Update time
        for j in range(len(dead_points)):
            if dead_points[j] == 1:
                dead_points_time[j] = dead_points_time[j] + 1
            else:
                dead_points_time[j] = 0

        # Find points that don't move anymore
        # and mark as dead
        for j,point in enumerate(tracks[i]):
            if dead_points[j] == 1:
                continue

            a,b = point.ravel()

            dead_point = True

            for k in range(i+1, len(tracks)):
                c,d = tracks[k][j].ravel()

                diff = getDistance(a, b, c, d)

                if diff > distance/2.:
                    dead_point = False
                    break

            if dead_point:
                dead_points[j] = 1

#        # Draw Circles Where Dead
#        for j in range(len(dead_points)):
#            if dead_points[j] == 1:
#                a,b = tracks[i][j].ravel()
#                cv = j % maxCorners
#                mask2 = cv2.circle(mask2,(a,b),5,color[cv].tolist(),-1)

    # Group Tracks

    mask = np.zeros_like(old_frame)

    old_points = tracks[0]
    # draw the tracks
    for i,points in enumerate(tracks):
        if i == 0:
            continue

        for j,point in enumerate(old_points):
            if len(points) <= j:
                break
            a,b = point.ravel()
            c,d = points[j].ravel()

            color_value = j % maxCorners
            mask = cv2.line(mask, (a,b),(c,d), color[color_value].tolist(), 2)
        old_points = points

    img = cv2.add(np.uint8(0.5*old_frame), mask)
    img = cv2.add(img, mask2)

    cv2.imshow('tracks',img)
    cv2.waitKey()

    # Reset Video
    cap.set(1, 0)

    while(1):
        ret, frame = cap.read()

        if not ret:
            break;

        #img = cv2.add(frame,mask)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            exit()
        if k == 10:
            break

    cv2.destroyAllWindows()
    cap.release()
