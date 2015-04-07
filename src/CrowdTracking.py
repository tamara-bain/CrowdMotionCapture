import numpy as np
from scipy.interpolate import UnivariateSpline
import cv2
import math
import sys
from Point import Point
from TrackInfo import TrackInfo
from Rectification import getRectification


help = 'Usage: python3 CrowdTracking.py <video file>'

# Parameters
maxCorners = 300
qualityLevel = 0.1
minDistance = 20
blockSize = 3

# Track Buffer Length
track_len = 120

# Threshold
threshold = 50
distance = 2.5
distance2 = 7.5

# Max Energy
max_energy = 7.0

# Find New Points
newPoints_on = True
tracks_on = False
draw_on = False


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
    # Calculate the temporal difference between two frames
    t = np.int16(frame_gray) - np.int16(old_gray)

    # If change is greater than threshold
    # Then set value to 1
    # Else set value to 0
    mask = abs(t) > threshold
    # Cast mask to unsigned byte
    mask = mask.astype('uint8')

    # Find good features to detect in areas of change
    p_tmp = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)

    # If no points were given then return new points
    if points is None:
        return p_tmp

    # If points were given and new points were found then merge
    # the two sets of points
    if p_tmp != None:
        tmp = None

        # Loop over new points
        for i,new in enumerate(p_tmp):

            # Flag checking if new point is close to an
            # existing point if so we don't add the new
            # point
            found = False

            # Loop over old points
            for j,old in enumerate(points):
                a,b = new.ravel()
                c,d = old.ravel()

                # Calculate distance between points
                diff = getDistance(a, b, c, d)

                # If distance is less than threshold
                # Then we don't add the new point
                if (diff < distance):
                    found = True
                    break;

            # If no existing point found then we add the new point
            if not found:
                # If first new point then we save point
                # Else we concatinate the new point with
                # the other new points being added
                if tmp == None:
                    tmp = np.array([np.copy(new)])
                else:
                    tmp = np.concatenate((tmp, np.array([new])))

        # If no new points found then no need to merge
        if tmp != None:
            if len(tmp) > 0:
                # Merge the old points with the new points
                points = np.concatenate((points, tmp))

    # Return the updated points
    return points

def fit_curve(track, deg = 1): 
    frames = track.getNumberOfFrames()
    X = np.zeros(frames)
    Y = np.zeros(frames)
    for j in range(0,frames):
        a,b = track.points[j].getCoords()
        X[j] = a
        Y[j] = b

    c = np.polyfit(X, Y, deg)
    return np.poly1d(c)

def remove_tracks_with_slope(track_info, deg = 1, m = 2.):
    n = len(track_info)
    slopes = np.zeros(n)
    for i in range(n):
        track = track_info[i]
        frames = track.getNumberOfFrames()
        X = np.zeros(frames)
        Y = np.zeros(frames)
        for j in range(0,frames):
            a,b = track.points[j].getCoords()
            X[j] = a
            Y[j] = b

        c = np.polyfit(X, Y, deg)
        slopes[i] = c[0]

    return reject_outliers(track_info, slopes, m)

def reject_outliers(result, data, m = 2.):
    n = len(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.

    for i in range(n):
        ind = n - i - 1
        if s[ind] >= m:
            result.pop(ind)

    return result

# Calculates the distance between two points
def getDistance(a, b, c, d):
    dx = c-a
    dy = d-b

    dx2 = dx*dx
    dy2 = dy*dy

    return math.sqrt(dx2 + dy2)


if __name__ == '__main__':
    print(help)

    # Get video file if given
    # Else open default camera
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

    # Take first frame
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Get rectification matrix
    print('Instructions:')
    print('1: You need to pick two pairs of parallel lines.')
    print('   Each line requires two points. To get a point')
    print('   point you have to double click. Once the 4')
    print('   lines are selected press the return key.')
    print('2: Repeat the process but with two pairs of')
    print('   orthoganal lines.')

    # H = getRectification(old_frame)
    H = np.eye(3)

    # Take second frame
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find new points, the function uses the difference
    # between the images to know where movement occured
    p0 = getNewPoints(frame_gray, old_gray, None)

    # Store tracks
    tracks = []
    # Keep track of the index of each point
    # This is needed because we are constantly adding
    # and removing points
    tracks_index = []

    # To keep track of the current frame
    frame_count = 0

    while(1):
        # Grab new frame
        ret, frame = cap.read()

        # Check if read was successful
        if not ret:
            break

        # Convert frame to gray scale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Grab new points that are not being tacked
        if newPoints_on:
            p0 = getNewPoints(frame_gray, old_gray, p0)

        # Calculate optical flow and new location of points
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, 
                                               frame_gray, 
                                               p0, 
                                               None, 
                                               **lk_params)

        # Create a mask image for drawing tracks
        mask = np.zeros_like(old_frame)

        if p1 != None:
            # Find points that don't move any more
            st = findGoodPoints(p0, p1, tracks, st)

            # Remove invalid points (stationary)
            good_new = p1[st==1]
            good_old = p0[st==1]

            # Draw the points
            if draw_on:
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    cv = i % maxCorners
                    frame = cv2.circle(frame,(a,b),5,color[cv].tolist(),-1)

            # Draw tracks
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
                        mask = cv2.line(mask, 
                                        (a,b),
                                        (c,d), 
                                        color[cv].tolist(), 
                                        2)

        # Combine current frame and mask 
        # (draws points and tracks on frame)
        img = cv2.add(frame,mask)

        # Show image
        cv2.imshow('frame',img)

        # Handle for keyboard input
        # ESC: Kills program
        # Return: stops capturing frames and moves onto next step
        k = cv2.waitKey(7) & 0xff
        if k == 27:
            exit()
        if k == 10:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        # Add new points to tracks
        if len(tracks) < 1:
            tracks.append(p0)
            tracks_index = [i for i in range(len(tracks))]
        else:
            # Find Valid Tracks for New Points

            # Get number of tracks
            i = len(tracks)

            # Duplicate last row of tracks
            tracks.append(tracks[i-1])

            # Keeps track of the number of points removed
            removed = 0

            # Loop through list of valid points this iteration
            for j in range(len(st)):

                # Get the index of the points
                index = j - removed

                # Check if index is in tracks_index
                if index not in tracks_index:
                    # If not add it to the end
                    tracks_index.append(index)

                # If point was removed due to being invalid
                # Also check if index of point is already in
                # tracks_index
                if st[j] == 0 and j <= max(tracks_index):

                    # Update removed count
                    removed = removed + 1

                    # Try to get the index of the point in tracks
                    try:
                        index = tracks_index.index(j)
                    except ValueError:
                        print("ValueError: no value ", j)
                        print(tracks_index)
                        exit()

                    # Update indices of points in tracks
                    tracks_index[index] = -1
                    for k in range(index+1,len(tracks_index)):
                        tracks_index[k] = tracks_index[k] - 1

            # Loop through points and add them to tracks
            for j,new in enumerate(p0):
                # Get index of point in tracks
                index = tracks_index.index(j)

                # If index found add point to track
                # Else we add it to the end as it is a
                # new track
                if index < len(tracks[i]):
                    tracks[i][index] = new
                else:
                    tracks[i] = np.concatenate((tracks[i], np.array([new])))

            # Resize track_index
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

        # Update frame count
        frame_count = frame_count+1




##### IN DEVELOMENT ######

    # Clean up Tracks
    # Kill tracks that move more than 20 pixels between frames

    # Keep track of tracks that ended
    tracks_ended = [0] * len(tracks[-1])

    # Loop through tracks
    for i in range(len(tracks)-1):
        # Get current number of points in track
        n = len(tracks[i])

        # Loop through points
        for j,point in enumerate(tracks[i]):
            # If track has ended skip new point
            if tracks_ended[j] == 1:
                continue

            # Calculate distance between current point
            # and next point
            a,b = point.ravel()
            c,d = tracks[i+1][j].ravel()

            diff = getDistance(a, b, c, d)

            # If point moved more than 20 pixels then we 
            # kill the current track
            if diff > 20:
                tracks_ended[j] = 1
                for k in range(i+1, len(tracks)):
                    tracks[k][j] = point


    # Store information for each track
    # See: TrackInfo() for more info
    track_info = [TrackInfo() for i in range(len(tracks[-1]))]

    dead_points = [0] * len(tracks[-1])
    dead_points_time = [0] * len(tracks[-1])
    for i in range(0,len(tracks)):
        # Update time
        for j in range(len(dead_points)):
            if dead_points[j] == 1:
                dead_points_time[j] = dead_points_time[j] + 1
            else:
                dead_points_time[j] = 0

        # Find points that don't move anymore
        # and mark as dead
        for j,point in enumerate(tracks[i]):
            track = track_info[j]
            track.start(i)

            if dead_points[j] == 1:
                track.end(i)
                continue

            a,b = point.ravel()

            track.addPoint(a,b)

            dead_point = True

            # Check if point moves
            for k in range(i+1, len(tracks)):
                c,d = tracks[k][j].ravel()

                diff = getDistance(a, b, c, d)

                if diff > distance:
                    dead_point = False
                    break

            if dead_point:
                dead_points[j] = 1


    # Apply Rectification to Points
    n = len(track_info)
    for i in range(n):
        track = track_info[i]
        track.applyMatrix(H)

    # Combine tracks?

    # Delete short tracks
    n = len(track_info)
    for i in range(n):
        index = n - i - 1
        track = track_info[index]
        diff = track.getDistanceTraveled()
        if diff < distance2:
            track_info.pop(index)

    # Calclate energy
    n = len(track_info)
    for i in range(n):
        index = n - i - 1
        e = track_info[index].calcMotionEnergy()
#        print("Track ", index, " has energy ", e)
        if e > max_energy:
            track_info.pop(index)

    # Does track stop
#    n = len(track_info)
#    for i in range(n):
#        index = n - i - 1
#        if track_info[index].doesMotionStop():
#            track_info.pop(index)

    # Calculate direction
    n = len(track_info)
    for i in range(n):
        track_info[i].calcDirection()

    # Remove Outlier Tracks
    track_info = remove_tracks_with_slope(track_info, 1, 1.75)
    track_info = remove_tracks_with_slope(track_info, 2, 1)

    # Group Tracks
    

    # draw the tracks
    old_frame_warp = cv2.warpPerspective(old_frame, H, (old_frame.shape[1], old_frame.shape[0]))
    mask = np.zeros_like(old_frame_warp)

    for i,track in enumerate(track_info):
        frames = track.getNumberOfFrames()
        for j in range(1,frames):
            a,b = track.points[j-1].getCoords()
            c,d = track.points[j].getCoords()

            cv = i % maxCorners
            cv2.line(mask, (a,b),(c,d), color[cv].tolist(), 2)


    img = cv2.add(np.uint8(0.5*old_frame_warp), mask)

    cv2.imshow('tracks', img)

    track_i = 0

    while(1):
        track = track_info[track_i]
        frames = track.getNumberOfFrames()
        X = np.zeros(frames)
        Y = np.zeros(frames)
        
        for i in range(0,frames):
            a,b = track.points[i].getCoords()
            X[i] = a
            Y[i] = b

        p2 = np.poly1d(np.polyfit(X, Y, 2))
        p1 = np.poly1d(np.polyfit(X, Y, 1))

        mask = np.zeros_like(old_frame_warp)

        for i in range(mask.shape[1]):
            j1 = p1(i)
            j2 = p2(i)

            if j1 != j1 or j2 != j2:
                continue

            if j1 >= 0 or j1 <= mask.shape[0]:
                j = int(j1)
                cv2.circle(mask, (i,j), 1, (255, 0, 0), 2)

            if j2 >= 0 or j2 <= mask.shape[0]:
                j = int(j2)
                cv2.circle(mask, (i,j), 1, (0, 255, 0), 2)
            

        for i in range(1,frames):
            a,b = track.points[i-1].getCoords()
            c,d = track.points[i].getCoords()

            cv = track_i % maxCorners
            cv2.line(mask, (a,b),(c,d), color[cv].tolist(), 2)


        img = cv2.add(np.uint8(0.5*old_frame_warp), mask)

        cv2.imshow('track', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            exit()
        if k == ord(' '):
            break
        if k == ord('a'):
            if track_i > 0:
                track_i = track_i - 1
        if k == ord('d'):
            if track_i < len(track_info) - 1:
                track_i = track_i + 1

    cv2.waitKey()

    # Reset Video
    cap.set(1, 0)

    frame_count = 0
    grid = 0
    while(1):
        ret, frame = cap.read()

        if not ret:
            frame_count = 0
            cap.set(1, 0)
            continue

        frame = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))
        mask = np.zeros_like(frame)
        mask2 = np.zeros((512, 512, 3))

        # draw the tracks
        for i,track in enumerate(track_info):
            if track.startFrame <= frame_count and track.endFrame > frame_count:
                frames = track.getNumberOfFrames()
                for j in range(0,10):
                    index = frame_count - track.startFrame - j

                    if index <= 0:
                        break

                    a,b = track.points[index-1].getCoords()
                    c,d = track.points[index].getCoords()

                    cv = i % maxCorners
                    cv2.line(mask, (a,b), (c,d), color[cv].tolist(), 2)

        grid_size = 5
        vector_size = 20
        for x in range(grid_size):
            for y in range(grid_size):
                p = Point(56+100*x,56+100*y)
                i = grid_size*y + x + grid*grid_size*grid_size

                if i >= len(track_info):
                    break

                track = track_info[i]

                a = c = p.x
                b = d = p.y

                if track.startFrame <= frame_count and track.endFrame > frame_count:
                    index = frame_count - track.startFrame

                    if index < len(track.direction):
                        d = track.direction[index]
                        dx = vector_size*d.x
                        dy = vector_size*d.y

                        c = p.x + dx
                        d = p.y + dy
                
                cv = int(i % maxCorners)
                a = int(a)
                b = int(b)
                c = int(c)
                d = int(d)
                cv2.line(mask2, (a,b), (c,d), color[cv].tolist(), 2)

        img = cv2.add(np.uint8(0.5*frame), mask)

        cv2.imshow('frame', img)
        cv2.imshow('direction', mask2)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            exit()
        if k == ord(' '):
            break
        if k == ord('a'):
            if grid > 0:
                grid = grid - 1
        if k == ord('d'):
            if grid*grid_size*grid_size < len(track_info):
                grid = grid + 1

        frame_count = frame_count + 1

    cv2.destroyAllWindows()
    cap.release()
