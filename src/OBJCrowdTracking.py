import numpy as np
from scipy.interpolate import UnivariateSpline
import cv2
import math
import sys
import os
from Point import Point
from OBJTrackInfo import OBJTrackInfo
from Rectification import getRectification

help = 'Usage: python OBJCrowdTracking.py <video file>'

num_colors = 100

threshold = 50
block_size = 16

density_threshold = 0.1
density_growth = 0.5
density_decay = 0.25

draw_density = False

save_video = False


# Draw rectangles over areas detected by a cascade
def drawDetected(frame, detected, color):
	for i in range(len(detected)):
		(x,y,w,h) = detected[i]
		cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

def drawDensity(img, density, step=16):
	# Get width and height
	h, w = img.shape[:2]

	# Get size of density array
	sy, sx = density.shape[:2]

	mask = np.zeros((h, w, 3))

	for i in range(sx):
		# If out of bounds skip
		if i*step+step > w:
			continue

		for j in range(sy):
			# If out of bounds skip
			if j*step+step > h:
				continue

			mask[j*step:j*step+step, i*step:i*step+step, 1] = 0
			mask[j*step:j*step+step, i*step:i*step+step, 2] = 255
			mask[j*step:j*step+step, i*step:i*step+step, 0] = 0

			if density[j][i] < 0.02:
				mask[j*step:j*step+step, i*step:i*step+step, 0] = 0
				mask[j*step:j*step+step, i*step:i*step+step, 1] = 255
				mask[j*step:j*step+step, i*step:i*step+step, 2] = 0
			elif density[j][i] < 0.5:
				mask[j*step:j*step+step, i*step:i*step+step, 0] = 0
				mask[j*step:j*step+step, i*step:i*step+step, 1] = 255
				mask[j*step:j*step+step, i*step:i*step+step, 2] = 255

	return mask

def getDensity(img, prev, density, step=16):
	# Get width and height
	h, w = img.shape[:2]

	# Get size of density array
	sy, sx = density.shape[:2]
	
	# Calculate temperal difference
	td = np.int16(img) - np.int16(prev)
	
	# Calculate density
	for i in range(sx):
		# If out of bounds skip
		if i*step+step > w:
			continue

		for j in range(sy):
			# If out of bounds skip
			if j*step+step > h:
				continue
			
			# Get block from temporal difference
			b = td[j*step:j*step+step, i*step:i*step+step]
			# Create threshold image of block
			thresh = abs(b) > threshold
			
			# Iterate over threshold image and caculate number of changes
			d = np.count_nonzero(thresh)
			
			# Density is the amount of change in a block divided by block size
			density[j][i] += density_growth*d/(step*step)
			
			# Decrease density over time
			density[j][i] -= density_decay
			
			# Set limits on density
			if (density[j][i] > 1):
				density[j][i] = 1
				
			if (density[j][i] < 0):
				density[j][i] = 0

	return density

def removeDetected(detected, density, frame, step=16):
	if len(detected) == 0:
		return detected

	dh, dw = density.shape[:2]
	ch = cw = 0
	sp = np.ones(len(detected))
	for k in range(len(detected)):
		(x,y,w,h) = detected[k]
		_x = int(np.ceil(x/step))
		_y = int(np.ceil(y/step))
		_w = int(np.floor(w/step))
		_h = int(np.floor(h/step))

		while _x+_w >= dw:
			_w -= 1

		while _y+_h >= dh:
			_h -= 1

		d = np.count_nonzero(density[_y:_y+_h, _x:_x+_w])

		d = float(d)/((_w+1)*(_h+1))

		if d < density_threshold:
			sp[k] = 0

	return sp

def cleanTracks(tracks):
	sp = np.ones(len(tracks), dtype='int')

	for i,track in enumerate(tracks):
		if not track.active():
			frames = len(track.points)
			if frames < 10:
				sp[i] = 0

	return sp

def trackWithFlow(prevgray, gray, x, y, w, h):
	cx = cy = 0.0
	for i in range(5):
		if x < 0 or y < 0 or x+w >= gray.shape[1] or y+h >= gray.shape[0]:
			break

		sub_prevgray = prevgray[y:y+h, x:x+w]
		sub_gray = gray[y:y+h, x:x+w]
		flow = cv2.calcOpticalFlowFarneback(sub_prevgray, sub_gray, 0.5, 1, 3, 15, 3, 5, 1)

		cx += np.mean(flow[:,:,0])
		cy += np.mean(flow[:,:,1])

	return (cx, cy)

def updateTracks(tracks, detected, prevgray, gray, frame):
	for i,track in enumerate(tracks):
		if track.active():
			#(cx, cy) = trackWithFlow(prevgray, gray, track.bx, track.by, track.bw, track.bh)

			#track.x += cx
			#track.y += cy

			track.lastFound += 1

			if track.lastFound > 5:
				track.end(frame)

	for k in range(len(detected)):
		(x,y,w,h) = detected[k]

		mxd = x+(w/2.)
		myd = y+(h/2.)

		found = -1
		min_distance = float('inf')
		for i,track in enumerate(tracks):
			if not track.active():
				continue

			wDiff = float(track.bw)/w
			hDiff = float(track.bh)/h

			if wDiff < 0.8 or wDiff > 1.2:
				continue

			if hDiff < 0.8 or hDiff > 1.2:
				continue
			diffx = mxd - track.x
			diffy = myd - track.y

			distance = np.sqrt(diffx*diffx + diffy*diffy)

#			print("mxd " + str(mxd))
#			print("myd " + str(myd))
#			print("track.x " + str(track.x))
#			print("track.y " + str(track.y))
#			print("wDiff " + str(wDiff))
#			print("hDiff " + str(hDiff))
#			print("diffx " + str(diffx))
#			print("diffy " + str(diffy))
#			print("distance " + str(distance))
#			print((x,y,w,h))
#			print((track.bx,track.by,track.bw,track.bh))

			if distance < min_distance and distance < 20:
				min_distance = distance
				found = i

		if found >= 0:
			tracks[found].x = mxd
			tracks[found].y = myd
			tracks[found].addPoint(mxd, myd)
			tracks[found].setBoundingBox(x, y, w, h)
			tracks[found].lastFound = 0
		else:
			track = OBJTrackInfo()
			track.x = mxd
			track.y = myd
			track.addPoint(mxd, myd)
			track.setBoundingBox(x, y, w, h)
			track.lastFound = 0
			track.start(frame)
			tracks.append(track)

	return tracks

def drawTracks(frame, tracks, frame_count):
	mask = np.zeros_like(frame)
	
	# draw the tracks
	for i,track in enumerate(tracks):
		cv = i % num_colors

		if track.active():
			cv2.rectangle(mask, (int(track.bx), int(track.by)), (int(track.bx+track.bw), int(track.by+track.bh)), color[cv].tolist(), 3)
			cv2.circle(mask, (int(track.x), int(track.y)), 5, color[cv].tolist(), 3)

		frames = len(track.points)
		if frames <= 2:
			continue

		for j in range(1,frames):
			a,b = track.points[j-1].getCoords()
			c,d = track.points[j].getCoords()

			cv2.line(mask, (int(a),int(b)), (int(c),int(d)), color[cv].tolist(), 2)

	return cv2.add(np.uint8(0.5*frame), np.uint8(0.5*mask))

if __name__ == '__main__':
	print(help)

	out_path = '../Output/output.out'

	# Get video file if given
	# Else open default camera
	if len(sys.argv) < 2:
		cap = cv2.VideoCapture(-1)
	else:
		videoPath = sys.argv[1]
		cap = cv2.VideoCapture(videoPath)
		out_name = os.path.split(videoPath)[1]
		out_path = '../Output/' + os.path.splitext(out_name)[0] + '.out'

	# Create some random colors
	color = np.random.randint(0,255,(num_colors,3))

	#Load Cascades
	full_body_cascade = cv2.CascadeClassifier('../Cascades/full_body_cascade.xml')
	#upper_body_cascade = cv2.CascadeClassifier('../Cascades/haarcascade_upperbody.xml')

	 # Grab first frame
	ret, prev = cap.read()

	# Convert to grey scale
	prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

	# Get width and height of image
	h, w = prevgray.shape[:2]
	
	# Calculate size of density array
	dw = int(w/block_size)
	dh = int(h/block_size)
	
	# Allocate memory for density array
	density = np.zeros((dh, dw))

	tracks = []

	frame_num = 0
	while(1):
		# Grab new frame
		ret, frame = cap.read()

		# Check if read was successful
		if not ret:
			break

		# Convert to grey scale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Calculate Density
		density = getDensity(gray, prevgray, density, block_size)

		# Draw Density
		if draw_density:
			mask = drawDensity(frame, density, block_size)
			frame = cv2.add(np.uint8(0.6*frame), np.uint8(0.4*mask))

		# Find objects in the scene (Example: Full Body)
		detected = full_body_cascade.detectMultiScale(frame, 1.2, 5)
		#np.append(detected, upper_body_cascade.detectMultiScale(frame, 1.2, 5))

		# Find detected areas that are not moving and remove them
		sp = removeDetected(detected, density, block_size)
		detected = detected[sp==1]

		# Connect new detected objects with previously tracked
		tracks = updateTracks(tracks, detected, prevgray, gray, frame_num)

		# Draw bounding box around detected
		#drawDetected(frame, detected, (0, 255, 0))

		# Remove short tracks
		if len(tracks) > 1:
			sp = cleanTracks(tracks)

			n = len(sp)
			for i in range(n):
				index = n - i - 1

				if sp[index] == 0:
					del tracks[index]
		
		# Draw Tracks
		frame = drawTracks(frame, tracks, frame_num)

		cv2.imshow('Frame', frame)

		if save_video:
			 cv2.imwrite('../Output/Video/{:0>5d}.bmp'.format(frame_num), frame)

		k = cv2.waitKey(7) & 0xff
		if k == 27:
			break

		frame_num += 1

	cap.release()
	cv2.destroyAllWindows()