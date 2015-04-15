import argparse
import sys
import os

import numpy as np
from scipy.interpolate import UnivariateSpline
import cv2
import math

from Point import Point
from OBJTrackInfo import OBJTrackInfo

num_colors = 100

threshold = 50
block_size = 16

density_threshold = 0.15
density_growth = 0.5
density_decay = 0.25

draw_density = False

save_video = False

H = np.eye(3)

parser = argparse.ArgumentParser(
		prog='OBJCrowdTracking', 
		usage='python %(prog)s.py [options]')
parser.add_argument(
	'--video', 
	type=str, 
	help='path to input video.')
parser.add_argument(
	'--blockSize', 
	type=int, 
	help='size of blocks used for density.')
parser.add_argument(
	'--dThresh', 
	type=float, 
	help='threshold of needed density for accepting objects.')
parser.add_argument(
	'--dDraw', 
	type=bool, 
	help='if true the density will be drawn over the image.')
parser.add_argument(
	'--saveFrames', 
	type=str, 
	help='path to save images.')
parser.add_argument(
	'--homography', 
	type=str, 
	help='numpy 3x3 homography matrix.')
parser.add_argument(
	'--homographyPath', 
	type=str, 
	help='reads numpy 3x3 homography matrix from file.')
parser.add_argument(
	'--output', 
	type=str, 
	help='path to output tracks.')


# Write tracks to file
def outputTracks(tracks, outputPath):
	# Open file
	f = open(outputPath, 'w')

	# Write tracks start frame and frame number:
	# Example:
	# 1 20
	# 5 5
	#
	# So track 1 will start at frame 1 and go for 20 frames
	# and track 2 will start at frame 5 and go for 5 frames
	for i,track in enumerate(tracks):
		frames = track.getNumberOfFrames()
		start = track.startFrame
		f.write(str(start) + ' ' + str(frames) + '\n')

	# Write blank line
	f.write('\n')

	# Write positions of tracks
	for i,track in enumerate(tracks):
		frames = track.getNumberOfFrames()
		for j in range(0,frames):
			a,b = track.points[j].getCoords()
			f.write(str(int(a)) + ' ' + str(int(b)) + '\n')

# Draw rectangles over areas detected by a cascade
def drawDetected(frame, detected, color):
	for i in range(len(detected)):
		(x,y,w,h) = detected[i]
		cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

# Create density mask
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

# Calculate the density
# The density is based off how much the crowd is moving
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

# Using the density remove detected objects that are not moving
# as these most likely don't belong to people in the crowd
def removeDetected(detected, density, frame, step=16):
	# If not people then detected then there is nothing to
	# remove
	if len(detected) == 0:
		return detected

	# Get size of density array
	dh, dw = density.shape[:2]

	# Create binary array indicating what elements in detected should be
	# removed. '1' means we should keep the detected object and '0' means
	# we should remove the object
	sp = np.ones(len(detected))

	# Loop over detected
	for k in range(len(detected)):
		# Get bounding box of detected
		(x,y,w,h) = detected[k]

		# Map the bounding box to the density array
		_x = int(np.ceil(x/step))
		_y = int(np.ceil(y/step))
		_w = int(np.floor(w/step))
		_h = int(np.floor(h/step))

		# Shrink the width if out of bounds
		while _x+_w >= dw:
			_w -= 1

		# Shrink the height if out of bounds
		while _y+_h >= dh:
			_h -= 1

		# Calculate the average density
		d = np.mean(np.mean(density[_y:_y+_h, _x:_x+_w]))

		# If density less than some threshold then we remove the detected
		# object
		if d < density_threshold:
			sp[k] = 0

	return sp


def cleanTracks(tracks):
	# Create binary array indicating what tracks should be removed. '1' means 
	# we should keep the track and '0' means we should remove the track
	sp = np.ones(len(tracks))

	# Iterate over tracks
	for i,track in enumerate(tracks):
		# If not active and has less than certain # of frames then we 
		# should remove the track
		if not track.active():
			frames = len(track.points)
			if frames < 100:
				sp[i] = 0

	return sp

# Old Unsused Code
# Tried to get the new location of bounding box using optic flow
def trackWithFlow(prevgray, gray, x, y, w, h):
	cx = cy = 0.0
	for i in range(5):
		if x < 0 or y < 0 or x+w >= gray.shape[1] or y+h >= gray.shape[0]:
			break

		sub_prevgray = prevgray[y:y+h, x:x+w]
		sub_gray = gray[y:y+h, x:x+w]
		flow = cv2.calcOpticalFlowFarneback(
			sub_prevgray, sub_gray, 0.5, 1, 3, 15, 3, 5, 1)

		cx += np.mean(flow[:,:,0])
		cy += np.mean(flow[:,:,1])

	return (cx, cy)

def updateTracks(tracks, detected, prevgray, gray, frame):
	for i,track in enumerate(tracks):
		if track.active():
			track.lastFound += 1

			if track.lastFound > 100:
				track.end(frame)

	found_tracks = np.zeros(len(detected), dtype=int)
	found_tracks_distance = np.zeros(len(detected))
	for k in range(len(found_tracks)):
		found_tracks[k] = -1
		found_tracks_distance[k] = float('inf')

	for k in range(len(detected)):
		(x,y,w,h) = detected[k]

		mxd = x+(w/2.)
		myd = y+(h/2.)

		for i,track in enumerate(tracks):
			thres_dis = 20+track.lastFound/2.

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

			if distance < found_tracks_distance[k] and distance < thres_dis:
				# Check if a deteceted object already clamed block
				# if so break tie
				update = True
				for j in range(k):
					if found_tracks[j] == i:
						if found_tracks_distance[j] < distance:
							update = False
						else:
							found_tracks[j] = -1
							found_tracks_distance[j] = float('inf')
				if update:
					found_tracks_distance[k] = distance
					found_tracks[k] = i


		if found_tracks[k] >= 0:
			step = tracks[found_tracks[k]].lastFound
			stepx = (mxd - tracks[found_tracks[k]].x)/step
			stepy = (myd - tracks[found_tracks[k]].y)/step
			for i in range(step):
				mx = (i+1)*stepx + tracks[found_tracks[k]].x
				my = (i+1)*stepy + tracks[found_tracks[k]].y
				tracks[found_tracks[k]].addPoint(mx, my)

			tracks[found_tracks[k]].x = mxd
			tracks[found_tracks[k]].y = myd
			
			tracks[found_tracks[k]].setBoundingBox(x, y, w, h)
			tracks[found_tracks[k]].lastFound = 0
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
		# Get color index
		cv = i % num_colors

		# If track is still active then draw a bounding box and
		# a circle indicating the center
		if track.active():
			cv2.rectangle(
				mask, 
				(int(track.bx), int(track.by)), 
				(int(track.bx+track.bw), int(track.by+track.bh)), 
				color[cv].tolist(), 
				3)
			cv2.circle(
				mask, 
				(int(track.x), int(track.y)), 
				5, 
				color[cv].tolist(), 
				3)

		# If frames captured less than 2 then don't bother drawing track
		frames = len(track.points)
		if frames < 2:
			continue

		# Draw the track as a bunch of lines connecting the captured locations`
		for j in range(1,frames):
			a,b = track.points[j-1].getCoords()
			c,d = track.points[j].getCoords()

			cv2.line(
				mask, 
				(int(a),int(b)), 
				(int(c),int(d)), 
				color[cv].tolist(), 
				2)

	return mask

if __name__ == '__main__':
	parser.print_help()

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

	if args.dThresh != None:
		density_threshold = args.dThresh

	if args.dDraw != None:
		draw_density = args.dDraw

	if args.saveFrames != None:
		save_video = True
		videoPath = args.saveFrames

	if args.homography != None:
		# Example Format:
		# 1.26 0.70 -12.30; -0.64 2.08 79.20; -0.00 0.01 1.0
		H = np.matrix(args.homography)

	if args.homographyPath != None:
		with open(args.homographyPath) as f:
			content = f.readlines()[0]
		H = np.matrix(content)

	# Create some random colors
	color = np.random.randint(0,255,(num_colors,3))

	#Load Cascades
	full_body_cascade = cv2.CascadeClassifier(
		'../Cascades/full_body_cascade.xml')
	upper_body_cascade = cv2.CascadeClassifier(
		'../Cascades/haarcascade_upperbody.xml')

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

	# List to store tracks
	tracks = []

	# Keeps track of the current frame
	frame_num = 0
	while(1):
		# Grab new frame
		ret, frame = cap.read()

		# Check if read was successful
		if not ret:
			break

		# Update previous frame
		prev = frame

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
		mask = drawTracks(frame, tracks, frame_num)
		frame = cv2.add(np.uint8(0.5*frame), np.uint8(0.5*mask))

		# Show frame
		cv2.imshow('Frame', frame)

		# If save video true then write frame to specified location
		if save_video:
			 cv2.imwrite('{}{:0>5d}.bmp'.format(videoPath, frame_num), frame)

		# Handle keyboard input
		k = cv2.waitKey(7) & 0xff

		# If esc key pressed then exit loop
		if k == 27:
			break

		# Increment frame count
		frame_num += 1

	# Apply homography to each track and end track in still active
	for i,track in enumerate(tracks):
		track.applyMatrix(H)
		track.end(frame_num)

	# Clean up short tracks
	if len(tracks) > 1:
		sp = cleanTracks(tracks)

		n = len(sp)
		for i in range(n):
			index = n - i - 1

			if sp[index] == 0:
				del tracks[index]

	# Draw tracks on top of previous frame with applied homography
	tracks_mask = drawTracks(prev, tracks, frame_num)
	warped = cv2.warpPerspective(prev, H, (prev.shape[1], prev.shape[0]))
	img = cv2.add(np.uint8(warped), np.uint8(tracks_mask))
	cv2.imshow('Tracks', img)

	if args.output != None:
		outputTracks(tracks, args.output)

	# Wait for user to press a key
	cv2.waitKey()

	# Clean up scene
	cap.release()
	cv2.destroyAllWindows()