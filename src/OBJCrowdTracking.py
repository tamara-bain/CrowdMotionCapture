import numpy as np
from scipy.interpolate import UnivariateSpline
import cv2
import math
import sys
import os
from Point import Point
from TrackInfo import TrackInfo
from Rectification import getRectification

help = 'Usage: python OBJCrowdTracking.py <video file>'

threshold = 50
block_size = 16

density_threshold = 0.1
density_growth = 0.5
density_decay = 0.25

draw_density = True

save_video = True


# Draw rectangles over areas detected by a cascade
def draw_detected(frame, detected, color):
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

	#Load Cascades
	full_body_cascade = cv2.CascadeClassifier('../Cascades/full_body_cascade.xml')
	upper_body_cascade = cv2.CascadeClassifier('../Cascades/haarcascade_upperbody.xml')
	profile_face_cascade = cv2.CascadeClassifier('../Cascades/haarcascade_profileface.xml')

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

	frame_num = 0
	while(1):
		# Grab new frame
		ret, frame = cap.read()

		# Check if read was successful
		if not ret:
			break

		 # Convert to grey scale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		density = getDensity(gray, prevgray, density, block_size)

		if draw_density:
			mask = drawDensity(frame, density, block_size)
			frame = cv2.add(np.uint8(0.6*frame), np.uint8(0.4*mask))

		detected = full_body_cascade.detectMultiScale(frame, 1.2, 5)
		np.append(detected, upper_body_cascade.detectMultiScale(frame, 1.2, 5))

		sp = removeDetected(detected, density, block_size)
		detected = detected[sp==1]

		draw_detected(frame, detected, (0, 255, 0))

		cv2.imshow('Frame', frame)

		if save_video:
			 cv2.imwrite('../Output/Video/{:0>5d}.bmp'.format(frame_num), frame)

		k = cv2.waitKey(7) & 0xff
		if k == 27:
			break

		frame_num += 1

	cap.release()
	cv2.destroyAllWindows()