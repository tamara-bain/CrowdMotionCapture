##############################################################################
# Rectification.py
# Code Written By: Michael Feist
#
# Algorithm found in Section 2.7 of:
# R. Hartley and A. Zisserman, 'Multiple View Geometry',
# Cambridge University Publishers, 2nd ed. 2004
#
# Algorithm first takes two pairs of parallel lines calculates the Affine
# rectification. This takes the perspective image to an affine image.
#
# Next the algorithm takes two pairs of orthogonal lines calculates Metric
# rectification. This takes the affine image to a metric image.
#
# The user is then given a display of the final rectified image where they can
# scale, rotate, and translate the results.
#
# Finally, the resulting matrix is printed to the screen where the user can
# copy it for uses in other programs.
#
# To run:
# python Rectification.py [OPTIONS]
#
# For Help:
# python Rectification.py --help
##############################################################################

import argparse
import sys

import numpy as np
import math
import cv2

parser = argparse.ArgumentParser(
		prog='Rectification', 
		usage='python %(prog)s.py [options]')
parser.add_argument(
	'--video', 
	type=str, 
	help='path to input video')
parser.add_argument(
	'--image', 
	type=str, 
	help='path to input image')

dragging = False
m_x = m_y = 0
points = []

# Handles mouse events
def on_mouse(event,x,y,flags,param):
	global dragging, m_x, m_y
	if event == cv2.EVENT_LBUTTONDOWN:
		dragging = True
		points.append([x, y])
		m_x = x
		m_y = y
	elif event == cv2.EVENT_MOUSEMOVE:
		if dragging:
			m_x = x
			m_y = y
	elif event == cv2.EVENT_LBUTTONUP:
		dragging = False
		points.append([x, y])

def nothing(x):
    pass

def getLines(img):
	global m_x, m_y
	del points[:]

	# Create window and set mouse call back
	cv2.namedWindow('Get Lines')
	cv2.setMouseCallback('Get Lines', on_mouse)

	while(1):
		tmp = img.copy()

		# Draw lines
		for i in range(np.int(np.floor(len(points)/2))):
			a = points[2*i][0]
			b = points[2*i][1]
			c = points[2*i+1][0]
			d = points[2*i+1][1]
			cv2.line(tmp, (a,b), (c,d), (255, 0, 0), 2)

		if len(points) % 2 == 1:
			a = points[-1][0]
			b = points[-1][1]
			cv2.line(tmp, (a,b), (m_x,m_y), (255, 0, 0), 2)

		cv2.imshow('Get Lines', tmp)

		# Handle keyboard input
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			exit()
		if k == 10 or k == 13:
			break

	# Copy points to lines list
	lines = np.ones((len(points), 3))

	for i in range(len(points)):
		p = points[i]
		lines[i][0] = p[0]
		lines[i][1] = p[1]

	# Close window
	cv2.destroyWindow('Get Lines')

	return lines

def AffineRectification(lines):
	l1 = np.cross(lines[0],lines[1])
	l2 = np.cross(lines[2],lines[3])
	m1 = np.cross(lines[4],lines[5])
	m2 = np.cross(lines[6],lines[7])

	v1 = np.cross(l1, l2)
	v2 = np.cross(m1, m2)

	vanishL = np.cross(v1, v2)

	vanishL[0] = vanishL[0]/vanishL[2]
	vanishL[1] = vanishL[1]/vanishL[2]
	vanishL[2] = 1.0

	H1 = np.eye(3)
	H1[2][0] = vanishL[0]
	H1[2][1] = vanishL[1]
	H1[2][2] = vanishL[2]

	return H1


def MetricRectification(lines):
	l1 = np.cross(lines[0],lines[1])
	l2 = np.cross(lines[2],lines[3])
	m1 = np.cross(lines[4],lines[5])
	m2 = np.cross(lines[6],lines[7])

	l11 = l1[0]
	l12 = l1[1]
	l21 = l2[0]
	l22 = l2[1]
	m11 = m1[0]
	m12 = m1[1]
	m21 = m2[0]
	m22 = m2[1]

	M = np.zeros((2,2))
	b = np.zeros((2,1))

	M[0][0] = l11*m11
	M[0][1] = l11*m12 + l12*m11
	M[1][0] = l21*m21
	M[1][1] = l21*m22 + l22*m21

	b[0] = -l12*m12
	b[1] = -l22*m22

	x = np.linalg.solve(M, b)

	S  = np.eye(2)
	S[0][0] = x[0]
	S[0][1] = x[1]
	S[1][0] = x[1]

	U,D,V = np.linalg.svd(S)

	sqrtD = np.sqrt(D)
	U_T = np.transpose(U)

	A = U_T*sqrtD
	A = A*V

	H2 = np.eye(3)
	H2[0][0] = A[0][0]
	H2[0][1] = A[0][1]
	H2[1][0] = A[1][0]
	H2[1][0] = A[1][1]

	if H2[0][0] < 0:
		H2[0][0] = -H2[0][0]

	if H2[1][1] < 0:
		H2[1][1] = -H2[1][1]

	invH2 = np.linalg.inv(H2)

	return invH2

# Get rectification matrix
def getRectification(img):
	print('Instructions:')
	print('1: You need to pick two pairs of parallel lines.')
	print('   Each line requires two points. To get a point')
	print('   point you have to double click. Once the 4')
	print('   lines are selected press the return key.')
	print('2: Repeat the process but with two pairs of')
	print('   orthoganal lines.')

	# Get lines used for affine rectification
	lines = getLines(img)

	# Calculate affine rectification given the lines
	H1 = AffineRectification(lines)

	# Apply affine rectification to image
	affine_img_retification = cv2.warpPerspective(
		img, 
		H1, 
		(img.shape[1], img.shape[0]))

	# Get lines used for metric rectification
	lines = getLines(affine_img_retification)

	# Calculate metric rectification given the lines
	H2 = MetricRectification(lines)

	# Combine affine and metric rectification
	H3 = H2*H1

	# Create window and GUI for user to rotate, scale, and translate results
	cv2.namedWindow('image')
	cv2.createTrackbar('Angle','image',0,360,nothing)
	cv2.createTrackbar('Scale','image',0,5000,nothing)
	cv2.createTrackbar('X','image',-img.shape[0],img.shape[0],nothing)
	cv2.createTrackbar('Y','image',-img.shape[1],img.shape[1],nothing)

	while(1):
		# Get current value of track bars
		angle = cv2.getTrackbarPos('Angle','image')
		scale = cv2.getTrackbarPos('Scale','image')

		x = cv2.getTrackbarPos('X','image')
		y = cv2.getTrackbarPos('Y','image')

		s = scale/1000.
		r = math.radians(angle)

		# Create rotation matrix given the angle
		R = np.eye(3)
		R[0][0] = s*math.cos(r)
		R[0][1] = -s*math.sin(r)
		R[1][0] = s*math.sin(r)
		R[1][1] = s*math.cos(r)
		R[0][2] = x
		R[1][2] = y

		H = np.dot(H3, H1)

		# Calculate the center of the image
		center_1 = np.ones((3, 1))
		center_1[0][0] = img.shape[0]/2.
		center_1[1][0] = img.shape[1]/2.

		# Center image
		center_2 = np.dot(H, center_1)

		# Calculate translation matrix
		T = np.eye(3)

		T[0][2] = -center_2[0][0]/2.
		T[1][2] = -center_2[1][0]/2.

		# Apply translation to rotation matrix
		Transform = np.dot(R,T)
		# Apply transform to homography
		H = np.dot(Transform,H)

		# Warp image according to new homography
		img2 = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
		
		# Show warped image
		cv2.imshow('image', img2)

		# Handle key board input
		k = cv2.waitKey(30) & 0xFF
		if k == 27:
			exit()
		if k == 10 or k == 13:
			break

	cv2.destroyWindow('image')

	return H

# Run Program
if __name__ == '__main__':
	parser.print_help()

	# Parse Arguments
	args = parser.parse_args(sys.argv[1:])

	img = None

	# Get video file if given
	if args.video != None:
		cap = cv2.VideoCapture(args.video)
		ret, img = cap.read()
		if ret == None:
			print "Error getting frame from video."
			exit()

	# Get image
	if args.image != None:
		img = cv2.imread(args.image)
		
	if img == None:
		img = cv2.imread('../Images/Test/floor.jpg')

	H = getRectification(img)

	# Print homography matrix
	r, c = H.shape
	out = ''
	for i in range(r):
		out += ' '.join(map(str, H[i,:]))
		if i < r-1:
			out += '; '
	
	print(out)
