import numpy as np
import math
import cv2
import sys

help = 'Usage: python3 Rectification.py <image file>'


dragging = False
m_x = m_y = 0
points = []
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

	cv2.namedWindow('Get Lines')
	cv2.setMouseCallback('Get Lines', on_mouse)

	while(1):
		tmp = img.copy()

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

		k = cv2.waitKey(30) & 0xff
		if k == 27:
			exit()
		if k == 10 or k == 13:
			break

	lines = np.ones((len(points), 3))

	for i in range(len(points)):
		p = points[i]
		lines[i][0] = p[0]
		lines[i][1] = p[1]

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

def getRectification(img):
	# Get rectification matrix
	print('Instructions:')
	print('1: You need to pick two pairs of parallel lines.')
	print('   Each line requires two points. To get a point')
	print('   point you have to double click. Once the 4')
	print('   lines are selected press the return key.')
	print('2: Repeat the process but with two pairs of')
	print('   orthoganal lines.')
#	cv2.imshow('Original', img)

	lines = getLines(img)

#	lines = np.ones((8, 3))
#	lines[0][0] = 220.
#	lines[1][0] = 499.
#	lines[2][0] = 257.
#	lines[3][0] = 527.
#	lines[4][0] = 160.
#	lines[5][0] = 494.
#	lines[6][0] = 214.
#	lines[7][0] = 449.
#
#	lines[0][1] = 299.
#	lines[1][1] = 36. 
#	lines[2][1] = 388.
#	lines[3][1] = 98. 
#	lines[4][1] = 169.
#	lines[5][1] = 256.
#	lines[6][1] = 29. 
#	lines[7][1] = 83. 

	H1 = AffineRectification(lines)

	affine_img_retification = cv2.warpPerspective(img, H1, (img.shape[1], img.shape[0]))
#	cv2.imshow('Affine Retification', affine_img_retification)

	lines = getLines(affine_img_retification)

#	lines = np.ones((8, 3))
#	lines[0][0] = 216.
#	lines[1][0] = 339.
#	lines[2][0] = 424.
#	lines[3][0] = 249.
#	lines[4][0] = 307.
#	lines[5][0] = 340.
#	lines[6][0] = 215.
#	lines[7][0] = 425.
#
#	lines[0][1] = 91. 
#	lines[1][1] = 123.
#	lines[2][1] = 78. 
#	lines[3][1] = 169.
#	lines[4][1] = 50. 
#	lines[5][1] = 123.
#	lines[6][1] = 93. 
#	lines[7][1] = 78. 

	H2 = MetricRectification(lines)
	H3 = H2*H1

#	metric_img_retification = cv2.warpPerspective(affine_img_retification, H3, (img.shape[1], img.shape[0]))
#	cv2.imshow('Metric Retification', metric_img_retification)

	cv2.namedWindow('image')
	cv2.createTrackbar('Angle','image',0,360,nothing)
	cv2.createTrackbar('Scale','image',0,5000,nothing)
	cv2.createTrackbar('X','image',-img.shape[0],img.shape[0],nothing)
	cv2.createTrackbar('Y','image',-img.shape[1],img.shape[1],nothing)

	while(1):
		angle = cv2.getTrackbarPos('Angle','image')
		scale = cv2.getTrackbarPos('Scale','image')

		x = cv2.getTrackbarPos('X','image')
		y = cv2.getTrackbarPos('Y','image')

		s = scale/1000.
		r = math.radians(angle)

		R = np.eye(3)
		R[0][0] = s*math.cos(r)
		R[0][1] = -s*math.sin(r)
		R[1][0] = s*math.sin(r)
		R[1][1] = s*math.cos(r)
		R[0][2] = x
		R[1][2] = y

		H = np.dot(H3, H1)

		center_1 = np.ones((3, 1))
		center_1[0][0] = img.shape[0]/2.
		center_1[1][0] = img.shape[1]/2.

		center_2 = np.dot(H, center_1)

		T = np.eye(3)

		T[0][2] = -center_2[0][0]/2.
		T[1][2] = -center_2[1][0]/2.

		R = np.dot(R,T)
		H = np.dot(R,H)

		img2 = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
		
		cv2.imshow('image', img2)

		k = cv2.waitKey(30) & 0xFF
		if k == 27:
			exit()
		if k == 10 or k == 13:
			break

	cv2.destroyWindow('image')

	return H

# Test
if __name__ == '__main__':
	if len(sys.argv) < 2:
		print(help)
		img = cv2.imread('../Images/Test/floor.jpg')
	else:
		img = cv2.imread(sys.argv[1])

	getRectification(img)

	cv2.waitKey()
