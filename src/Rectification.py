import numpy as np
import math
import cv2
import sys

help = 'Usage: python3 Rectification.py <image file>'

points = []
def on_mouse(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		print('Mouse Position: ', x, ', ', y)
		points.append([x, y])

def nothing(x):
    pass

def getLines(img):
	del points[:]

	cv2.namedWindow('Get Lines')
	cv2.setMouseCallback('Get Lines', on_mouse)

	while(1):
		cv2.imshow('Get Lines', img)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			exit()
		if k == 10:
			break

	lines = np.ones((len(points), 3))

	for i in range(len(points)):
		p = points[i]
		lines[i][0] = p[0]
		lines[i][1] = p[1]

	cv2.destroyWindow('Get Lines')

	return lines

def AffineRectification(lines):
	print(lines)

	l1 = np.cross(lines[0],lines[1])
	l2 = np.cross(lines[2],lines[3])
	m1 = np.cross(lines[4],lines[5])
	m2 = np.cross(lines[6],lines[7])

	v1 = np.cross(l1, l2)
	v2 = np.cross(m1, m2)

	vanishL = np.cross(v1, v2)

	vanishL[0][0] = vanishL[0][0]/vanishL[0][2]
	vanishL[0][1] = vanishL[0][1]/vanishL[0][2]
	vanishL[0][2] = 1.0

	H1 = np.eye(3)
	H1[2][0] = vanishL[0][0]
	H1[2][1] = vanishL[0][1]
	H1[2][2] = vanishL[0][2]

	return H1


def MetricRectification(lines):
	print(lines)

	l1 = np.cross(lines[0],lines[1])
	l2 = np.cross(lines[2],lines[3])
	m1 = np.cross(lines[4],lines[5])
	m2 = np.cross(lines[6],lines[7])

	l11 = l1[0][0]
	l12 = l1[0][1]
	l21 = l2[0][0]
	l22 = l2[0][1]
	m11 = m1[0][0]
	m12 = m1[0][1]
	m21 = m2[0][0]
	m22 = m2[0][1]

	M = np.zeros((2,2))
	b = np.zeros((2,1))

	M[0][0] = l11*m11
	M[0][1] = l11*m12 + l12*m11
	M[1][0] = l21*m21
	M[1][1] = l21*m22 + l22*m21

	b[0][0] = -l12*m12
	b[1][0] = -l22*m22

	x = np.linalg.solve(M, b)

	S  = np.eye(2)
	S[0][0] = x[0][0]
	S[0][1] = x[1][0]
	S[1][0] = x[1][0]

	U,D,V = np.linalg.svd(S)

	sqrtD = np.sqrt(D)
	U_T = np.transpose(U)

	A = U_T*sqrtD
	A = A*V

	print(A)

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

#	scale = 2.0
#	angle = -0.2
#	R = np.eye(3)
#	R[0][0] = scale*math.cos(angle)
#	R[0][1] = -scale*math.sin(angle)
#	R[1][0] = scale*math.sin(angle)
#	R[1][1] = scale*math.cos(angle)
#
#	H = R*invH2
#
#	print(R)
#	print(H)

	return invH2


# Test
if __name__ == '__main__':
	if len(sys.argv) < 2:
		print(help)
		img = cv2.imread('../Images/Test/floor.jpg')
	else:
		img = cv2.imread(sys.argv[1])

	cv2.imshow('Original', img)

	#lines = getLines(affine_img_retification)

	lines = np.matrix('220. 299. 1.; \
					   499. 36. 1.; \
					   257. 388. 1.; \
					   527. 98. 1.; \
					   160. 169. 1.; \
					   494. 256. 1.; \
					   214. 29. 1.; \
					   449. 83. 1.')

	H1 = AffineRectification(lines)

	affine_img_retification = cv2.warpPerspective(img, H1, (img.shape[1], img.shape[0]))

	cv2.imshow('Affine Retification', affine_img_retification)

	#lines = getLines(affine_img_retification)

	lines = np.matrix('216. 91. 1.; \
				   339. 123. 1.; \
				   424. 78. 1.; \
				   249. 169. 1.; \
				   307. 50. 1.; \
				   340. 123. 1.; \
				   215. 93. 1.; \
				   425. 78. 1.')

	H2 = MetricRectification(lines)
	H3 = H2*H1

	metric_img_retification = cv2.warpPerspective(affine_img_retification, H3, (img.shape[1], img.shape[0]))

	cv2.imshow('Metric Retification', metric_img_retification)

	cv2.namedWindow('image')
	cv2.createTrackbar('Angle','image',0,360,nothing)
	cv2.createTrackbar('Scale','image',0,5000,nothing)

	while(1):
		angle = cv2.getTrackbarPos('Angle','image')
		scale = cv2.getTrackbarPos('Scale','image')

		s = scale/1000.
		r = math.radians(angle)

		R = np.eye(3)
		R[0][0] = s*math.cos(r)
		R[0][1] = -s*math.sin(r)
		R[1][0] = s*math.sin(r)
		R[1][1] = s*math.cos(r)
		R[0][2] = img.shape[0]/2.
		R[1][2] = img.shape[1]/2.

		H = R

		img2 = cv2.warpPerspective(metric_img_retification, H, (img.shape[1], img.shape[0]))
		
		cv2.imshow('image', img2)

		k = cv2.waitKey(30) & 0xFF
		if k == 27:
			break

	cv2.waitKey()
