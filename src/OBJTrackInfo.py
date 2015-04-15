##############################################################################
# OBJTrackInfo.py
# Code Written By: Michael Feist
#
# This is a helper class created to store info about tracks used in
# OBJCrowdTracking.py
##############################################################################

from Point import Point
import math

# Calculates the distance between two points
def getDistance(a, b, c, d):
    dx = c-a
    dy = d-b

    dx2 = dx*dx
    dy2 = dy*dy

    return math.sqrt(dx2 + dy2)

class OBJTrackInfo:
	def __init__(self):
		self.started = False
		self.ended = False
		self.startFrame = 0
		self.endFrame = 0

		# Path
		self.points = []

		# How long since the object was last seen?
		self.lastFound = 0

		# Center of bounding box
		self.x = 0
		self.y = 0

		# Bounding box
		self.bx = 0
		self.by = 0
		self.bw = 0
		self.bh = 0

	def setBoundingBox(self, x, y, w, h):
		self.bx = x
		self.by = y
		self.bw = w
		self.bh = h

	def start(self, frame):
		if not self.started:
			self.startFrame = frame
		self.started = True

	def end(self, frame):
		if not self.ended:
			self.endFrame = frame
		self.ended = True

	def active(self):
		return self.started and not self.ended

	def getNumberOfFrames(self):
		return len(self.points)

	def addPoint(self, x, y):
		self.points.append(Point(x,y))

	def applyMatrix(self, A):
		n = len(self.points)
		for i in range(0,n):
			self.points[i].applyMatrix(A)

	def split(self, frame):
		if frame < self.startFrame or frame > self.endFrame:
			return None

		i = self.endFrame - frame
		n = len(self.points)
		new_points = self.points[i:n]
		del self.points[i:n]

		track = TrackInfo()
		track.start(frame)
		track.end(self.endFrame)
		track.points = self.points

		self.points = new_points

		self.endFrame = frame

		return track


	def getDistanceTraveled(self):
		num_of_points = len(self.points)

		if num_of_points <= 0:
			return 0.0

		p1 = self.points[0]
		a,b = p1.getCoords()

		distance = 0.0

		for i in range(1,num_of_points):
			p2 = self.points[i]
			c,d = p2.getCoords()

			diff = getDistance(a,b,c,d)

			if diff > distance:
				distance = diff

		return distance

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		num_of_points = len(self.points)
		return "TrackInfo(" + \
			str(self.startFrame) + "," + \
			str(self.endFrame) + "," + \
			str(num_of_points) + ")"
