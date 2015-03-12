from Point import Point
import math

# Calculates the distance between two points
def getDistance(a, b, c, d):
    dx = c-a
    dy = d-b

    dx2 = dx*dx
    dy2 = dy*dy

    return math.sqrt(dx2 + dy2)

class TrackInfo:
	def __init__(self):
		self.started = False
		self.ended = False
		self.startFrame = 0
		self.endFrame = 0
		self.direction = []
		self.points = []

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
		return self.endFrame - self.startFrame

	def addPoint(self, x, y):
		self.points.append(Point(x,y))

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
