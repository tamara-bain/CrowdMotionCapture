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

	def applyMatrix(self, A):
		n = len(self.points)
		for i in range(0,n):
			self.points[i].applyMatrix(A)

	def doesMotionStop(self):
		n = len(self.points)
		l = 3
		speed = []
		for i in range(0,n-l-1):
			a,b = self.points[i].getCoords()
			dx = dy = 0
			for j in range(l):
				index = i+j+1
				c,d = self.points[i+l+1].getCoords()

				dx = dx + c - a
				dy = dy + d - b

			dx = dx/l
			dy = dy/l

			d = Point(dx, dy)
			speed.append(d)

		n = len(speed)
		for i in range(0,n):
			if speed[i].length() < 0.1:
				return True

		return False

	def calcMotionEnergy(self):
		n = len(self.points)
		l = 3
		speed = []
		for i in range(0,n-l-1):
			a,b = self.points[i].getCoords()
			dx = dy = 0
			for j in range(l):
				index = i+j+1
				c,d = self.points[i+l+1].getCoords()

				dx = dx + c - a
				dy = dy + d - b

			dx = dx/l
			dy = dy/l

			d = Point(dx, dy)
			speed.append(d)

		n = len(speed)

		if n == 0:
			return 0
			
		e = Point(0, 0)
		mx = my = 0
		for i in range(0,n):
			a,b = speed[i].getCoords()
			mx = mx + a
			my = my + b

		mx = mx/n
		my = my/n

		for i in range(0,n):
			a,b = speed[i].getCoords()

			dx = mx - a
			dy = my - b

			dx2 = dx*dx
			dy2 = dy*dy

			e.x = e.x + dx2
			e.y = e.y + dy2

		e.x = e.x/(n-1)
		e.y = e.y/(n-1)

		return e.length()


	def calcDirection(self):
		del self.direction[:]
		n = len(self.points)
		for i in range(0,n-1):
			a,b = self.points[i].getCoords()
			c,d = self.points[i+1].getCoords()

			dx = c - a
			dy = d - b

			d = Point(dx, dy)

			if d.length() < 0.001:
				d = Point(0, 0)
			else:
				d.normalize()

			self.direction.append(d)

	def findSharp(self):
		n = len(self.direction)
		for i in range(1,n):
			a,b = self.direction[i-1].getCoords()
			c,d = self.direction[i].getCoords()

			dot = a*c + b*d
			angle = math.acos(dot)

			if abs(angle) > 0.5:
				return i

		return -1

	def split(self, frame):
		if frame < self.startFrame or frame > self.endFrame:
			return None

		i = self.endFrame - frame
		n = len(self.points)
		new_points = self.points[i:n]
		del self.points[i:n]

		n = len(self.direction)
		new_directions = self.direction[i:n]
		del self.direction[i:n]

		track = TrackInfo()
		track.start(frame)
		track.end(self.endFrame)
		track.points = self.points
		track.direction = self.direction

		self.points = new_points
		self.direction = new_directions

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
