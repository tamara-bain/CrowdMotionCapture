from Point import Point

class TrackInfo:
	def __init__(self):
		self.started = False
		self.ended = False
		self.location = Point(0,0)
		self.direction = Point(0,0)

	def start(self):
		self.started = True

	def end(self):
		self.ended = True

	def active(self):
		return self.started and not self.ended

	def setLocation(x, y):
		self.location.set(x, y)

	def direction(x, y):
		self.direction.set(x, y)
		self.direction.normalize()


	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return "TrackInfo(" + \
			str(self.started) + "," + \
			str(self.ended) + "," + \
			str(self.location) + "," + \
			str(self.direction) + ")"
