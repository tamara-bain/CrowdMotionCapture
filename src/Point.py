import math

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def set(self, x, y):
		self.x = x
		self.y = y

	def getCoords(self):
		return self.x, self.y

	def length(self):
		x2 = self.x*self.x
		y2 = self.y*self.y

		return math.sqrt(x2+y2)

	def normalize(self):
		x2 = self.x*self.x
		y2 = self.y*self.y

		length = math.sqrt(x2+y2)

		if length == 0:
			return

		self.x = self.x/length
		self.y = self.y/length

	def __str__(self):
		return "(" + str(self.x) + "," + str(self.y) + ")"