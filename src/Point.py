

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def set(self, x, y):
		self.x = x
		self.y = y

	def getCoords(self):
		return x, y

	def normalize(self):
		x2 = self.x*self.x
		y2 = self.y*self.y

		length = sqrt(x2+y2)

		self.x = self.x/length
		self.y = self.y/length

	def __str__(self):
		return "(" + str(self.x) + "," + str(self.y) + ")"