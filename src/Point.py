##############################################################################
# Point.py
# Code Written By: Michael Feist
#
# This is a helper class created to store a point with x and y coordinates.
##############################################################################

import numpy as np
import math

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def set(self, x, y):
		self.x = x
		self.y = y

	def applyMatrix(self, A):
		p = np.matrix([[self.x], [self.y], [1]])
		p = A*p

		self.x = p[0]/p[2]
		self.y = p[1]/p[2]

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