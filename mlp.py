from random import random
import math
import sys

def tanh(u):
	return math.tanh(u)

def tanh_der(u):
	return 1.0 - math.tanh(u)**2

class MLP:

	def __init__(self, n_in, n_hid, n_out, eta=0.05, n_epochs=100000, epsilon=0.01):
		self.W1 = [[random() for i in range(n_in + 1)] for i in range(n_hid)]
		self.W2 = [[random() for i in range(n_hid + 1)] for i in range(n_out)]

		self.n_in = n_in
		self.n_hid = n_hid
		self.n_out = n_out

		self.eta = eta
		self.n_epochs = n_epochs
		self.epsilon = epsilon

	def execute(self, x):
		x = [-1] + x

		I1 = [0] * self.n_hid
		for j in range(self.n_hid):
			for i in range(self.n_in + 1):
				I1[j] += self.W1[j][i] * x[i]

		Y1 = [0] * self.n_hid
		for j in range(self.n_hid):
			Y1[j] = tanh(I1[j])
		Y1 = [-1] + Y1

		I2 = [0] * self.n_out
		for j in range(self.n_out):
			for i in range(self.n_hid + 1):
				I2[j] += self.W2[j][i] * Y1[i]

		Y2 = [0] * self.n_out
		for j in range(self.n_out):
			Y2[j] = tanh(I2[j])

		return Y2

	def train(self, X, D):
		MSE = []
		epoch = 0
		prevEM = sys.float_info.max

		done = False
		while done is not True:
			epoch += 1
			E = []
			for x, d in zip(X, D):
				x = [-1] + x

				I1 = [0] * self.n_hid
				for j in range(self.n_hid):
					for i in range(self.n_in + 1):
						I1[j] += self.W1[j][i] * x[i]

				Y1 = [0] * self.n_hid
				for j in range(self.n_hid):
					Y1[j] = tanh(I1[j])
				Y1 = [-1] + Y1

				I2 = [0] * self.n_out
				for j in range(self.n_out):
					for i in range(self.n_hid + 1):
						I2[j] += self.W2[j][i] * Y1[i]

				Y2 = [0] * self.n_out
				for j in range(self.n_out):
					Y2[j] = tanh(I2[j])

				e = 0
				for i in range(self.n_out):
					e += (d[i] - Y2[i])**2
				E.append(0.5 * e)

				delta1 = [0] * self.n_out
				for i in range(self.n_out):
					delta1[i] = (d[i] - Y2[i]) * tanh_der(I2[i])

				for j in range(self.n_out):
					for i in range(self.n_hid + 1):
						self.W2[j][i] = self.W2[j][i] + self.eta * delta1[j] * Y1[i]

				delta2 = [0] * self.n_hid
				for i in range(self.n_hid):
					for j in range(self.n_out):
						delta2[i] += delta1[j] * self.W2[j][i + 1] * tanh_der(I1[i])

				for j in range(self.n_hid):
					for i in range(self.n_in + 1):
						self.W1[j][i] = self.W1[j][i] + self.eta * delta2[j] * x[i]

			EM = (1.0 / len(E)) * sum(E)
			MSE.append(EM)
			if (prevEM - EM) <= self.epsilon:
				done = True

			prevEM = EM

		return MSE

X = [[0.0, 0.45], [0.3, 0.5], [0.4, 0.2], [0.5, 0.6], [0.7, 0.4], [0.9, 0.8], [1.0, 0.9], [0.0, 0.6], 
[0.0, 1.4], [0.1, 0.8], [0.2, 0.7], [0.2, 1.2], [0.2, 1.3], [0.4, 0.8], [0.4, 1.0], [0.4, 1.2], 
[0.5, 0.8], [0.5, 1.0], [0.6, 1.1], [0.7, 1.0], [0.8, 1.0], [0.0, 1.6], [0.2, 1.5], [0.3, 1.8], 
[0.4, 1.4], [0.5, 1.3], [0.5, 1.9], [0.6, 1.6], [0.7, 1.2], [0.9, 1.1], [0.9, 1.6], [1.0, 1.1]]

D = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 
[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0], 
[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 
[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 
[0.0, 0.0, 1.0]]

mlp = MLP(2, 2, 3)
MSE = mlp.train(X, D)

import matplotlib.pyplot as plt
plt.plot(MSE)
plt.title("MSE")
plt.show()

for x in X:
	print(mlp.execute(x))