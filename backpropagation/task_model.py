import numpy as np
import random
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

num_classes = 10
img_rows, img_cols = 28, 28
input_size = img_rows * img_cols
beta = 0.9  # for momentum


def crossentropy_loss(Y, Y_hat):
	L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
	m = Y.shape[1]
	L = -(1/m) * L_sum

	return L


def relu(x):
	return np.maximum(x, 0)
 
 
def relu_prime(x):
	out = x[:]
	out[out <= 0] = 0
	out[out > 0] = 1
	return out


def sigmoid(x):
	s = 1. / (1. + np.exp(-x))
	return s


def sigmoid_prime(x):
	return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)


class Network():
	def __init__(self, units):
		self.W1 = np.random.randn(units, input_size) * np.sqrt(1. / input_size)
		self.b1 = np.zeros((units, 1)) * np.sqrt(1. / input_size)
		self.W2 = np.random.randn(num_classes, units) * np.sqrt(1. / units)
		self.b2 = np.zeros((num_classes, 1)) * np.sqrt(1. / units)

	def feed_forward(self, X):
		cache = {}

		cache['Z1'] = np.matmul(self.W1, X) + self.b1
		cache['A1'] = relu(cache['Z1'])
		cache['Z2'] = np.matmul(self.W2, cache['A1']) + self.b2
		cache['A2'] = softmax(cache['Z2'])

		return cache

	def back_propagate(self, X, y, cache, m_batch):
		# error at last layer
		dZ2 = cache["A2"] - y

		# gradients at last layer
		dW2 = (1. / m_batch) * np.matmul(dZ2, cache["A1"].T)
		db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

		# back propgate through first layer
		dA1 = np.matmul(self.W2.T, dZ2)
		dZ1 = dA1 * relu_prime(cache["Z1"])

		# gradients at first layer
		dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
		db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

		grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

		return grads

	def fit(self, X, y, epochs=20, batch_size=256, learning_rate=0.1):
		dW1 = np.zeros(self.W1.shape)
		db1 = np.zeros(self.b1.shape)
		dW2 = np.zeros(self.W2.shape)
		db2 = np.zeros(self.b2.shape)

		for i in range(epochs):
			for j in range(X.shape[1] // batch_size):

				# get mini-batch
				begin = j * batch_size
				end = min(begin + batch_size, X.shape[1] - 1)
				X_batch = X[:, begin:end]
				y_batch = y[:, begin:end]

				# forward and backward
				cache = self.feed_forward(X_batch)
				grads = self.back_propagate(X_batch, y_batch, cache, batch_size)

				# with momentum
				dW1 = beta*dW1 + (1.-beta)*grads["dW1"]
				db1 = beta*db1 + (1.-beta)*grads["db1"]
				dW2 = beta*dW2 + (1.-beta)*grads["dW2"]
				db2 = beta*db2 + (1.-beta)*grads["db2"]

				# gradient descent
				self.W1 = self.W1 - learning_rate * dW1
				self.b1 = self.b1 - learning_rate * db1
				self.W2 = self.W2 - learning_rate * dW2
				self.b2 = self.b2 - learning_rate * db2

			# forward pass on training set
			cache = self.feed_forward(X)
			train_loss = crossentropy_loss(y, cache["A2"])
			print("Epoch {}: training loss = {}".format(i + 1, train_loss))

	def evaluate(self, X, y):
		cache = self.feed_forward(X)
		test_loss = crossentropy_loss(y, cache["A2"])
		print("test loss = {}".format(test_loss))

	def predict(self, X):
		cache = self.feed_forward(X)
		predictions = np.argmax(cache["A2"], axis=0)
		return predictions


def get_mnist_data():
	mnist = datasets.fetch_openml('mnist_784')
	X, y = mnist["data"], mnist["target"]

	X = X.astype(np.float32)
	X /= 255.

	encoder = OneHotEncoder(sparse=False)
	y = encoder.fit_transform(y.reshape(-1, 1))

	x_train, x_test, y_train, y_test = train_test_split(
		X.astype(np.float32),
		y.astype(np.float32),
		test_size=(1 / 10.))

	return x_train.T, y_train.T, x_test.T, y_test.T


def main():
	print('get data...')
	x_train, y_train, x_test, y_test = get_mnist_data()
	
	print('create model...')
	model = Network(units=300)

	print('fit model...')
	model.fit(x_train, y_train)

	model.evaluate(x_test, y_test)

	predictions = model.predict(x_test)
	labels = np.argmax(y_test, axis=0)
	# print(predictions, labels)
	print(classification_report(predictions, labels))


if __name__ == '__main__':
	main()
