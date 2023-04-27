import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.bias1 = np.random.randn(hidden_dim)

        self.weights2 = np.random.randn(hidden_dim, output_dim)
        self.bias2 = np.random.randn(output_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, x):
        z1 = np.dot(x, self.weights1) + self.bias1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self.sigmoid(z2)
        return a2

    def backpropagation(self, x, y, learning_rate):
        # Feedforward
        z1 = np.dot(x, self.weights1) + self.bias1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self.sigmoid(z2)

        # Compute output layer error
        error2 = (y - a2) * self.sigmoid_derivative(a2)

        # Compute hidden layer error
        error1 = np.dot(error2, self.weights2.T) * self.sigmoid_derivative(a1)

        # Update weights and biases
        self.weights2 += learning_rate * np.dot(a1.T, error2)
        self.bias2 += learning_rate * np.sum(error2, axis=0)
        self.weights1 += learning_rate * np.dot(x.T, error1)
        self.bias1 += learning_rate * np.sum(error1, axis=0)

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            for x, label in zip(X, y):
                self.backpropagation(x, label, learning_rate)