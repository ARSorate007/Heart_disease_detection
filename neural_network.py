import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, layers=[13, 8, 1], learning_rate=0.001, iterations=100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None

    def init_weights(self):
        np.random.seed(1)
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])
        self.params["b1"] = np.random.randn(self.layers[1])
        self.params["W2"] = np.random.randn(self.layers[1], self.layers[2])
        self.params["b2"] = np.random.randn(self.layers[2])

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def sigmoid(Z):
        return 1.0 / (1.0 + np.exp(-Z))

    def entropy_loss(self, y, yhat):
        nsample = len(y)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat))))
        return loss

    def forward_propagation(self):
        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y, yhat)

        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat, loss

    def back_propagation(self, yhat):
        d1_wrt_yhat = -(np.divide(self.y, yhat) - np.divide((1 - self.y), (1 - yhat)))
        d1_wrt_sig = yhat * (1 - yhat)
        d1_wrt_z2 = d1_wrt_yhat * d1_wrt_sig

        d1_wrt_A1 = d1_wrt_z2.dot(self.params['W2'].T)
        d1_wrt_w2 = self.params['A1'].T.dot(d1_wrt_z2)
        d1_wrt_b2 = np.sum(d1_wrt_z2, axis=0)

        d1_wrt_z1 = d1_wrt_A1 * self.relu(self.params['Z1'])
        d1_wrt_w1 = self.X.T.dot(d1_wrt_z1)
        d1_wrt_b1 = np.sum(d1_wrt_z1, axis=0)

        self.params['W1'] = self.params['W1'] - self.learning_rate * d1_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * d1_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * d1_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * d1_wrt_b2

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.init_weights()

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

    def predict(self, X):
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)

    def acc(self, y, yhat):
        acc = int(sum(y == yhat) / len(y) * 100)
        print(sum(y==yhat),"\n",len(y))
        print(f"accuracy is : {int(sum(y == yhat) / len(y) * 100)}")
        return acc

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("Log Loss")
        plt.title("Loss Curve for Training")
        plt.show()
