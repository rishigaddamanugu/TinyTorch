from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from neural_network import Linear
import numpy as np
from backprop import Node
from neural_network import softmax

class NN():
    def __init__(self):
        self.layer1 = Linear(64, 2)
        self.layer2 = Linear(2, 10)
        self.layers = [self.layer1, self.layer2]

    def forward(self, x):
        output = softmax(self.layer2(self.layer1(x)))
        return output
    
    def update_weights(self, alpha):
        self.layer1.update_weights(alpha)
        self.layer2.update_weights(alpha)

    def __call__(self, x):
        return self.forward(x)

    def clear_gradients(self):
        for layer in self.layers:
            layer.clear_gradients()



digits = load_digits()
X, y = digits.data, digits.target
k = 10
X, y = X[:k, :], y[:k]
X, y = X.T, y.T  # X is d x n
y = np.eye(10)[y] # d x n
X = np.array([[Node(float(val)) for val in row] for row in X], dtype=object)
y = np.array([[Node(float(val)) for val in row] for row in y], dtype=object)

## My neural network the input shape should be d x n where n is samples

nn = NN()
epochs = 10

for i in range(epochs):
    output = nn(X)
    loss = Node(-1) * np.sum(y * np.log(output + Node(1e-9))) / Node(output.shape[0])
    print("EPOCH", i, "LOSS", loss)
    loss.grad = 1.0
    loss.backward()
    nn.update_weights(0.001)
    nn.clear_gradients()


# plt.imshow(digits.images[0], cmap="gray")
# plt.title(f"Label: {digits.target[0]}")
# plt.show()


