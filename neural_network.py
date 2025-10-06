import numpy as np
from backprop import *

class Linear():
    def __init__(self, input_dim, output_dim):
        self.weights = np.array([[Node(np.random.randn() + 1e-6) for _ in range(input_dim)] for _ in range(output_dim)], dtype=object)

    def forward(self, x):
        output = self.weights @ x
        return output

    def update_weights(self, alpha):
        extract_grad = np.vectorize(lambda node: node.grad)
        gradient_matrix = extract_grad(self.weights)

        np.vectorize(
            lambda node, grad: setattr(node, 'val', node.val - alpha * grad)
        )(self.weights, gradient_matrix)

    def clear_gradients(self):
        for node in self.weights.flat:
            node.grad = 0.0

    def __call__(self, x):
        return self.forward(x)
    
    
def softmax(nodes):
    nodes = np.array(nodes, dtype=object)
    exp_vals = np.exp(nodes - np.max(nodes, axis=0, keepdims=True))  # Stabilized exponentials over axis 0
    return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)  # Normalize over axis 0

def log_softmax(nodes):
    log_softmax = nodes - np.max(nodes, axis=0, keepdims=True) - np.log(np.sum(np.exp(nodes - np.max(nodes, axis=0, keepdims=True)), axis=0, keepdims=True))
    return log_softmax


# class NN():
#     def __init__(self):
#         self.layer1 = Linear(2, 16)
#         self.layer2 = Linear(16, 1)
#         self.layers = [self.layer1, self.layer2]

#     def forward(self, x):
#         output = softmax(self.layer2(self.layer1(x)))
#         return output
    
#     def update_weights(self, alpha):
#         self.layer1.update_weights(alpha)
#         self.layer2.update_weights(alpha)

#     def __call__(self, x):
#         return self.forward(x)

#     def clear_gradients(self):
#         for layer in self.layers:
#             layer.clear_gradients()

