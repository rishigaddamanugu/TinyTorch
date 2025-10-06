from neural_network import *

input1 = Node(3, "input1")
input2 = Node(5, "input2")
input = np.array([input1, input2])

label = 50
nn = NN()
epochs = 10

for i in range(epochs):
    output = nn(input)
    loss = (label - output[0].val) ** 2
    print("Loss:", loss)
    dL_dO = -2 * (label - output[0].val)
    output[0].grad = dL_dO
    output[0].backward()
    nn.update_weights(0.001)
    nn.clear_gradients()

print(nn(input).item())