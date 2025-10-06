import numpy as np

class Node:
    def __init__(self, val = np.random.rand(), name = "label"):
        assert isinstance(val, (int, float, np.number)), f"val must be numeric, but got {type(val)}"
        self.val = float(val)
        self.grad = 0.0
        self.backward = None
        self.name = name
        self.num_children = 0
        self.num_backwards_invoked = 0
    
    def __mul__(self, b):
        if type(b) != Node:
            b = Node(b)
        output = Node(self.val * b.val, self.name + "*" + b.name)

        def _backward():
            # do/dw is x
            # do/dx is w
            # dL/dx is dL/do * do/dx
            # dL/dw is dL/do * do/dw
            # cannot adjust nodes (obviously) but can adjust weights
            # Assuming that self is the weight
            # b represents the input node
            self.grad += (output.grad * b.val) # this is dL/dz * do/dw = dL/dw
            b.grad += (output.grad * self.val) # this is dL/dz * do/db = dL/db
            self.num_backwards_invoked += 1
            b.num_backwards_invoked += 1

            if self.num_backwards_invoked == self.num_children:
                if self.backward:
                    self.num_backwards_invoked = 0
                    self.backward()
            if b.num_backwards_invoked == b.num_children:
                if b.backward:
                    b.num_backwards_invoked = 0
                    b.backward()

        b.num_children += 1
        self.num_children += 1
        output.backward = _backward
        return output

    def __add__(self, b):
        if type(b) != Node:
            b = Node(b)
        output = Node(self.val + b.val, self.name + "+" + b.name)

        def _backward():
            self.grad += (output.grad) # this is dL/dz * do/dw = dL/dw
            b.grad += (output.grad) # this is dL/dz * do/db = dL/db
            self.num_backwards_invoked += 1
            b.num_backwards_invoked += 1

            if self.num_backwards_invoked == self.num_children:
                if self.backward:
                    self.num_backwards_invoked = 0
                    self.backward()
            if b.num_backwards_invoked == b.num_children:
                if b.backward:
                    b.num_backwards_invoked = 0
                    b.backward()

        b.num_children += 1

        self.num_children += 1
        output.backward = _backward
        return output
    
    def __sub__(self, b):
        if type(b) != Node:
            b = Node(b)
        output = Node(self.val - b.val, self.name + "-" + b.name)
        def _backward():
            self.grad += (output.grad) # this is dL/dz * do/dw = dL/dw
            b.grad += (output.grad) # this is dL/dz * do/db = dL/db
            self.num_backwards_invoked += 1
            b.num_backwards_invoked += 1

            if self.num_backwards_invoked == self.num_children:
                if self.backward:
                    self.num_backwards_invoked = 0
                    self.backward()
            if b.num_backwards_invoked == b.num_children:
                if b.backward:
                    b.num_backwards_invoked = 0
                    b.backward()

        b.num_children += 1
        self.num_children += 1
        output.backward = _backward
        return output

    def log(self):
        output = Node(np.log(self.val), f"log({self.name})")
    
        def _backward():
            # Propagate the gradient: dL/dx = dL/d(log(x)) * 1/x
            self.grad += output.grad / self.val
            self.num_backwards_invoked += 1
            # Optionally chain backward calls if needed:
            if self.num_backwards_invoked == self.num_children:
                if self.backward:
                    self.num_backwards_invoked = 0
                    self.backward()
        
        self.num_children += 1
        output.backward = _backward
        return output


    def __truediv__(self, b):
        if type(b) != Node:
            b = Node(b)
        output = Node(self.val / b.val, self.name + "/" + b.name)

        def _backward():
            # do/dself = 1/b
            # do/db = -self/b^2
            self.grad += output.grad / b.val  # dL/dself = dL/do * do/dself
            b.grad += -output.grad * self.val / (b.val ** 2)  # dL/db = dL/do * do/db
            self.num_backwards_invoked += 1
            b.num_backwards_invoked += 1

            if self.num_backwards_invoked == self.num_children:
                if self.backward:
                    self.num_backwards_invoked = 0
                    self.backward()
            if b.num_backwards_invoked == b.num_children:
                if b.backward:
                    b.num_backwards_invoked = 0
                    b.backward()

        b.num_children += 1
        self.num_children += 1
        output.backward = _backward
        return output
    
    def exp(self):
        safe_val = np.clip(self.val, -50, 50)  # Clip to avoid overflow
        output = Node(np.exp(safe_val), f"exp({self.name})")

        def _backward():
            self.grad += output.grad * output.val  # dL/dx = dL/do * e^x
            self.num_backwards_invoked += 1

            if self.num_backwards_invoked == self.num_children:
                if self.backward:
                    self.num_backwards_invoked = 0
                    self.backward()
            
        self.num_children += 1
        output.backward = _backward
        return output

    
    def __ge__(self, other):
        other = other if isinstance(other, Node) else Node(other, str(other))
        output = Node(float(self.val >= other.val), f"({self.name} >= {other.name})")  # Convert bool to float

        def _backward():
            # No gradients propagate through comparisons
            self.grad += 0
            other.grad += 0
            self.num_backwards_invoked += 1
            other.num_backwards_invoked += 1
            
            if self.num_backwards_invoked == self.num_children:
                if self.backward:
                    self.num_backwards_invoked = 0
                    self.backward()
            if other.num_backwards_invoked == other.num_children:
                if other.backward:
                    other.num_backwards_invoked = 0
                    other.backward()

        other.num_children += 1
        self.num_children += 1
        output.backward = _backward
        return output




    
    def __str__(self):
        output = " Val: " + str(self.val) + " Grad: " + str(self.grad)  
        return output
    
    def __repr__(self):
        output = " Val: " + str(self.val) + " Grad: " + str(self.grad)  
        return output

# def use_normal():
#     input1 = Node(3, "input1")
#     input2 = Node(5, "input2")
#     weight1 = Node(10, "weight")
#     weight2 = Node(10, "weight")

#     output = (weight1 * input1) + (weight2 * input2)

#     label = 50
#     loss = (label - output.val) ** 2
#     dL_dO = -2 * (label - output.val)
#     output.grad = dL_dO
#     output.backward()
    
# def use_np():
#     input1 = Node(3, "input1")
#     input2 = Node(5, "input2")
#     weight1 = Node(10, "weight")
#     weight2 = Node(10, "weight")

#     layer1 = np.array([input1, input2])
#     weights1 = np.array([weight1, weight2]).T
#     output = weights1 @ layer1
    
#     label = 50
#     loss = (label - output.val) ** 2
#     dL_dO = -2 * (label - output.val)
#     output.grad = dL_dO
#     output.backward()

# use_np()
# use_normal()
# o = w*x
# do/dx = w
# do/dw = x

# dL/dx = dL/do * do/dx = dL/do * w
# dL/dw = dL/do * do/dw = dL/do * x

