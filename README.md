## 🧠 MiniGrad — A Lightweight Automatic Differentiation Engine

MiniGrad is a from-scratch deep learning framework that replicates the core mechanics of **PyTorch’s autograd** system.  
It builds and executes dynamic computation graphs for tensor operations, enabling fully automated differentiation and gradient-based optimization.

This project was created to understand the internal workings of deep learning frameworks by reimplementing every major component — tensors, operations, graph construction, and backpropagation — without external ML libraries.

---

### 🎯 Objectives
- Rebuild the foundation of a modern deep learning framework from scratch.
- Implement dynamic computation graphs that track operations during forward passes.
- Automate gradient computation through recursive backpropagation.
- Train neural networks using the custom autograd and optimization layers.

---

### ⚙️ Key Features

#### 🧩 Autograd Engine
- Dynamically constructs a computation graph during tensor operations.  
- Supports addition, multiplication, matrix operations, and nonlinear activations.  
- Performs recursive backward passes to compute gradients automatically.  

#### 🔁 Neural Network Trainer
- Custom training loop for small feedforward networks.  
- Implements stochastic gradient descent with adjustable learning rate and loss functions.  
- Tested on toy datasets (e.g., handwritten digit samples) to verify convergence and gradient correctness.  

#### 🧱 Extensible Design
- Modular Tensor class encapsulates data, gradient, and operation metadata.  
- Operators and layers can be extended easily to support new functionality.  
- Provides educational insight into how frameworks like PyTorch and TensorFlow manage autograd internally.

---

### 🧠 Tech Stack
Python • NumPy • Computational Graphs • Gradient Descent • Neural Networks

---

### 🚀 Vision
MiniGrad aims to serve as an **educational framework** for understanding the inner mechanics of modern deep learning systems. It bridges theory and implementation, showing how automatic differentiation, optimization, and neural computation interact under the hood.
