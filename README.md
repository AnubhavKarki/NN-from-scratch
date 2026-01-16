# Building Neural Networks from Scratch

Minimal NumPy implementations of neural networks. Builds foundational understanding of forward propagation, backpropagation, and gradient descent without framework dependencies.

## Overview

Two progressive implementations:

**Single Neuron Network** (`basic-neural-net.py`)
- Linear model: `y = x * w1 * w2`
- MSE loss with manual gradient computation
- SGD optimization

**Two-Layer Network with Activation** (`complex-neural-net.py`)  
- Hidden layer with sigmoid activation
- Multi-input forward pass: `x1*w1 + x2*w2 -> sigmoid -> w3 -> sigmoid`
- Chain rule backpropagation through layers

## Architecture

### Single Neuron
```
Input (x) → w1 → Hidden (h) → w2 → Output (y_hat)
Loss: 0.5*(y_hat - y)^2
```

### Two-Layer Network
```
x1 ─┐
    ├── w1 ─┐
x2 ─┤        ├─ sigmoid ─ w3 ─ sigmoid ─ y_hat
            └─ z1 ───────┘
Loss: 0.5*(y_hat - y)^2
```

## Core Components

- **Forward Pass**: Matrix multiplications and activations
- **Loss**: Mean Squared Error  
- **Gradients**: Chain rule derivatives (no autograd)
- **Optimization**: Stochastic Gradient Descent
- **Training**: Epoch-wise weight updates with loss monitoring

## Quick Start

```bash
# Single neuron (targets y=20 for x=2)
python neural_net.py

# Two-layer network (binary classification target)
python complex_net.py
```

## Training Parameters

| Parameter | Single Neuron | Two-Layer | Purpose |
|-----------|---------------|-----------|---------|
| `alpha` | 0.01 | 0.01 | Learning rate |
| `epochs` | 10 | 10 | Training iterations |
| `w_init` | Random | Random | Weight initialization |

## Expected Convergence

**Single Neuron**: `w1→5.0, w2→4.0` (approximates `y=20x`)
**Two-Layer**: Loss → 0.01+ targeting binary output

## Learning Outcomes

- Manual derivative computation
- Backpropagation through sigmoid nonlinearity
- Weight update mechanics
- Loss landscape navigation

## Navigation

```
.
├── neural_net.py      # Single neuron baseline
├── complex_net.py     # Hidden layer + sigmoid
└── README.md          # This file
```

## Extensions

1. Add bias terms to layers
2. Vectorize for batch training  
3. Implement ReLU/Leaky ReLU
4. Mini-batch SGD
5. Momentum/Adam optimizers
6. Multi-class softmax output

## Prerequisites

- Python 3.8+
- NumPy (`pip install numpy`)

Pure NumPy. No PyTorch/TensorFlow/JAX.

## License - MIT
