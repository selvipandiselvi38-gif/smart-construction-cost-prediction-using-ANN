"""
ANN from scratch (NumPy) that follows the slide 'ALGORITHMS' exactly:

  1. Data initialization (random weights)
  2. Forward propagation
  3. Calculate loss (MSE)
  4. Backpropagation
  5. Update weights (gradient descent)
  6. Repeat until convergence
"""
import numpy as np


class ANNRegressor:
    def __init__(self, layer_sizes=(7, 16, 8, 1), lr=0.01, epochs=3000,
                 tol=1e-7, patience=100, seed=42, verbose=True):
        self.sizes = layer_sizes
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.patience = patience
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.loss_history = []
        self._init_weights()

    # Step 1: Data / weight initialization (He initialization for ReLU)
    def _init_weights(self):
        self.W, self.b = [], []
        for i in range(len(self.sizes) - 1):
            fan_in = self.sizes[i]
            self.W.append(self.rng.normal(0, np.sqrt(2.0 / fan_in),
                                          (self.sizes[i], self.sizes[i + 1])))
            self.b.append(np.zeros((1, self.sizes[i + 1])))

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_grad(z):
        return (z > 0).astype(float)

    # Step 2: Forward propagation
    def _forward(self, X):
        activations, zs = [X], []
        a = X
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            a = z if i == len(self.W) - 1 else self._relu(z)  # linear output
            activations.append(a)
        return activations, zs

    # Step 3: Loss (Mean Squared Error)
    @staticmethod
    def _mse(y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    # Step 4: Backpropagation
    def _backward(self, activations, zs, y):
        m = y.shape[0]
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        delta = 2 * (activations[-1] - y) / m               # dLoss/d(output)
        for i in reversed(range(len(self.W))):
            grads_W[i] = activations[i].T @ delta
            grads_b[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * self._relu_grad(zs[i - 1])
        return grads_W, grads_b

    # Step 5: Update weights
    def _update(self, grads_W, grads_b):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

    # Step 6: Repeat until convergence
    def fit(self, X, y):
        y = y.reshape(-1, 1)
        best, wait = np.inf, 0
        for epoch in range(1, self.epochs + 1):
            acts, zs = self._forward(X)
            loss = self._mse(y, acts[-1])
            self.loss_history.append(loss)
            gW, gb = self._backward(acts, zs, y)
            self._update(gW, gb)

            if best - loss > self.tol:
                best, wait = loss, 0
            else:
                wait += 1
            if wait >= self.patience:
                if self.verbose:
                    print(f"Converged at epoch {epoch} (loss={loss:.6f})")
                break
            if self.verbose and epoch % 500 == 0:
                print(f"Epoch {epoch:5d} | MSE loss = {loss:.6f}")
        return self

    def predict(self, X):
        return self._forward(X)[0][-1].ravel()
