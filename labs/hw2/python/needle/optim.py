"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for index, p in enumerate(self.params):
            if index not in self.u:
                self.u[index] = 0
            grad = ndl.Tensor(p.grad, dtype="float32").data + self.weight_decay * p.data
            self.u[index] = self.momentum * self.u[index] + (1 - self.momentum) * grad
            p.data -= self.lr * self.u[index]

        # raise NotImplementedError()
        ### END YOUR SOLUTION



class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for index, p in enumerate(self.params):
            if index not in self.m:
                self.m[index] = ndl.init.zeros(*p.shape)
                self.v[index] = ndl.init.zeros(*p.shape)
            grad = ndl.Tensor(p.grad, dtype='float32').data + p.data * self.weight_decay
            # m_{t+1}, v{t+1}
            self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * grad
            self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * grad**2
            # bias correction
            m_hat = (self.m[index]) / (1 - self.beta1 ** self.t)
            v_hat = (self.v[index]) / (1 - self.beta2 ** self.t)
            p.data = p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION
