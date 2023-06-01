"""The module.
"""
import random
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        ### BEGIN YOUR SOLUTION
        #self.bias = bias
        self.device = device
        self.dtype = dtype
        # 如果有偏置的话
        weight = init.kaiming_uniform(in_features, out_features, nonlinearity="relu", device=device, dtype=dtype)
        self.weight = Tensor(weight)
        if bias:
            # 然而，如果你的代码中某些地方假定偏置是一个二维张量，而且第一个维度为1，那么reshape就是必须的。
            # 这可能是为了保持与特定操作的兼容性，例如广播或矩阵乘法。
            # 另外一种可能的原因是，在使用Kaiming初始化时，fan_in和fan_out参数的顺序可能会影响到初始化的结果。
            # 在你的代码中，可能作者想要使用out_features作为fan_in来进行初始化。然后，通过reshape操作将偏置调整为正确的形状。
            # 不过在偏置初始化中，一般不需要这样做，因为偏置通常都被初始化为零或者接近零的小值。
            self.bias = init.kaiming_uniform(out_features, 1, dtype=dtype).reshape((1, out_features))
            # self.bias = init.kaiming_uniform(1, out_features, nonlinearity="relu", device=device, dtype=dtype)
        else:
            self.bias = None
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias:
            # 如果有偏置的话
            a_l = ops.matmul(X, self.weight)
            return a_l + self.bias
        else:
            return ops.matmul(X, self.weight)
        # raise NotImplementedError()
        ## END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch = X.numpy().shape[0]
        return X.reshape((batch, -1))
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # 这段代码是一个模块，用于计算
        # softmax
        # 损失（也被称为交叉熵损失）的前向传播。这是一种常见的用于多分类任务的损失函数。
        #
        # 输入参数有两个：logits
        # 和
        # y。logits
        # 是模型的输出，它是一个二维
        # tensor，每一行对应一个样本，每一列对应一个类别。y
        # 是真实标签，是一个一维
        # tensor，长度等于样本数量，每个元素是一个样本的真实类别。
        #
        # 在
        # forward
        # 函数中，首先计算
        # logits
        # 的
        # logsumexp，axes = (1,)
        # 表示对每一行（也就是每一个样本）进行操作，然后对结果求和，得到
        # exp_sum。
        #
        # 然后，通过
        # init.one_hot(logits.shape[1], y)
        # 对真实标签
        # y
        # 进行
        # one - hot
        # 编码。编码后的结果是一个二维
        # tensor，每一行对应一个样本，每一列对应一个类别，如果某个样本的真实类别是某个类别，那么对应的元素是
        # 1，否则是
        # 0。将这个结果与
        # logits
        # 逐元素相乘，然后对结果求和，得到
        # z_y_sum。
        #
        # 最后，返回(exp_sum - z_y_sum) / logits.shape[0]。这实际上是
        # softmax
        # 损失的公式：对每个样本，取真实类别对应的
        # logits（也就是
        # z_y_sum），然后减去所有
        # logits
        # 经
        # softmax
        # 转换后的结果的对数的和（也就是
        # exp_sum），再对所有样本的结果取平均（即除以样本数量
        # logits.shape[0]）。
        #
        # 这种实现方式可以避免直接计算
        # softmax
        # 函数的数值问题，同时使用矩阵运算可以提高计算效率。

        ### BEGIN YOUR SOLUTION
        exp_sum = ops.logsumexp(logits, axes=(1, )).sum()
        z_y_sum = (logits * init.one_hot(logits.shape[1], y)).sum()
        return (exp_sum - z_y_sum) / logits.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.device = device
        self.dtype = dtype
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=self.device, dtype=self.dtype))
        self.bias = Parameter(init.zeros(self.dim, device=self.device, dtype=self.dtype))
        self.running_mean = init.zeros(self.dim, device=self.device, dtype=self.dtype)
        self.running_var = init.ones(self.dim, device=self.device, dtype=self.dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        # running estimates
        mean = x.sum(axes=(0,)) / batch_size
        x_minus_mean = x - mean.broadcast_to(x.shape)
        print(mean)
        print(mean.broadcast_to(x.shape))
        var = (x_minus_mean ** 2).sum(axes=(0, )) / batch_size

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data

            x_std = ((var + self.eps) ** 0.5).broadcast_to(x.shape)
            normed = x_minus_mean / x_std
            return normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        else:
            normed = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
            return normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION




class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.device = device
        self.dtype = dtype
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype), requires_grad=True)
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype), requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        feature_size = x.shape[1]
        mean = x.sum(axes=(1, )).reshape((batch_size, 1)) / feature_size
        x_minus_mean = x - mean.broadcast_to(x.shape)
        x_std = ((x_minus_mean ** 2).sum(axes=(1, )).reshape((batch_size, 1)) / feature_size + self.eps) ** 0.5
        normed = x_minus_mean / x_std.broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * normed + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return x * init.randb(*x.shape, p=self.p) / (1 - self.p)
        else:
            return x
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION



