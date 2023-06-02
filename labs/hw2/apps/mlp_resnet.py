import sys

sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    drop_out = nn.Dropout(drop_prob)
    residual = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        drop_out,
        nn.Linear(hidden_dim, dim),
        norm(dim),

    )
    return nn.Sequential(nn.Residual(residual),
                         nn.ReLU())
    # raise NotImplementedError()
    ### END YOUR SOLUTION


# def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
#     ### BEGIN YOUR SOLUTION
#     residula_modlues = []
#     for _ in range(num_blocks):
#         residula_modlues.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
#     model = nn.Sequential(
#         nn.Linear(dim, hidden_dim),
#         nn.ReLU(),
#         *residula_modlues,
#         nn.Linear(hidden_dim, num_classes),
#     )
#     return model
#
#     # raise NotImplementedError()
#     ### END YOUR SOLUTION
def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for i in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    loss_func = nn.SoftmaxLoss()
    total_loss = 0
    match = 0
    sample_number = 0
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
        for idx, data in enumerate(dataloader):
            X, y = data
            output = model(X)
            opt.reset_grad()
            loss = loss_func(output, y)
            total_loss += loss.numpy()
            loss.backward()
            opt.step()
            match += (output.numpy().argmax(1) == y.numpy()).sum()
            sample_number += y.shape[0]
    else:
        model.eval()
        for idx, data in enumerate(dataloader):
            X, y = data
            output = model(X)
            loss = loss_func(output, y)
            total_loss += loss.numpy()
            match += (output.numpy().argmax(1) == y.numpy()).sum()
            sample_number += y.shape[0]
    # 英文这里用了残差所以是 sample_number - match
    return (sample_number - match) / sample_number, total_loss / (idx + 1)
    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model = MLPResNet(784, hidden_dim=hidden_dim)
    train_data = ndl.data.MNISTDataset(
        data_dir + '/train-images-idx3-ubyte.gz',
        data_dir + '/train-labels-idx1-ubyte.gz'
    )
    test_data = ndl.data.MNISTDataset(
        data_dir + '/t10k-images-idx3-ubyte.gz',
        data_dir + '/t10k-labels-idx1-ubyte.gz',
    )
    train_loader = ndl.data.DataLoader(train_data, batch_size)
    test_loader = ndl.data.DataLoader(test_data, batch_size)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
        # print("train_acc: ", train_acc, "train_loss: ", train_loss)
    test_acc, test_loss = epoch(test_loader, model)
    return (train_acc, train_loss, test_acc, test_loss)

    # raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
