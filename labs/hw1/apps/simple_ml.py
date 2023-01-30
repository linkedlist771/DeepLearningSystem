import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # with gzip.GzipFile(image_filesname, 'rb') as image_file:
    #     img_buf = image_file.read()
    #     with gzip.GzipFile(label_filename, 'rb') as label_file:
    #         lab_buf = label_file.read()
    #         # step_label = 0
    #         offset_img = 0
    #         # read from Big-endian
    #         # get file info from magic byte
    #         # image file : 16B
    #         magic_byte_img = '>IIII'
    #         magic_img, image_num, rows, cols = struct.unpack_from(
    #             magic_byte_img, img_buf, offset_img)
    #         offset_img += struct.calcsize(magic_byte_img)
    #         offset_lab = 0
    #         # label file : 8B
    #         magic_byte_lab = '>II'
    #         magic_lab, label_num = struct.unpack_from(magic_byte_lab,
    #                                                   lab_buf, offset_lab)
    #         offset_lab += struct.calcsize(magic_byte_lab)
    #         # 设置读取图片的数量
    #         buffer_size = label_num
    #
    #         fmt_label = '>' + str(label_num) + 'B'
    #         labels = struct.unpack_from(fmt_label, lab_buf, offset_lab)
    #         labels = np.reshape(labels, (buffer_size, -1)).astype(np.uint8)
    #         offset_lab += struct.calcsize(fmt_label)
    #
    #         fmt_images = '>' + str(buffer_size * rows * cols) + 'B'
    #         images_temp = struct.unpack_from(fmt_images, img_buf, offset_img)
    #         images = np.reshape(images_temp, (buffer_size, rows * cols)).astype('float32')
    #         # 保持原来的数据格式
    #         images_max = np.max(images, axis=1, keepdims=True)
    #         images_min = np.min(images, axis=1, keepdims=True)
    #         images = ((images - images_min) / (images_max - images_min))
    #         return images, labels
    f = gzip.open(image_filesname)
    data = f.read()
    f.close()
    h = struct.unpack_from('>IIII', data, 0)
    offset = struct.calcsize('>IIII')
    imgNum = h[1]
    rows = h[2]
    columns = h[3]
    pixelString = '>' + str(imgNum * rows * columns) + 'B'
    pixels = struct.unpack_from(pixelString, data, offset)
    X = np.reshape(pixels, [imgNum, rows * columns]).astype('float32')
    X_max = np.max(X)
    X_min = np.min(X)
    # X_max = np.max(X, axis=1, keepdims=True)
    # X_min = np.min(X, axis=1, keepdims=True)

    X_normalized = ((X - X_min) / (X_max - X_min))

    f = gzip.open(label_filename)
    data = f.read()
    f.close()
    h = struct.unpack_from('>II', data, 0)
    offset = struct.calcsize('>II')
    num = h[1]
    labelString = '>' + str(num) + 'B'
    labels = struct.unpack_from(labelString, data, offset)
    y = np.reshape(labels, [num]).astype('uint8')

    return (X_normalized, y)
    # END YOUR SOLUTION

def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # 找到预测输出中最大的值exp/sum(exp) 的值， 也就是用真实的one hot 想乘去掉其他的
    batch_size = Z.shape[0]
    # y_one_hot = y_one_hot.astype('float32')
    y_one_hot = ndl.summation(Z * y_one_hot, axes=1)
    Z1 = ndl.log(ndl.summation(ndl.exp(Z), axes=1))
    loss = ndl.summation(Z1-y_one_hot)/batch_size
    return loss

    # m = Z.shape[0]
    # Z1 = ndl.ops.summation(ndl.ops.log(ndl.ops.summation(ndl.ops.exp(Z), axes=(1,))))
    # Z2 = ndl.ops.summation(Z * y_one_hot)
    # return (Z1 - Z2) / m
    # raise NotImplementedError()
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    input_dim = X.shape[0]
    for i in range(0, input_dim, batch):
        X_batch = X[i:i + batch]
        y_batch = y[i:i + batch]
        X_batch = ndl.Tensor(X_batch)
        layer1_output = ndl.relu(ndl.matmul(X_batch, W1))
        layer2_output = ndl.matmul(layer1_output, W2)
        y_one_hot = np.zeros(layer2_output.shape)
        y_one_hot = y_one_hot.astype('float32')
        y_one_hot[np.arange(layer2_output.shape[0]), y_batch] = 1
        y_one_hot = ndl.Tensor(y_one_hot)
        loss = softmax_loss(layer2_output, y_one_hot)
        loss.backward()
        W1 -= lr * W1.grad
        W2 -= lr * W2.grad
        W1 = W1.detach()
        W2 = W2.detach()
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)

