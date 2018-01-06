# encoding=utf8

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

learning_rate = 0.01

input_node_num = 2;
hidden_node_num = 8;
output_node_num = 1;

example_num = 200; # 输入数据点的个数
epoch_num = 10000; #迭代次数

# lambd = 0.001 # L2 正则化参数
lambd = 0.01 # L2 正则化参数

epsilon = 10 ** (-10)

MOVING_AVERAGE = {'std1':0, 'mean1':0, 'std2':0, 'mean2':0}
MOVING_AVERAGE_BATA = 0.9


def get_case_data(example_num = 200):
    """
    获取点数据，需要进一步处理
    :param example_num: 点的个数
    :return:
    """
    np.random.seed(6)
    X, y = sklearn.datasets.make_moons(example_num, noise=0.30)
    return X, y;


def process_data(X, Y):
    """
    处理原始数据集
    :param X:
    :param Y: label
    :return:
    """
    Y = np.reshape(Y, (1, -1))
    return X.T, Y;


def random_mini_batches(X, Y, batch_size=64):
    """
    :param X:
    :param Y:
    :param batch_size:
    :return:
    """
    m = Y.shape[1]
    permutation = np.random.permutation(Y.shape[1])
    X_shuffle = X[:, permutation]
    Y_shuffle = Y[:, permutation]

    mini_batches = []
    batch_num = m//batch_size
    for i in range(batch_num):
        mini_batch_x = X_shuffle[:, i * batch_size: (i + 1) * batch_size]
        mini_batch_y = Y_shuffle[:, i * batch_size: (i + 1) * batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    if batch_num * batch_size < m:
        mini_batch_x = X_shuffle[:, batch_num * batch_size: m]
        mini_batch_y = Y_shuffle[:, batch_num * batch_size: m]
        mini_batch = [mini_batch_x, mini_batch_y]
        mini_batches.append(mini_batch)

    mini_batches = np.array(mini_batches)
    return mini_batches


def initialize_parameters():
    """
    初始化参数
    :return:
    """
    W1 = np.random.randn(hidden_node_num, input_node_num) * 0.01;
    b1 = np.zeros((hidden_node_num, 1))
    W2 = np.random.randn(output_node_num, hidden_node_num) * 0.01;
    b2 = np.zeros((output_node_num, 1))

    varphi1 = np.random.randn(hidden_node_num, hidden_node_num) * 0.01
    bata1 = np.zeros((hidden_node_num, 1))
    varphi2 = np.random.randn(output_node_num, output_node_num) * 0.01
    bata2 = np.zeros((output_node_num, 1))

    assert (W1.shape == (hidden_node_num, input_node_num))
    assert (b1.shape == (hidden_node_num, 1))
    assert (W2.shape == (output_node_num, hidden_node_num))
    assert (b2.shape == (output_node_num, 1))

    return {'W1': W1, 'b1': b1,
            'W2': W2, 'b2': b2,
            'varphi1':varphi1,
            'varphi2':varphi2,
            'bata1':bata1,
            'bata2':bata2}


def sigmoid(x):
    s = 1 / (1 + (np.exp(-x)))
    return s;


def forward_propagate(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    varphi1 = parameters['varphi1']
    varphi2 = parameters['varphi2']
    bata1 = parameters['bata1']
    bata2 = parameters['bata2']

    m = X.shape[1]

    z1 = np.dot(W1, X) + b1

    mean1 = np.mean(z1, axis=1, keepdims=True)
    std1 = np.std(z1, axis=1, keepdims=True)

    z1_norm = (z1-mean1)/(std1 + epsilon)
    z1_hat = np.dot(varphi1, z1_norm) + bata1

    a1 = np.tanh(z1_hat)

    z2 = np.dot(W2, a1) + b2

    mean2 = np.mean(z2, axis=1, keepdims=True)
    std2 = np.std(z2, axis=1, keepdims=True)

    z2_norm = (z2-mean2)/(std2 + epsilon)
    z2_hat = np.dot(varphi2, z2_norm) + bata2

    a2 = sigmoid(z2_hat)

    cache = {'A1':a1, 'Z1':z1, 'z1_norm':z1_norm,
            'z2_norm':z2_norm, 'A2':a2, 'Z2':z2, 'std1':std1, 'std2':std2}
    return cache


def cost_compute(A2, Y, parameters, lambd):
    m = Y.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2'];
    loss = Y * np.log(A2) + (1-Y)*np.log(1-A2+10**(-10))
    cost1 = -np.sum(loss)/m

    # l2_regularization_cost = lambd*(np.sum(np.square(W1)) + np.sum(np.square(W2)))/m/2
    # return cost1 + l2_regularization_cost
    return cost1


def backward_propagate(X, Y, parameters, cache, lambd):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    varphi1 = parameters['varphi1']
    varphi2 = parameters['varphi2']
    bata1 = parameters['bata1']
    bata2 = parameters['bata2']


    A1 = cache['A1']
    Z1 = cache['Z1']
    A2 = cache['A2']
    Z1_norm = cache['z1_norm']
    Z2_norm = cache['z2_norm']
    std1 = cache['std1']
    std2 = cache['std2']

    m = Y.shape[1]

    dZ_hat = A2 - Y
    dVarphi2 = np.dot(Z2_norm, dZ_hat.T)/m
    dBata2 = np.sum(dZ_hat, axis=1, keepdims=True)/m

    dZ2_norm = np.dot(varphi2, dZ_hat)
    dZ2 = dZ2_norm/(std2 + epsilon)

    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1_hat = np.multiply(np.dot(W2.T, dZ2), 1-A1**2)

    dVarphi1 = np.dot(dZ1_hat, Z1_norm.T)/m
    dBata1 = np.sum(dZ1_hat, axis=1, keepdims=True)/m

    dZ1_norm = np.dot(varphi1, dZ1_hat)
    dZ1 = dZ1_norm/(std1 + epsilon)

    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    # dZ2 = A2 - Y
    # dW2 = np.dot(dZ2, A1.T)/m + lambd / m * W2
    # db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    # dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    # dW1 = np.dot(dZ1, X.T)/m + lambd / m * W1
    # db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    varphi1 -= learning_rate * dVarphi1
    bata1 -= learning_rate * dBata1
    varphi2 -= learning_rate * dVarphi2
    bata2 -= learning_rate * dBata2

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
            'varphi1':varphi1,
            'varphi2':varphi2,
            'bata1':bata1,
            'bata2':bata2}


def do_predict(X, parameters):
    cache = forward_propagate(X, parameters)
    return np.around(cache['A2'])


def update_moving_average(std1, mean1, std2, mean2):
    MOVING_AVERAGE['std1'] = std1
    MOVING_AVERAGE['std2'] = std2
    MOVING_AVERAGE['mean1'] = mean1
    MOVING_AVERAGE['mean2'] = mean2


def build_model():
    X_original, Y_original = get_case_data(example_num)
    X, Y = process_data(X_original, Y_original)

    assert (Y.shape == (1, example_num))
    assert (X.shape == (2, example_num))

    parameters = initialize_parameters()

    for i in range(epoch_num):
        mini_batches = random_mini_batches(X, Y)
        for j in range(mini_batches.shape[0]):
            x_mini_batch, y_mini_batch = mini_batches[j]
            cache = forward_propagate(x_mini_batch, parameters=parameters)
            A2 = cache['A2']
            cost = cost_compute(A2, y_mini_batch, parameters, lambd)
            parameters = backward_propagate(x_mini_batch, y_mini_batch, parameters, cache, lambd)

        if i % 300 == 0:
            print(cost)

    def predict(X):
        return do_predict(X.T, parameters)

    matplotlib.rcParams['figure.figsize'] = (5.0, 4.0)
    plot_decision_boundary(X_original, Y_original, lambda x: predict(x))
    # plt.scatter(X_original[:, 0], X_original[:, 1], c=Y_original, s=20, cmap=plt.cm.Spectral)
    plt.title("model")
    plt.show()


def check_gradient():
    pass


def plot_decision_boundary(X, y, pred_func):
    """
    可视化结果
    :param X:
    :param y:
    :param pred_func: 模型的predict函数

    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])#
    Z = np.reshape(Z, xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral)


build_model()