import numpy
import matplotlib.pyplot as plt

numpy.random.seed(seed = 1);

x = numpy.random.uniform(0, 1, 20)

def f(x):return x * 2;

noise_variance = 0.2;
noise = numpy.random.randn(x.shape[0]) * noise_variance;
t = f(x) + noise;


def neural_network(x, w): return x * w;


def cost(y, t): return ((t-y) ** 2).sum();


def gradient(w, x, t):
    return 2 * x * (neural_network(x, w) - t);


def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient(w_k, x, t).sum();


weight = 0.1;
learning_rate = 0.1;
number_of_iterations = 4;
weight_cost = [(weight, cost(neural_network(x, weight), t))];
for i in range(number_of_iterations):
    dw = delta_w(weight, x, t, learning_rate);
    weight = weight-dw;
    weight_cost.append((weight, cost(neural_network(x, weight), t)));


for i in range(0, len(weight_cost)):
    print('w({}): {:.4f} \t cost: {:.4f}'.format(i, weight_cost[i][0], weight_cost[i][1]));