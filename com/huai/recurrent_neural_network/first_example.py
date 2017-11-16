import numpy as np;

x = [1, 2]
state = [0.0, 0.0];

w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1;

for i in range(len(x)):
    before_activation = np.dot(state, w_cell_state)+x[i]*w_cell_input+b_cell;
    state = np.tanh(before_activation);

    final_output = np.dot(state, w_output) + b_output
    print('before activation: ', before_activation);
    print('state: ', state);
    print('output: ', final_output);











