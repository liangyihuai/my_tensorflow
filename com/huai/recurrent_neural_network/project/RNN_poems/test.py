import numpy as np
nb_classes = 6
targets = np.array([2, 3, 4, 0]).reshape(-1)
one_hot_targets = np.eye(nb_classes)[targets]

print(one_hot_targets)

liangyihuai