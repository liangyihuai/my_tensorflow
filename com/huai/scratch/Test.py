import numpy as np




a = np.array([1,2,3,4])
a_pad = np.pad(a, (1, 1), mode="constant", constant_values=(0, 0))
print(a_pad)

matric = np.array([[[1,2], [3,4], [5,6]], [[7,8],[9,10],[11,12]]])
print(matric.shape)
matric_pad = np.pad(matric, ((1, 1), (1, 1),(0, 0)), mode='constant', constant_values=(0,0))
print(matric_pad)
b = 1

