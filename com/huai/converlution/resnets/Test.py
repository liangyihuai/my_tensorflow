from com.huai.converlution.resnets.resnets_utils import *
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

path = 'D:/LiangYiHuai/deepleanring/resnets/datasets'
orig_data = load_dataset(path)

X_train, Y_train, X_test, Y_test = process_orig_datasets(orig_data)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

print('x train max, ', np.max(X_train), '; x train min, ', np.min(X_train))
print('x test max, ', np.max(X_test), '; x test min, ', np.min(X_test))

mini_batch = random_mini_batches(X_train, Y_train)
print(mini_batch[0][0].shape)

mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=16)

X_mini_batch, Y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]


