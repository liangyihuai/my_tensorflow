import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def main():
    mnist = input_data.read_data_sets('dir_with_mnist_data_files', one_hot=False)

    print("train example number: ", mnist.train.num_examples)
    print("train image shape: ", mnist.train.images.shape)
    # print(mnist.train.images[0:1])
    print("train label shape: ", mnist.train.labels.shape)
    print("test image shape: ", mnist.test.images.shape)
    print("test image label shape: ", mnist.test.labels.shape)
    print("validataion image shape: ", mnist.validation.images.shape)
    print("validataion label shape: ", mnist.validation.labels.shape)

    batch_xs, batch_ys = mnist.train.next_batch(batch_size=100)
    print("batch_xs shape is , batch_ys shape is ", batch_xs.shape, batch_ys.shape)

    print("max value in image: ", np.max(mnist.test.images))
    print("min value in image", np.min(mnist.test.images))
    print("max value in label", np.max(mnist.test.labels))
    print("min value in label", np.min(mnist.test.labels))

    one_hot_mnist = input_data.read_data_sets('dir_with_mnist_data_files', one_hot=True)
    print("train image shape of one_hot mnist shape: ", one_hot_mnist.train.images.shape)
    print("train label shape of one_hot mnist shape: ", one_hot_mnist.train.labels.shape)


if __name__=='__main__':
    main()