import tensorflow as tf
from resnets_utils import *
import numpy as np

TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)


def identity_block(X, kernel_size, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("id_block_stage"+str(stage)):
        filter1, filter2, filter3 = filters
        X_shortcut = X

        # First component of main path
        conv1 = tf.layers.conv2d(X, filter1,
                 kernel_size=(1, 1), strides=(1, 1),
                                 name=conv_name_base+'2a')
        # batch_norm1 = tf.layers.batch_normalization(conv1, axis=3, name=bn_name_base+'2a', training=TRAINING)
        activate1 = tf.nn.relu(conv1)

        # Second component of main path
        conv2 = tf.layers.conv2d(activate1, filter2,
                                 (kernel_size, kernel_size),
                                 padding='same',
                             name=conv_name_base+'2b')
        # batch_norm2 = tf.layers.batch_normalization(conv2, axis=3, name=bn_name_base+'2b', training=TRAINING)
        activate2 = tf.nn.relu(conv2)

        # Third component of main path
        conv3 = tf.layers.conv2d(activate2, filter3, kernel_size=(1, 1),
                             name=conv_name_base+'2c')
        # batch_norm3 = tf.layers.batch_normalization(conv3, axis=3, name=bn_name_base + '2c', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(conv3, X_shortcut)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def convolutional_block(X, kernel_size, filters, stage, block, stride = 2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    stride -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):

        # Retrieve Filters
        filter1, filter2, filter3 = filters

        # Save the input value
        X_shortcut = X

        # First component of main path
        conv1 = tf.layers.conv2d(X, filter1,
                                 kernel_size=(1, 1),
                                 strides=(stride, stride),
                                 name=conv_name_base+'2a')
        # batch_norm1 = tf.layers.batch_normalization(conv1, axis=3, name=bn_name_base+'2a', training=TRAINING)
        activate1 = tf.nn.relu(conv1)

        # Second component of main path
        conv2 = tf.layers.conv2d(activate1, filter2, (kernel_size, kernel_size), name=conv_name_base + '2b',
                             padding='same')
        # batch_norm2 = tf.layers.batch_normalization(conv2, axis=3, name=bn_name_base + '2b', training=TRAINING)
        activate2 = tf.nn.relu(conv2)

        # Third component of main path
        conv3 = tf.layers.conv2d(activate2, filter3,
                                 (1, 1),
                             name=conv_name_base + '2c')
        # batch_norm3 = tf.layers.batch_normalization(conv3, axis=3, name=bn_name_base + '2c', training=TRAINING)

        # SHORTCUT PATH
        X_shortcut = tf.layers.conv2d(
            X_shortcut, filter3, (1,1),
            strides=(stride, stride),
            name=conv_name_base + '1')
        # X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(X_shortcut, conv3)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result



def ResNet50_reference(X, classes= 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:

    Returns:
    """

    X_padding = tf.pad(X, tf.constant([[0, 0],[3, 3,], [3, 3], [0, 0]]), "CONSTANT")

    assert(X_padding.shape == (X_padding.shape[0], 70, 70, 3))

    # stage 1
    conv_stage1 = tf.layers.conv2d(X_padding, filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1')
    # batch_norm1_stage1 = tf.layers.batch_normalization(conv_stage1, axis=3, name='bn_conv1')
    activate_stage1 = tf.nn.relu(conv_stage1)
    pool_stage1 = tf.layers.max_pooling2d(
        activate_stage1, pool_size=(3, 3),
                strides=(2, 2))
    assert(pool_stage1.shape == (pool_stage1.shape[0], 15, 15, 64))

    # stage 2
    conv_block_stage2 = convolutional_block(pool_stage1, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', stride=1)
    ide_block_stage2_1 = identity_block(conv_block_stage2, 3, [64, 64, 256], stage=2, block='b')
    ide_block_stage2_2 = identity_block(ide_block_stage2_1, 3, [64, 64, 256], stage=2, block='c')

    # stage 3
    conv_block_stage3 = convolutional_block(ide_block_stage2_2, kernel_size=3, filters=[128,128,512],
                                            stage=3, block='a', stride=2)
    iden_block_stage3_1 = identity_block(conv_block_stage3, 3, [128,128,512], stage=3, block='b')
    iden_block_stage3_2 = identity_block(iden_block_stage3_1, 3, [128,128,512], stage=3, block='c')
    iden_block_stage3_3 = identity_block(iden_block_stage3_2, 3, [128,128,512], stage=3, block='d')

    # stage 4
    conv_block_stage4 = convolutional_block(iden_block_stage3_3, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', stride=2)
    ide_block_stage4_1 = identity_block(conv_block_stage4, 3, [256, 256, 1024], stage=4, block='b')
    ide_block_stage4_2 = identity_block(ide_block_stage4_1, 3, [256, 256, 1024], stage=4, block='c')
    ide_block_stage4_3 = identity_block(ide_block_stage4_2, 3, [256, 256, 1024], stage=4, block='d')
    ide_block_stage4_4 = identity_block(ide_block_stage4_3, 3, [256, 256, 1024], stage=4, block='e')
    ide_block_stage4_5 = identity_block(ide_block_stage4_4, 3, [256, 256, 1024], stage=4, block='f')

    # stage 5
    conv_block_stage5 = convolutional_block(ide_block_stage4_5,
                    kernel_size=3,
                     filters=[512, 512, 2048], stage=5, block='a', stride=2)
    ide_block_stage5_1 = identity_block(conv_block_stage5, 3, [512, 512, 2048], stage=5, block='b')
    ide_block_stage5_2 = identity_block(ide_block_stage5_1, 3, [512, 512, 2048], stage=5, block='c')

    ave_pool1 = tf.layers.average_pooling2d(
        ide_block_stage5_2, pool_size=(2, 2), strides=(1,1))

    flatten = tf.layers.flatten(ave_pool1, name='flatten')
    dense1 = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
    logits = tf.layers.dense(dense1, units=6, activation=tf.nn.softmax)
    return logits



def main():
    path = 'D:/LiangYiHuai/deepleanring/resnets/datasets'
    orig_data = load_dataset(path)

    global TRAINING

    X_train, Y_train, X_test, Y_test = process_orig_datasets(orig_data)
    """
    number of training examples = 1080
    number of test examples = 120
    X_train shape: (1080, 64, 64, 3)
    Y_train shape: (1080, 6)
    X_test shape: (120, 64, 64, 3)
    Y_test shape: (120, 6)"""


    m, H_size, W_size, C_size = X_train.shape
    classes = 6
    assert ((H_size, W_size, C_size) == (64, 64, 3))

    X = tf.placeholder(tf.float32, shape=(None, H_size, W_size, C_size), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, classes), name='Y')

    logits = ResNet50_reference(X)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        assert (X_train.shape == (X_train.shape[0], 64, 64, 3))
        assert (Y_train.shape[1] == 6)
        mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=16)

        for i in range(20):
            X_mini_batch, Y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]
            _, cost_sess = sess.run([train_op, loss], feed_dict={X: X_mini_batch, Y: Y_mini_batch})

            print(cost_sess)


        sess.run(tf.assign(TRAINING, False))

        training_acur = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
        testing_acur = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
        print("traing acurracy: ", training_acur)
        print("testing acurracy: ", testing_acur)


if __name__ == '__main__':
    main()


