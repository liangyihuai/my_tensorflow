# coding= utf8
import tensorflow as tf;

"""移动滑动模型
shadow_variable = decay*shadow_variable+(1-decay)*variable;
为了使得模型的前期训练更新更快，提供了num_updates参数来动态设置decay的大小。
num_updates表示已经更新到了第几步。
如果提供了该参数，每次的衰减率为：min{decay, (1+num_updates)/(10+num_updates)}.
the bigger num_updates, the bigger value of [(1+num_updates)/(10+num_updates)].
"""

x = tf.get_variable('x', shape=[1], dtype='float32', initializer=tf.constant_initializer(5));

y = tf.square(x);
MOVING_AVERAGE_DECAY = 0.99;
step_global = tf.Variable(0, trainable=False); # current step number.
exponential_moving_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, step_global);
# an operation to update the moving average value, which is trainable_variables.
exponential_moving_average_op = exponential_moving_average.apply(tf.trainable_variables());

LEARNING_RATE = 0.01;
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y);

# make the two operators of 'train_step' and 'exponential_moving_average_op' in a same group.
with tf.control_dependencies([train_step, exponential_moving_average_op]):
    # before this line executes, 'train_step' and 'exponential_moving_average_op' execute.
    train_op = tf.no_op(name='train');

# train_op = tf.group(train_step, exponential_moving_average_op); # the same

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    for i in range(1000):
        sess.run(train_op)
        if(i % 10) == 0:
            x_value = sess.run(exponential_moving_average.average(x));
            print("iteration num: %s, the x value is: %f"%(i, x_value));








