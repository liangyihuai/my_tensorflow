import tensorflow as tf
import numpy as np

variable1 = tf.Variable(0, name='variable1', dtype=tf.float32)#初始化v1变量
step = tf.Variable(0, trainable=False) #初始化step为0
ema = tf.train.ExponentialMovingAverage(0.99, step) #定义平滑类，设置参数以及step
maintain_averages_op = ema.apply([variable1]) #定义更新变量平均操作

for vars in tf.all_variables():
    print(vars.name)

print(ema.variables_to_restore())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(tf.assign(variable1, np.random.rand() * 10))
        sess.run(tf.assign(step, i))
        sess.run(maintain_averages_op)
        print(sess.run([variable1, ema.average(variable1), ema.variables_to_restore()]))






