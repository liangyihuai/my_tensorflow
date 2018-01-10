import tensorflow as tf

log_dir = 'C:/Users/USER/Desktop/temp'  # 输出日志保存的路径

with tf.name_scope("foo"):
    with tf.variable_scope("var_scope"):
        v = tf.get_variable("var", [1])
        tf.summary.scalar('var', v)
with tf.name_scope("bar"):
    with tf.variable_scope("var_scope", reuse=True):
        v1 = tf.get_variable("var", [1])
        tf.summary.scalar('var', v)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

    assert(v.name == 'var_scope/var:0')
    assert(v1.name == 'var_scope/var:0')
    assert v1 == v