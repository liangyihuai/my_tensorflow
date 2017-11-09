import tensorflow as tf;

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

# with tf.variable_scope("foo"):
# v = tf.get_variable("v", [1])

with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v1.name);

print(v == v1)
print(v);
print(v1);

# with tf.variable_scope("bar", reuse=True):
# v = tf.get_variable("v", [1])