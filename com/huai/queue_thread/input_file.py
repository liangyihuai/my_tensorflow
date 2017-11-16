import tensorflow as tf;

def _init64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]));

num_shards = 2;
instances_per_shard = 2;
for i in range(num_shards):
    file_name = (""%(i, num_shards))