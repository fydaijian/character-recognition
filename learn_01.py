import tensorflow as tf
import  numpy as np
import numpy as np
import tensorflow as tf
x = tf.placeholder(tf.float32, shape=( None))
a=[[1,2,3],[1,2,3]]
with tf.Session() as sess:
  output = sess.run(x, feed_dict={x: a})
  print(output)
# def dynamic_rnn():
#     x = np.random.randn(3,6,4)
#     x[1, 4:] = 0
#     x_lengths = [6, 4, 6]
#     rnn_hidden_size = 5
#     cell = tf.contrib.rnn.BasicLSTMCell(num_units= rnn_hidden_size)
#     outputs, last_states = tf.nn.dynamic_rnn( cell = cell,
#                                               dtype = tf.float64,
#                                               sequence_length = x_lengths,
#                                               inputs = x)
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         o1, s1 =session.run([outputs, last_states])
#         print(o1)
#
# dynamic_rnn()
#
# import tensorflow as tf
# import numpy as np
# x = np.random.randn(2, 9, 8)
# x[1,6:] = 0
# x_length = tf.float64
# cell = tf.contrib.rnn.BasicLSTMCell(num_units = 64)
# outputs, last_states = tf.nn.dynamic_rnn(cell = cell, dtype= tf.float64,
#                                          sequence_length = x_length)
# with tf.Session() as session:
#     session.run(tf.global_variables_initializer())
#     o1, s1 = session.run([outputs, last_states])