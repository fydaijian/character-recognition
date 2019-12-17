import tensorflow as tf
import numpy as np
# def CNN(inputs):
#     """
#         Convolutionnal Neural Network part
#     """

    # 64 / 3 x 3 / 1 / 1

images = np.random.randn(10,400,32,1)
inputs = tf.convert_to_tensor(images, dtype=tf.float32)
conv1 = tf.layers.conv2d(
    inputs=inputs,
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    activation=tf.nn.relu,
)

# 2 x 2 / 1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# 128 / 3 x 3 / 1 / 1
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=128,
    kernel_size=(3, 3),
    padding="same",
    activation=tf.nn.relu,
)

# 2 x 2 / 1
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 256 / 3 x 3 / 1 / 1
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=256,
    kernel_size=(3, 3),
    padding="same",
    activation=tf.nn.relu,
)

# Batch normalization layer
bnorm1 = tf.layers.batch_normalization(conv3)

# 256 / 3 x 3 / 1 / 1
conv4 = tf.layers.conv2d(
    inputs=bnorm1,
    filters=256,
    kernel_size=(3, 3),
    padding="same",
    activation=tf.nn.relu,
)

# 1 x 2 / 1
pool3 = tf.layers.max_pooling2d(
    inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding="same"
)

# 512 / 3 x 3 / 1 / 1
conv5 = tf.layers.conv2d(
    inputs=pool3,
    filters=512,
    kernel_size=(3, 3),
    padding="same",
    activation=tf.nn.relu,
)

# Batch normalization layer
bnorm2 = tf.layers.batch_normalization(conv5)

# 512 / 3 x 3 / 1 / 1
conv6 = tf.layers.conv2d(
    inputs=bnorm2,
    filters=512,
    kernel_size=(3, 3),
    padding="same",
    activation=tf.nn.relu,
)

# 1 x 2 / 2
pool4 = tf.layers.max_pooling2d(
    inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding="same"
)

# 512 / 2 x 2 / 1 / 0
conv7 = tf.layers.conv2d(
    inputs=pool4,
    filters=512,
    kernel_size=(2, 2),
    padding="valid",
    activation=tf.nn.relu,
)
reshaped_cnn_output = tf.squeeze(conv7 , [2])

    # return conv7
images = np.random.randn(10,400,32,1)
images = tf.convert_to_tensor(images, dtype=tf.float32)
sess = tf.Session()

init_g = tf.global_variables_initializer()

sess.run(init_g)
conv_shape = tf.shape(conv7)
print(sess.run(conv_shape ))
output_shape = tf.shape(reshaped_cnn_output)

print(sess.run(output_shape ))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     sess.run( CNN(images))
