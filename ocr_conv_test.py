import cv2
import numpy as np
import random
from sys import path
path.append(r'C:\Users\USST-HUANG2\Desktop\ocr\CRNN')
from scipy.misc import imread
import os
from crnn import CRNN
from PIL import Image,ImageFont,ImageDraw
import tensorflow as tf
import utils
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

char_set_string = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(char_set_string) + 1
def crnn_network(max_width, batch_size):
    # 双向RNN
    def BidirectionnalRNN(inputs, seq_len):
        # rnn-1
        with tf.variable_scope(None, default_name="bidirectional-rnn-1"):
            # Forward
            lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
            # Backward
            lstm_bw_cell_1 = rnn.BasicLSTMCell(256)
            inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, seq_len, dtype=tf.float32)
            inter_output = tf.concat(inter_output, 2)

        # rnn-2
        with tf.variable_scope(None, default_name="bidirectional-rnn-2"):
            # Forward
            lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
            # Backward
            lstm_bw_cell_2 = rnn.BasicLSTMCell(256)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, seq_len, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)
        return outputs

    # CNN，用于提取特征
    def CNN(inputs):
        # 64 / 3 x 3 / 1 / 1
        conv1 = tf.layers.conv2d(inputs=inputs, filters = 64, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
        # 2 x 2 / 1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        # 128 / 3 x 3 / 1 / 1
        conv2 = tf.layers.conv2d(inputs=pool1, filters = 128, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
        # 2 x 2 / 1
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # 256 / 3 x 3 / 1 / 1
        conv3 = tf.layers.conv2d(inputs=pool2, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
        # Batch normalization layer
        bnorm1 = tf.layers.batch_normalization(conv3)
        # 256 / 3 x 3 / 1 / 1
        conv4 = tf.layers.conv2d(inputs=bnorm1, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
        # 1 x 2 / 1
        pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding="same")
        # 512 / 3 x 3 / 1 / 1
        conv5 = tf.layers.conv2d(inputs=pool3, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
        # Batch normalization layer
        bnorm2 = tf.layers.batch_normalization(conv5)
        # 512 / 3 x 3 / 1 / 1
        conv6 = tf.layers.conv2d(inputs=bnorm2, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)
        # 1 x 2 / 2
        pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding="same")
        # 512 / 2 x 2 / 1 / 0
        conv7 = tf.layers.conv2d(inputs=pool4, filters = 512, kernel_size = (2, 2), padding = "valid", activation=tf.nn.relu)
        return conv7

    # 定义输入、输出、序列长度
    inputs = tf.placeholder(tf.float32, [batch_size, max_width, 32, 1])
    targets = tf.sparse_placeholder(tf.int32, name='targets')
    seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

    # 卷积层提取特征
    cnn_output = CNN(inputs)
    reshaped_cnn_output = tf.reshape(cnn_output, [batch_size, -1, 512])
    max_char_count = reshaped_cnn_output.get_shape().as_list()[1]

    # 循环层处理序列
    crnn_model = BidirectionnalRNN(reshaped_cnn_output, seq_len)
    logits = tf.reshape(crnn_model, [-1, 512])

    # 转录层预测结果
    W = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]), name="b")
    logits = tf.matmul(logits, W) + b
    logits = tf.reshape(logits, [batch_size, -1, NUM_CLASSES])
    logits = tf.transpose(logits, (1, 0, 2))

    # 定义损失函数、优化器
    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    # 初始化
    init = tf.global_variables_initializer()

    return inputs, targets, seq_len, logits, dense_decoded, optimizer, acc, cost, max_char_count, init

# CRNN 识别文字
# 输入：图片路径
# 输出：识别文字结果
def predict(img_path):

    # 定义模型路径、最长图片宽度
    batch_size = 1
    model_path = 'model_conv_test_01/'
    max_image_width = 400

    # 创建会话
    __session = tf.Session()
    with __session.as_default():
        (
            __inputs,
            __targets,
            __seq_len,
            __logits,
            __decoded,
            __optimizer,
            __acc,
            __cost,
            __max_char_count,
            __init
        ) = crnn_network(max_image_width, batch_size)
        __init.run()

    # 加载模型
    with __session.as_default():
        __saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(model_path)
        if ckpt:
            __saver.restore(__session, ckpt)

    # 读取图片作为输入
    arr, initial_len = utils.resize_image(imread(img_path, mode="L"),max_image_width)
    batch_x = np.reshape(
        np.array(arr),
        (-1, max_image_width, 32, 1)
    )

    # 利用模型识别文字
    with __session.as_default():
        decoded = __session.run(
            __decoded,
            feed_dict={
                __inputs: batch_x,
                __seq_len: [__max_char_count] * batch_size
            }
        )
        pred_result = utils.ground_truth_to_word(decoded[0],char_set_string)

    return pred_result
contest = predict(r'C:\Users\USST-HUANG2\Desktop\ocr\result_label\42G1_004.jpg')
print(contest)
