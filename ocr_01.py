import numpy as np
import cv2
import os
import tensorflow as tf
import random
import time
import datetime
from captcha.image import ImageCaptcha
from PIL import Image,ImageFont,ImageDraw

# 定义一些常量
# 元数据集
DIGITS = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
               "A":10, "B":11, "C":12, "D":13, "E":14, "F":15, "G":16,
               "H":17, "I":18, "J":19, "K":20, "L":21, "M":22, "N":23,
               "O":24, "P":25, "Q":26, "R":27, "S":28, "T":29, "U":30,
               "V":31, "W":32, "X":33, "Y":34, "Z":35}

DIGITS_decode = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
               10:"A", 11:"B", 12:"C", 13:"D", 14:"E", 15:"F", 16:"G",
               17:"H", 18:"I", 19:"J", 20:"K", 21:"L", 22:"M", 23:"N",
               24:"O", 25:"P", 26:"Q", 27:"R", 28:"S", 29:"T", 30:"U",
               31:"V", 32:"W", 33:"X", 34:"Y", 35:"Z"}
# 图片大小
OUTPUT_SHAPE = (32, 256)

# 训练最大轮次
num_epochs = 50000
num_hidden = 128
num_layers = 2
num_classes = len(DIGITS) + 1

# 初始化学习速率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9

BATCHES = 10
BATCH_SIZE = 10
TRAIN_SIZE = BATCHES * BATCH_SIZE

data_dir = r'C:\Users\USST-HUANG2\Desktop\ocr\data'
model_dir = 'model/'

#椒盐噪声
def img_salt_pepper_noise(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.randint(0,1)==0:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

# 随机生成不定长数据
def gen_text_2(cnt):
    font_path = '/data/work/tensorflow/fonts/arial.ttf'
    font_size = 30
    font=ImageFont.truetype(font_path,font_size)

    for i in range(cnt):
        rnd = random.randint(1, 10)
        text = ''
        for j in range(rnd):
            text = text + DIGITS[random.randint(0, len(DIGITS) - 1)]
        img=Image.new("RGB",(256,32))
        draw=ImageDraw.Draw(img)
        draw.text((1,1),text,font=font,fill='white')

        img=np.array(img)
        img = img_salt_pepper_noise(img, float(random.randint(1,10)/100.0))

        cv2.imwrite(data_dir + text + '_' + str(i+1) + '.jpg',img)

# 稀疏矩阵转序列
def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = DIGITS_decode[spars_tensor[1][m]]
        decoded.append(str)
    return decoded

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []

    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)

    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result

# 准确性评估
# 输入：预测结果序列 decoded_list ,目标序列 test_targets
# 返回：准确率
def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)

    # 正确数量
    true_numer = 0

    # 预测序列与目标序列的维度不一致，说明有些预测失败，直接返回
    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return

    # 比较预测序列与结果序列是否一致，并统计准确率
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    accuracy = true_numer * 1.0 / len(original_list)
    print("Test Accuracy:", accuracy)
    return accuracy

# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        for i in seq:

            values.append(DIGITS[str(i)])

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

# 将文件和标签读到数组
def get_file_text_array():
    file_name_array=[]
    text_array=[]

    for parent, dirnames, filenames in os.walk(data_dir):
        file_name_array=filenames

    for f in file_name_array:
        text = f.split('_')[0]
        text_array.append(text)

    return file_name_array,text_array

# 生成一个训练batch
def get_next_batch(file_name_array,text_array,batch_size=128):
    inputs = np.zeros([batch_size, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
    codes = []

    # 获取训练样本
    for i in range(batch_size):
        index = random.randint(0, len(file_name_array) - 1)
        image = cv2.imread(os.path.join(data_dir , file_name_array[index]))

        image = cv2.resize(image, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]), 3)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        text = text_array[index]
        inputs[i, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
        codes.append(list(text))

    targets = [np.asarray(i) for i in codes]
    sparse_targets = sparse_tuple_from(targets)
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]

    return inputs, sparse_targets, seq_len

def get_train_model():
    inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]])  # old
    targets = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])

    # 定义LSTM网络
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]
    outputs = tf.reshape(outputs, [-1, num_hidden])
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # 转置矩阵
    logits = tf.transpose(logits, (1, 0, 2))

    return logits, inputs, targets, seq_len, W, b

def train():
    # 获取训练样本数据
    file_name_array, text_array = get_file_text_array()

    # 定义学习率
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    # 获取网络结构
    logits, inputs, targets, seq_len, W, b = get_train_model()

    # 设置损失函数
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    # 设置优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        for curr_epoch in range(num_epochs):
            train_cost = 0
            train_ler = 0
            for batch in range(BATCHES):
                # 训练模型
                train_inputs, train_targets, train_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
                feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
                b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
                    [loss, targets, logits, seq_len, cost, global_step, optimizer], feed)

                # 评估模型
                if steps > 0 and steps % REPORT_STEPS == 0:
                    test_inputs, test_targets, test_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
                    test_feed = {inputs: test_inputs,targets: test_targets,seq_len: test_seq_len}
                    dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
                    report_accuracy(dd, test_targets)

                    # 保存识别模型
                    save_path = saver.save(session, os.path.join(model_dir , "lstm_ctc_model.ctpk"),global_step=steps)

                c = b_cost
                train_cost += c * BATCH_SIZE

            train_cost /= TRAIN_SIZE
            # 计算 loss
            train_inputs, train_targets, train_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
            val_feed = {inputs: train_inputs,targets: train_targets,seq_len: train_seq_len}
            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

            log = "{} Epoch {}/{}, steps = {}, train_cost = {:.3f}, val_cost = {:.3f}"
            print(log.format(curr_epoch, curr_epoch + 1, num_epochs, steps, train_cost, val_cost))


def predict(image_path):
    # 获取网络结构
    logits, inputs, targets, seq_len, W, b = get_train_model()
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)


    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        image = cv2.imread(image_path)
        # 图像预处理
        image = cv2.resize(image, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imshow("a", image)
        cv2.waitKey(0)

    #     pred_inputs = np.zeros([1, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
    #     pred_inputs[0, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
    #     pred_seq_len = np.ones(1) * OUTPUT_SHAPE[1]
    #     # 模型预测
    #     pred_feed = {inputs: pred_inputs, seq_len: pred_seq_len}
    #     dd, log_probs = sess.run([decoded[0], log_prob], pred_feed)
    #     # 识别结果转换
    #     detected_list = decode_sparse_tensor(dd)[0]
    #     detected_text = ''
    #     for d in detected_list:
    #         detected_text = detected_text + d
    #
    #
    # return detected_text


if __name__ == "__main__":
    a=predict(r"C:\Users\USST-HUANG2\Desktop\ocr\conv_data\155039-8.48-45.74-0.jpg")
    print(a)
