import cv2
import numpy as np
import random
from sys import path
path.append(r'C:\Users\USST-HUANG2\Desktop\ocr\CRNN')

import cv2
import numpy as np
import random
from crnn import CRNN
from PIL import Image,ImageFont,ImageDraw
import tensorflow as tf
import utils
from tensorflow.contrib import rnn

data_dir = r"C:\Users\USST-HUANG2\Desktop\ocr\result_validation"
model_dir = 'model_conv_test_03/'
char_set_string = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
use_trdg = None
language = "en"
# 模型训练


def test():
    # 设置基本属性
    batch_size = 32
    max_image_width = 150
    restore = True
    # 初始化CRNN
    crnn = CRNN(
        batch_size,
        model_dir,
        data_dir,
        max_image_width,
        0,
        restore, char_set_string,
        use_trdg,
        language
    )
    # 测试模型
    crnn.test()



test()