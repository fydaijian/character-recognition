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

data_dir = r"C:\Users\USST-HUANG2\Desktop\ocr\result_label"
model_dir = 'model_conv_test_05/'
char_set_string = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
use_trdg = True
language = "en"
# 模型训练
def train():

    batch_size=32
    max_image_width=150
    train_test_ratio=0.9
    restore=True
    iteration_count=15000

    crnn = CRNN(
        batch_size,
        model_dir,
        data_dir,
        max_image_width,
        train_test_ratio,
        restore,
        char_set_string,
        use_trdg,
        language


    )

    crnn.train(iteration_count)
train()
