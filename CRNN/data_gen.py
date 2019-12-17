import re
import os
import numpy as np
import random
import tensorflow as tf
from multiprocessing import Queue, Process
from utils import sparse_tuple_from, resize_image, label_to_array

from scipy.misc import imread

class DataManager(object):
    def __init__(
        self,
        batch_size,
        model_path,
        examples_path,
        max_image_width,
        train_test_ratio,
        max_char_count,
        char_vector,
        use_trdg,
        language,
    ):
        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception("Incoherent ratio!")

        self.char_vector = char_vector

        self.train_test_ratio = train_test_ratio
        self.max_image_width = max_image_width
        self.batch_size = batch_size
        self.model_path = model_path
        self.current_train_offset = 0
        self.examples_path = examples_path
        self.max_char_count = max_char_count
        self.use_trdg = use_trdg
        self.language = language
        #加载数据
        self.data, self.data_len = self.load_data()
        #打乱数据
        random.shuffle(self.data)
        self.len_train = int(train_test_ratio * self.data_len)
        self.len_test = self.data_len - self.len_train
        # 划分训练数据集
        self.train_data = self.data[:self.len_train]
        # 划分测试数据集
        self.test_data = self.data[self.len_train:]

    def load_data(self):
        """Load all the images in the folder
        """

        print("Loading data")

        examples = []

        count = 0
        skipped = 0
        train_path = os.listdir(self.examples_path)
        random.shuffle(train_path)
        for f in train_path:
            if len(f.split("_")[0]) > self.max_char_count:
                continue
            arr, initial_len = resize_image(
                imread(os.path.join(self.examples_path, f), mode="L"),
                self.max_image_width,
            )
            examples.append(
                (
                    arr,
                    f.split("_")[0],
                    label_to_array(f.split("_")[0], self.char_vector),
                )
            )
            count += 1

        return examples, len(examples)

    def generate_train_batches(self):
        #打乱训练集
        random.shuffle(self.train_data)
        train_batches = []
        for i in np.arange(int(np.floor(self.len_train / self.batch_size))):
            raw_batch_x, raw_batch_y, raw_batch_la = zip(
                *self.train_data[i * self.batch_size :(i + 1) * self.batch_size]
            )
            batch_y = np.reshape(np.array(raw_batch_y), (-1))

            batch_dt = sparse_tuple_from(np.reshape(np.array(raw_batch_la), (-1)))

            raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

            batch_x = np.reshape(
                np.array(raw_batch_x), (len(raw_batch_x), self.max_image_width, 32, 1)
            )

            train_batches.append((batch_y, batch_dt, batch_x))
        return train_batches

    def generate_test_batches(self):
        test_batches = []
        for i in range(int(np.floor(self.len_test / self.batch_size))) :
            index = self.indexes[i * self.batch_size:(i + 1) * self.batch_size]
            raw_batch_x, raw_batch_y, raw_batch_la = zip(
                *self.test_data[index]
            )

            batch_y = np.reshape(np.array(raw_batch_y), (-1))

            batch_dt = sparse_tuple_from(np.reshape(np.array(raw_batch_la), (-1)))

            raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

            batch_x = np.reshape(
                np.array(raw_batch_x), (len(raw_batch_x), self.max_image_width, 32, 1)
            )

            test_batches.append((batch_y, batch_dt, batch_x))
        return test_batches


