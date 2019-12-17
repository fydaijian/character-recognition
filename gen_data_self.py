import re
import os
import numpy as np
import random
import tensorflow as tf
from sys import path
path.append(r'C:\Users\USST-HUANG2\Desktop\ocr\CRNN')
from multiprocessing import Queue, Process
from utils import sparse_tuple_from, resize_image, label_to_array
import matplotlib.pyplot as plt

from scipy.misc import imread
from trdg.generators import GeneratorFromDict
def batch_generator( queue):
    """Takes a queue and enqueue batches in it
    """

    generator = GeneratorFromDict(language="en")
    while True:
        batch = []
        while len(batch) <32:
            img, lbl = generator.next()
            plt.imshow(img)
            plt.show()
            print(lbl)
        #     batch.append(
        #         (
        #             resize_image(np.array(img.convert("L")),150)[
        #                 0
        #             ],
        #             lbl,
        #             label_to_array(lbl, self.char_vector),
        #         )
        #     )
        #
        # raw_batch_x, raw_batch_y, raw_batch_la = zip(*batch)
        #
        # batch_y = np.reshape(np.array(raw_batch_y), (-1))
        #
        # batch_dt = sparse_tuple_from(np.reshape(np.array(raw_batch_la), (-1)))
        #
        # raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)
        #
        # batch_x = np.reshape(
        #     np.array(raw_batch_x), (len(raw_batch_x), self.max_image_width, 32, 1)
        # )
        # if queue.qsize() < 20:
        #     queue.put((batch_y, batch_dt, batch_x))
        # else:
        #     pass
        #

# def multiprocess_batch_generator(self):
#     """Returns a batch generator to use in training
#     """
#
#     q = Queue()
#     processes = []
#     for i in range(2):
#         processes.append(Process(target=self.batch_generator, args=(q,)))
#         processes[-1].start()
#     while True:
#         yield q.get()
batch_generator(1)