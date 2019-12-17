import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize, imsave
char_vector = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def resize_image(im_arr, input_width):
    """Resize an image to the "good" input size
    """
    r, c = np.shape(im_arr)
    if c > input_width:
        c = input_width
        ratio = float(input_width) / c
        final_arr = imresize(im_arr, (int(32 * ratio), input_width))
    else:
        final_arr = np.zeros((32, input_width))
        ratio = 32.0 / r
        im_arr_resized = imresize(im_arr, (32, int(c * ratio)))
        # final_arr[
        #     :, 0 : min(input_width, np.shape(im_arr_resized)[1])
        # ] = im_arr_resized[:, 0:input_width]

    return   im_arr_resized, c

def label_to_array(label, char_vector):
    try:
        return [char_vector.index(x) for x in label]
    except Exception as ex:
        print(label)
        raise ex

def load_data(examples_path):
    """Load all the images in the folder
    """

    print("Loading data")

    examples = []

    count = 0
    skipped = 0
    for f in os.listdir(examples_path):
        if len(f.split("_")[0]) > 512:
            continue
        arr, initial_len = resize_image(
            imread(os.path.join(examples_path, f), mode="L"),
            150
        )

        rows, cols = arr.shape
        #旋转90度
        # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
        # dst = cv2.warpAffine(arr, M, (cols, rows))  # 仿射变换，以后再说
        # print(M)
        pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.7]])
        #仿射
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(arr, M, (cols, rows))

        plt.imshow(dst, cmap="gray")
        plt.show()
        examples.append(
            (
                arr,
                f.split("_")[0],
                label_to_array(f.split("_")[0], char_vector),
            )
        )
        count += 1

    return examples, len(examples)
load_data(r"C:\Users\USST-HUANG2\Desktop\ocr\ocr_datasetGen")