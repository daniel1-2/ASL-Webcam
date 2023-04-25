import tensorflow as tf
import os
import random
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import keras
from tensorflow.keras import layers

training_data_path = "data_archive/asl_alphabet_train"
image = cv2.imread("A1test.jpg")
IMG_WIDTH = random.randint(1, 800)
IMG_LENGTH= random.randint(1, 800)
resized = tf.image.resize_with_crop_or_pad(image, 200, 200)
resized = np.squeeze(resized)
plt.imsave("resized12.jpg", resized)
resized_big = tf.image.resize_with_crop_or_pad(image, 500, 500)
resized_big = np.squeeze(resized_big)
plt.imsave("resized_big12.jpg", resized_big)
# for dirlist in os.listdir(training_data_path):
#     for root, directories, filenames in os.walk(os.path.join(training_data_path, dirlist)):
#         print("Inside Folder", dirlist, "Consist :", len(filenames), "Imageset")
#         for filename in filenames:
#             image_path = os.path.join(root, filename)
#             image = cv.imread(image_path)
#             if random.randint() % 2 == 0:
#                 IMG_WIDTH = random.randint(50, 1080)
#                 IMG_LENGTH= random.randint(50, 1080)
#                 resize_and_rescale = layers.Resizing(image, IMG_LENGTH, IMG_WIDTH)