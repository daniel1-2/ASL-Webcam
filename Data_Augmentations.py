import tensorflow as tf
import os
import random
import keras
from tensorflow.keras import layers

training_data_path = "data_archive/asl_alphabet_train"


for dirlist in os.listdir(training_data_path):
    for root, directories, filenames in os.walk(os.path.join(training_data_path, dirlist)):
        print("Inside Folder", dirlist, "Consist :", len(filenames), "Imageset")
        for filename in filenames:
          if random.randint() % 2 == 0:
            IMG_WIDTH = random.randint(50, 1080)
            IMG_LENGTH= random.randint(50, 1080)

          resize_and_rescale = tf.keras.Sequential([
            layers.Resizing(IMG_LENGTH, IMG_WIDTH),
            layers.Rescaling(1./255)])