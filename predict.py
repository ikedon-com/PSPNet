
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from unet import *
from data import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import cv2



if __name__ == '__main__':

    input_shape = (256, 256, 1)
    num_classes = 3
    learning_rate = 0.001
    batch_size = 1
    num_epochs = 10
    steps_per_epoch= 50
    train_image_paths = "3ch_dataset/train"
    validation_image_paths = "3ch_dataset/validation"
    test_image_paths = "3ch_dataset/test"


    # モデルの構築
    model = unet()

    # データセットの作成
    data_gen_args = dict(rotation_range=1,
                        width_shift_range=1,
                        height_shift_range=1,
                        shear_range=1,
                        zoom_range=1,
                        horizontal_flip=False,
                        fill_mode='nearest')

    test_dataset = trainGenerator(batch_size,test_image_paths,num_classes)

    loaded_model = tf.keras.models.load_model("segmentation_model.h5")
    # モデルの推論
    results = model.predict_generator(test_dataset,1,verbose=1)

    # 出力結果の可視化
    Save_image(results)
    #saveResult("result",results)