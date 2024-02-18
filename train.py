import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from model import *
from data import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import cv2


def visualize_segmentation(image, mask, num_classes):
    # マスクをカラーマップに変換
    cmap = plt.cm.get_cmap('jet', num_classes)
    mask_colored = cmap(mask)

    # 元の画像とマスクを重ねて表示
    plt.imshow(image)
    plt.imshow(mask_colored, alpha=0.5)
    plt.axis('off')
    plt.show()


def overlay_segmentation(image, mask, num_classes, alpha=0.5):
    # マスクをカラーマップに変換
    cmap = plt.cm.get_cmap('jet', num_classes)
    mask_colored = cmap(mask)

    # 画像とマスクを重ねる
    overlaid = cv2.addWeighted(image, 1-alpha, mask_colored, alpha, 0)

    return overlaid


if __name__ == '__main__':

    input_shape = (512,512, 1)
    num_classes = 3
    learning_rate = 0.001
    batch_size = 4
    num_epochs = 10
    steps_per_epoch= 70
    train_image_paths = "3ch_dataset/train"
    validation_image_paths = "3ch_dataset/validation"
    test_image_paths = "3ch_dataset/test"


    # モデルの構築
    #model = unet_model(input_shape, num_classes)
    model = PSPNet()
    #model.compile(optimizer=Adam(learning_rate), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # データセットの作成
    data_gen_args = dict(rotation_range=1,
                        width_shift_range=1,
                        height_shift_range=1,
                        shear_range=1,
                        zoom_range=1,
                        horizontal_flip=False,
                        fill_mode='nearest')

    train_dataset = trainGenerator(batch_size,train_image_paths,num_classes)
    validation_dataset = trainGenerator(batch_size,validation_image_paths,num_classes)
    test_dataset = trainGenerator(batch_size,test_image_paths,num_classes)


    # トレーニングの実行

    model = Custum_class(model = model)
    model.compile(optimizer=Adam(lr=0.001),
                    loss='categorical_crossentropy',  # 多クラス単一ラベル分類
                    metrics=[tf.keras.metrics.CategoricalAccuracy()],                
                    alpha=0.1,
                    temperature=10,
                    run_eagerly=True,
                    model_loss_fn =  tf.keras.losses.CategoricalCrossentropy(), 
                    tversky_loss_fn = tversky_loss,
                    dice_loss_fn = dice_loss,                
                    )

    model_checkpoint = ModelCheckpoint('deeplab_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit(train_dataset,steps_per_epoch=steps_per_epoch,epochs=num_epochs,callbacks=[model_checkpoint])

    # モデルの保存
    model.save('segmentation_model.h5')



    # モデルの推論
    results = model.predict_generator(test_dataset,1,verbose=1)

    # 最も確信度の高いクラスを予測結果として選択
    #predicted_mask = np.argmax(results, axis=-1)[0]

    # 出力結果の可視化
    Save_image(results)
    #saveResult("result",results)