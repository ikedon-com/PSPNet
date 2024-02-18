from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import cv2
import matplotlib.pyplot as plt

A = [255,0,0]
B = [0,255,0]
C = [0,0,255]


COLOR_DICT = np.array([A,B,C])

def get_palette():
    palette = [
        [255,0,0],
        [0,255,0],
        [0,0,255],
    ]
    return np.asarray(palette)


def adjustData(img,mask,num_class):

    if np.max(img) > 1:
        img = img / 255.

    # マスク画像の方はOne-Hotベクトル化する
    # パレットカラーをndarrayで取得する
    palette = get_palette()

    # パレットとRGB値を比較してマスク画像をOne-hot化する
    onehot = np.zeros((mask.shape[0], 512,512, num_class), dtype=np.uint8)
    for i in range(num_class):
        # 現在カテゴリのRGB値を[R, G, B]の形で取得する
        cat_color = palette[i]

        # 画像が現在カテゴリ色と一致する画素に1を立てた(512,512)のndarrayを作る
        temp = np.where((mask[:, :, :, 0] == cat_color[0]) &
                        (mask[:, :, :, 1] == cat_color[1]) &
                        (mask[:, :, :, 2] == cat_color[2]), 1, 0)

        # 現在カテゴリに結果を割り当てる
        onehot[:, :, :, i] = temp

    return img ,onehot

def trainGenerator(batch_size, image_folder,  num_class = 3,save_to_dir=[None, None]):
    # 2つのジェネレータには同じパラメータを設定する必要がある
    data_gen_args = dict(
        zoom_range=[1, 1],  # 512*512の元画像上で256*256分を等倍で切り出したい
        rescale=None            # リスケールはadjustData()でやる
    )
    seed = 1                    # Shuffle時のSeedも共通にしないといけない

    # ImageDataGeneratorを準備
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # ジェネレータを準備
    image_generator = image_datagen.flow_from_directory(
        directory=image_folder,
        classes=['Input'],      # directoryの下のフォルダを1つ選び、
        class_mode=None,        # そのクラスだけを読み込んで、正解ラベルは返さない
        target_size=(512,512),
        batch_size=batch_size,
        seed=seed,
        save_to_dir=save_to_dir[0]
    )
    mask_generator = mask_datagen.flow_from_directory(
        directory=image_folder,
        classes=['GroundTruth'],
        class_mode=None,
        target_size=(512,512),
        batch_size=batch_size,
        seed=seed,
        save_to_dir=save_to_dir[1]
    )

    for (img, mask) in zip(image_generator, mask_generator):
        img, mask = adjustData(img, mask,num_class)
        yield img, mask


def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = True,as_gray = False,save_to_dir=[None, None]):
    data_gen_args = dict(
        zoom_range=[1, 1],  # 512*512の元画像上で256*256分を等倍で切り出したい
        rescale=None            # リスケールはadjustData()でやる
    )
    seed = 1                    # Shuffle時のSeedも共通にしないといけない

    # ImageDataGeneratorを準備
    test_datagen = ImageDataGenerator(**data_gen_args)
    test_generator = test_datagen.flow_from_directory(
        directory=test_path,
        classes=['Input'],      # directoryの下のフォルダを1つ選び、
        class_mode=None,        # そのクラスだけを読み込んで、正解ラベルは返さない
        target_size=(512,512),
        batch_size=1,
        seed=seed,
        save_to_dir=save_to_dir[0]
    )

    for (img) in zip(test_generator):
        if np.max(img) > 1:
            img = img / 255.
        yield img


# def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
#                     mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
#                     flag_multi_class = True,num_class = 3,save_to_dir = None,target_size = (256,256),seed = 1):
    
#     # 2つのジェネレータには同じパラメータを設定する必要がある
#     data_gen_args = dict(
#         width_shift_range=64,   # 元画像上でのシフト量128にzoom_ratioをかけてint型で設定する
#         height_shift_range=64,  # 同上
#         zoom_range=[0.5, 0.5],  # 512*512の元画像上で256*256分を等倍で切り出したい
#         horizontal_flip=True,
#         rescale=None            # リスケールはadjustData()でやる
#     )
#     seed = 1                    # Shuffle時のSeedも共通にしないといけない

#     # ImageDataGeneratorを準備
#     image_datagen = ImageDataGenerator(**data_gen_args)
#     mask_datagen = ImageDataGenerator(**data_gen_args)

#     # ジェネレータを準備
#     image_generator = image_datagen.flow_from_directory(
#         directory=train_path,
#         classes=['Input'],      # directoryの下のフォルダを1つ選び、
#         class_mode=None,        # そのクラスだけを読み込んで、正解ラベルは返さない
#         target_size=(512,512),
#         batch_size=batch_size,
#         seed=seed,
#     )
#     mask_generator = mask_datagen.flow_from_directory(
#         directory=train_path,
#         classes=['GroundTruth'],
#         class_mode=None,
#         target_size=(512,512),
#         batch_size=batch_size,
#         seed=seed,
#     )

#     for (img, mask) in zip(image_generator, mask_generator):

#         img, mask = adjustData(img, mask,flag_multi_class,num_class)
#         yield img, mask



# def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = True,as_gray = True):
#     for i in range(num_image):
#         img = io.imread(os.path.join(test_path,"Input/mCherry0041.png"),as_gray = as_gray)
#         img = img / 255
#         img = trans.resize(img,target_size)
#         img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
#         img = np.reshape(img,(1,)+img.shape)
#         yield img


# def geneTrainNpy(image_path,mask_path,flag_multi_class = True,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
#     image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
#     image_arr = []
#     mask_arr = []
#     for index,item in enumerate(image_name_arr):
#         img = io.imread(item,as_gray = image_as_gray)
#         img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
#         mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
#         mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
#         img,mask = adjustData(img,mask,flag_multi_class,num_class)
#         image_arr.append(img)
#         mask_arr.append(mask)
#     image_arr = np.array(image_arr)
#     mask_arr = np.array(mask_arr)
#     return image_arr,mask_arr


# def labelVisualize(num_class,color_dict,img):
#     img = img[:,:,0] if len(img.shape) == 3 else img
#     img_out = np.zeros(img.shape + (3,))
#     for i in range(num_class):
#         img_out[img == i,:] = color_dict[i]
#     return img_out / 255



# def saveResult(save_path,npyfile,flag_multi_class = True,num_class = 3):
#     for i,item in enumerate(npyfile):
#         img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
#         io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

def Save_image(img):
    #print(img.shape)

    img = np.argmax(img, axis=-1)[0]
    #print(predicted_mask.shape)
    #img = np.argmax(img,axis=1)
    print(img)

    #img = img[0]

    dst1 = np.zeros((256,256,3))
    dst1[img==0] = [1.0,0.0,0.0]#red
    dst1[img==1] = [0.0,1.0,0.0]#green
    dst1[img==2] = [0.0,0.0,1.0]#blue    

    plt.figure(figsize=(15,15))
    row = 2
    col = 2

    plt.imshow(dst1)
    plt.savefig("result/output2.png")
    plt.imsave("result/output2.png",dst1) #　imsaveにするとメモリとかがつかないからよい

    plt.clf()
    plt.close()