# import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.layers import Activation,Conv2D,MaxPooling2D,BatchNormalization,Input,DepthwiseConv2D,add,Dropout,AveragePooling2D,Concatenate
from keras.models import Model                                                          
import keras.backend as K
from keras.layers import Layer,InputSpec
from keras.utils import conv_utils

from keras.models import Model
from keras.layers import Input, MaxPooling2D, Concatenate, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras import regularizers
from keras.optimizers import *
import keras
from keras import layers


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    smooth = 1. # ゼロ除算回避のための定数
    y_true_flat = tf.reshape(y_true, [-1]) # 1次元に変換
    y_pred_flat = tf.reshape(y_pred, [-1]) # 同様

    y_true_flat =tf.cast(tf.reshape(y_true, [-1]), tf.float32)  # 1次元に変換
    y_pred_flat = tf.cast(tf.reshape(y_pred, [-1]) , tf.float32) # 同様

    tp = tf.reduce_sum(y_true_flat * y_pred_flat) # True Positive
    nominator = 2 * tp + smooth # 分子
    denominator = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth # 分母
    score = nominator / denominator
    return 1. - score

def tversky_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    alpha = 0.3 # FP、FNの重み
    smooth = 1.0 # ゼロ除算回避のための定数

    y_true_flat =tf.cast(tf.reshape(y_true, [-1]), tf.float32)  # 1次元に変換
    y_pred_flat = tf.cast(tf.reshape(y_pred, [-1]) , tf.float32) # 同様

    tp = tf.reduce_sum(y_true_flat * y_pred_flat) # True Positive
    fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat) # False Positive
    fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat)) # False Negative

    score = (tp + smooth)/(tp + alpha * fp + (1-alpha) * fn + smooth) # Tversky
    return 1. - score

def create_conv(input, filters, l2_reg, name):
    x = Conv2D(filters=filters,
               kernel_size=3,               # 論文の指定通り
               activation='relu',           # 論文の指定通り
               padding='same',              # sameにすることでConcatする際にContracting側の出力のCropが不要になる
               kernel_regularizer=regularizers.l2(l2_reg),
               name=name)(input)
    x = BatchNormalization()(x)
    return x

def create_trans(input, filters, l2_reg, name):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=2,      # 論文の指定通り
                        strides=2,          # このストライドにより出力サイズが入力の2倍に拡大されている
                        activation='relu',  # 論文の指定通り
                        padding='same',     # Concat時のCrop処理回避のため
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name=name)(input)
    x = BatchNormalization()(x)
    return x

def convolution_block(block_input,num_filters=256,kernel_size=3,dilation_rate=1,padding="same",use_bias=False):
    x = layers.Conv2D(num_filters,kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", use_bias=use_bias)(block_input)
    x = layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25, noise_shape=None, seed=None)(x)
    return tf.nn.relu(x)

def Interp(x, shape):
    """ interpolation """
    from keras.backend import tf as ktf

    new_height, new_width = shape

    resized = tf.keras.layers.Resizing(int(new_height), int(new_width),interpolation='bilinear',crop_to_aspect_ratio=False)(x)

    return resized



def interp_block(
    x, num_filters=512, level=1, input_shape=(512, 512, 3), output_stride=16
):
    """ interpolation block """
    feature_map_shape = (input_shape[0] / output_stride, input_shape[1] / output_stride)

    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    if output_stride == 16:
        scale = 5

    kernel = (level * scale, level * scale)
    strides = (level * scale, level * scale)

    global_feat = AveragePooling2D(
        kernel, strides=strides, name="pool_level_%s_%s" % (level, output_stride)
    )(x)

    global_feat = convolution_block(global_feat, kernel_size=1)

    global_feat = keras.layers.Lambda(Interp, arguments={"shape": feature_map_shape})(global_feat)

    return global_feat

def Aux(x):
    x = convolution_block(x,num_filters=256,kernel_size=3)
    x = convolution_block(x,num_filters=3,kernel_size=1)
    resize_x = tf.keras.layers.Resizing(512,512,interpolation='bilinear',crop_to_aspect_ratio=False)(x)
    return resize_x

def PyramidPookingModule(input, num_filters=512, input_shape=(512, 512, 3), output_stride=16, levels=[6, 3, 2, 1]
):
    """ pyramid pooling function """

    pyramid_pooling_blocks = [input]

    for level in levels:
        pyramid_pooling_blocks.append(
            interp_block(
                input,
                num_filters=num_filters,
                level=level,
                input_shape=input_shape,
                output_stride=output_stride,
            )
        )

    x = concatenate(pyramid_pooling_blocks)
    x = convolution_block(x,num_filters=num_filters)

    return x

def PSP_decoder(x):
    x = convolution_block(x,num_filters=256)
    # x = convolution_block(x,num_filters = 3,kernel_size=1)
    resize_x = tf.keras.layers.Resizing(512,512,interpolation='bilinear',crop_to_aspect_ratio=False)(x)
    x = layers.Conv2D(3, kernel_size=(1, 1), padding="same")(resize_x )

    return x

def PSPNet():
    num_filters=512
    levels=[6, 3, 2, 1]
    input_shape=(512,512,3)
    input = Input(input_shape)
    out_stride=16
    num_classes=3

    resnet = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=input)

    x = resnet.get_layer("conv4_block6_2_relu").output

    aux_output = Aux(x)

    x = PyramidPookingModule(x)


    model_output = PSP_decoder(x)
    model = keras.Model(inputs=input, outputs=[model_output,aux_output])
    
    return model

loss_object = tf.keras.losses.CategoricalCrossentropy()

def softmax_cross_entropy_loss(y_true, y_pred):
    loss = loss_object(y_true, y_pred)
    return loss

class Custum_class(tf.keras.Model):
    def __init__(self, model, **kwargs):
        super(Custum_class, self).__init__(**kwargs)
        self.model = model

    def compile(self, optimizer, metrics, model_loss_fn, tversky_loss_fn,dice_loss_fn, alpha=0.1, temperature=3, **kwargs):
        super(Custum_class, self).compile(optimizer=optimizer, metrics=metrics, **kwargs)
        self.model_loss = model_loss_fn
        self.tversky_loss = tversky_loss_fn
        self.dice_loss = dice_loss_fn
        self.alpha = alpha
        #self.alpha = alpha
        #self.temperature = temperature

    def train_step(self, data):
        x, y, sample_weight  = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            # 生徒モデルで推定
            cnn_output,aux_output = self.model(x,training=True)#, training=True)

            # lossを算出
            model_loss = self.model_loss(y, cnn_output )
            aux_loss = self.model_loss(y, aux_output )
            tversky_loss = self.tversky_loss(y, cnn_output )
            dice_loss = self.dice_loss(y, cnn_output )

            loss = self.alpha * model_loss + tversky_loss + dice_loss + aux_output

        # 勾配を算出
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # optimizerでweightを更新
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # compiled_metricsを更新
        y = tf.cast(y, tf.float32)
        self.compiled_metrics.update_state(y,cnn_output)

        # metricsを算出して返す
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"model_loss": model_loss, "tversky_loss": tversky_loss}
        )
        return results

