from __future__ import division
from keras.layers import Lambda, merge
from keras.layers.convolutional import Convolution2D, AtrousConvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
import keras.backend as K
import numpy as np
from drcn import DRCN
from gaussian_prior import LearningPrior
from attentive_convlstm import AttentiveConvLSTM
from config import *


def conv_block(input_tensor, kernel_size, filters, atrous_rate=(1, 1)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1

    x = Convolution2D(nb_filter1, 1, 1)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = AtrousConvolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same', atrous_rate=atrous_rate)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1)(x)
    x = BatchNormalization(axis=bn_axis)(x)

    shortcut = Convolution2D(nb_filter3, 1, 1)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis)(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def repeat(x):
    return K.reshape(K.repeat(K.batch_flatten(x), T), (b_s, T, 512, shape_r_gt, shape_c_gt))


def repeat_shape(s):
    return (s[0], T) + s[1:]


def priors_init(shape, name=None):
    means = np.random.uniform(low=0.3, high=0.7, size=shape[0] // 2)
    covars = np.random.uniform(low=0.05, high=0.3, size=shape[0] // 2)
    return K.variable(np.concatenate((means, covars), axis=0), name=name)


def kl_divergence(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), shape_r_gt, axis=-1)), shape_c_gt, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)), shape_r_gt, axis=-1)), shape_c_gt, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)), shape_r_gt, axis=-1)), shape_c_gt, axis=-1)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    return K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=-1)


def schedule(epoch):
    lr = [1e-4, 1e-4, 1e-5, 1e-5, 1e-6,
          1e-6, 1e-7, 1e-7, 1e-8, 1e-8]
    return lr[epoch]


def sam(x):
    resnet = DRCN(input_tensor=x[0])

    conv_feat = conv_block(input_tensor=resnet.output, kernel_size=3, filters=[512, 512, 512])

    convlstm = Lambda(repeat, repeat_shape)(conv_feat)
    convlstm = AttentiveConvLSTM(nb_filters_in=512, nb_filters_out=512, nb_filters_att=512, nb_cols=3, nb_rows=3)(convlstm)

    priors = LearningPrior(nb_gaussian=nb_gaussian, init=priors_init)(x[1])
    concateneted = merge([convlstm, priors], mode='concat', concat_axis=1)
    conv_priors = conv_block(input_tensor=concateneted, kernel_size=5, filters=[512, 512, 512], atrous_rate=(4, 4))

    priors = LearningPrior(nb_gaussian=nb_gaussian, init=priors_init)(x[1])
    concateneted = merge([conv_priors, priors], mode='concat', concat_axis=1)
    conv_priors = conv_block(input_tensor=concateneted, kernel_size=5, filters=[512, 512, 512], atrous_rate=(4, 4))

    return Convolution2D(1, 1, 1, border_mode='same', activation='relu')(conv_priors)