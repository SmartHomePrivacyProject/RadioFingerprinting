#! /usr/bin/python3
'''
Nov 6h, updated the API, pass a D2 value to make it adapt to 1D or 2D Conv model, all test done!
'''
import os
import sys
import argparse
import pdb
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import EarlyStopping, Callback
import keras.backend as K
from keras.utils import np_utils

import numpy as np
import pandas as pd

#import config

#ROOT_DIR = os.getenv('PROJECT_DIR')
#print(ROOT_DIR)
#resDir = os.path.join(ROOT_DIR, 'resDir')
#modelDir = os.path.join(resDir, 'modelDir')
#os.makedirs(modelDir, exist_ok=True)


def createHomegrown(inp_shape, emb_size, data_format, D2):
    # -----------------Entry flow -----------------
    if D2:
        from keras.layers import MaxPooling2D as MaxPooling
        from keras.layers import GlobalAveragePooling2D as GlobalAveragePooling
        from keras.layers import Conv2D as Conv
    else:
        from keras.layers import MaxPooling1D as MaxPooling
        from keras.layers import GlobalAveragePooling1D as GlobalAveragePooling
        from keras.layers import Conv1D as Conv

    input_data = Input(shape=inp_shape)
    if D2:
        kernel_size = ['None', (1, 7), (2, 5)]
    else:
        kernel_size = ['None', 7, 7]
    filter_num = ['None', 50, 50]
    conv_stride_size = ['None', 1, 1]
    pool_stride_size = ['None', 1, 1]
    activation_func = ['None', 'relu', 'relu']
    #activation_func = ['None', 'elu', 'elu']
    dense_layer_size = ['None', 256, 80]

    model = Conv(filters=filter_num[1], kernel_size=kernel_size[1],
                 strides=conv_stride_size[1], padding='same', name='block1_conv1', data_format=data_format)(input_data)
    model = Activation(activation_func[1], name='block1_act1')(model)
    model = Dropout(0.5, name='block1_dropout')(model)

    model = Conv(filters=filter_num[2], kernel_size=kernel_size[2],
                 strides=conv_stride_size[2], padding='same', name='block2_conv1', data_format=data_format)(model)
    model = Activation(activation_func[2], name='block2_act1')(model)
    model = Dropout(0.5, name='block2_dropout')(model)

    output = GlobalAveragePooling()(model)

    dense_layer = Dense(dense_layer_size[1], name='dense1', activation='relu')(output)
    dense_layer = Dense(dense_layer_size[2], name='dense2', activation='relu')(dense_layer)
    dense_layer = Dense(emb_size, name='dense3', activation='softmax')(dense_layer)

    shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    return shared_conv2


def baselineBlock(input, block_idx, D2):
    if D2:
        from keras.layers import MaxPooling2D as MaxPooling
        from keras.layers import Conv2D as Conv
    else:
        import resnet50_1D as resnet50
        from keras.layers import MaxPooling1D as MaxPooling
        from keras.layers import Conv1D as Conv

    if D2:
        kernel_size = ['None', (1, 7), (2, 5)]
    else:
        kernel_size = ['None', 7, 5]
    filter_num = ['None', 128, 128]
    conv_stride = ['None', 1, 1]
    pool_size = ['None', 2]
    pool_stride = ['None', 1]
    act_func = 'relu'

    model = Conv(filters=filter_num[1], kernel_size=kernel_size[1], name='conv1_{}'.format(block_idx),
                 strides=conv_stride[1], padding='same', activation=act_func)(input)
    model = Conv(filters=filter_num[2], kernel_size=kernel_size[2], name='conv2_{}'.format(block_idx),
                 strides=conv_stride[2], padding='same', activation=act_func)(model)
    output = MaxPooling(pool_size=pool_size[1], strides=pool_stride[1], padding='same',
                        name='pool_{}'.format(block_idx))(model)

    return output


def createBaseline(inp_shape, emb_size, data_format, D2):
    if D2:
        from keras.layers import GlobalAveragePooling2D as GlobalAveragePooling
    else:
        from keras.layers import GlobalAveragePooling1D as GlobalAveragePooling

    dense_layer_size = ['None', 256, 256, 128]
    act_func = ['None', 'relu', 'relu', 'relu']

    blockNum = 4
    input_data = Input(shape=inp_shape)
    for i in range(blockNum):
        idx = i + 1
        if 0 == i:
            model = baselineBlock(input_data, idx, D2)
        else:
            model = baselineBlock(model, idx, D2)

    middle = GlobalAveragePooling()(model)

    dense_layer = Dense(dense_layer_size[1], name='dense1', activation=act_func[1])(middle)
    dense_layer = Dense(dense_layer_size[2], name='dense2', activation=act_func[2])(dense_layer)
    dense_layer = Dense(dense_layer_size[3], name='dense3', activation=act_func[3])(dense_layer)
    dense_layer = Dense(emb_size, name='dense4', activation='softmax')(dense_layer)

    conv_model = Model(inputs=input_data, outputs=dense_layer)
    return conv_model


def createResnet(inp_shape, emb_size, data_format, D2):
    if D2:
        import resnet50_2D as resnet50
    else:
        import resnet50_1D as resnet50
    return resnet50.create_model(inp_shape, emb_size)


def create_model(modelType, inp_shape, NUM_CLASS, D2=False, channel='last'):

    if 'first' == channel:
        data_format = 'channels_first'
    elif 'last' == channel:
        data_format = 'channels_last'
    else:
        raise

    emb_size = NUM_CLASS
    if 'homegrown' == modelType:
        model = createHomegrown(inp_shape, emb_size, data_format, D2)
    elif 'baseline' == modelType:
        model = createBaseline(inp_shape, emb_size, data_format, D2)
    elif 'resnet' == modelType:
        model = createResnet(inp_shape, emb_size, data_format, D2)
    else:
        raise ValueError('model type {} not support yet'.format(modelType))

    return model


def test_run(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def test(D2):
    modelTypes = ['homegrown', 'baseline', 'resnet']
    NUM_CLASS = 10
    signal = True
    inp_shape = (1, 2, 288) if D2 else (2, 288)

    for modelType in modelTypes:
        model = create_model(modelType, inp_shape, NUM_CLASS, D2=D2, channel='first')
        try:
            flag = test_run(model)
        except Exception as e:
            print(e)

    print('all done!') if signal else print('test failed')


if __name__ == "__main__":
    D2_Type = [True, False]
    for D2 in D2_Type:
        test(D2)
