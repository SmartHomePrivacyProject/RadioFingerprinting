#!/usr/bin/python3
'''
Nov 6th: updated for D2 option, which can easily change between 1D and 2D with out change the code
'''
import os
import sys
import argparse
import pdb

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split

import radioConv
import config

ROOT_DIR = os.getenv('ROOT_DIR')
resDir = os.path.join(ROOT_DIR, 'resDir')
modelDir = os.path.join(resDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


def loadData(dpath, data_dim=600, expandDim=True):
    wholePack = np.load(dpath)
    allData, allLabel = wholePack['x'], wholePack['y']

    # limit data dim
    allData = allData[:, :data_dim]

    X_train, X_test, y_train, y_test = train_test_split(allData, allLabel, test_size=0.2, shuffle=True, random_state=47)
    NUM_CLASS = len(set(list(y_test)))
    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)

    if expandDim:
        X_train = X_train[:, np.newaxis, :]
        X_test = X_test[:, np.newaxis, :]

    # delete all no use data
    del wholePack
    del allData
    del allLabel

    return X_train, y_train, X_test, y_test, NUM_CLASS


def main(opts):
    # setup params
    Batch_Size = 128
    Epoch_Num = 100
    saveModelPath = os.path.join(modelDir, 'best_model_{}.h5'.format(opts.modelType))
    checkpointer = ModelCheckpoint(filepath=saveModelPath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
    callBackList = [checkpointer, earlyStopper]

    print('loading data...')
    dpath = opts.input
    data_dim = 600
    inp_shape = (1, data_dim)
    train_x, train_y, test_x, test_y, NUM_CLASS = loadData(dpath, data_dim=data_dim)

    print('get the model and compile it...')
    model = radioConv.create_model(opts.modelType, inp_shape, NUM_CLASS, D2=False, channel='first')
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

    print('fit the model with data...')
    model.fit(x=train_x, y=train_y,
              batch_size=Batch_Size,
              epochs=Epoch_Num,
              verbose=opts.verbose,
              callbacks=callBackList,
              validation_split=0.1,
              shuffle=True)

    print('test the trained model...')
    score, acc = model.evaluate(test_x, test_y, batch_size=Batch_Size, verbose=0)
    print('test acc is: ', acc)

    print('all test done!')


if __name__ == "__main__":
    opts = config.parse_args(sys.argv)
    main(opts)
