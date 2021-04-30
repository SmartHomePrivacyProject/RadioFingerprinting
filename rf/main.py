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

import radioConv
#import readSigmf2 as readSigmf
import load_slice_IQ
import config
import get_simu_data
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = os.getenv('ROOT_DIR')
print(ROOT_DIR)
resDir = os.path.join(ROOT_DIR, 'resDir')
modelDir = os.path.join(resDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


def loadNeuData(opts):
    print('loading data...')
    x_day_dir = opts.input
    #train_x, train_y, val_x, val_y, test_x, test_y = readSigmf.getData(opts, x_day_dir)
    dataOpts = load_slice_IQ.loadDataOpts(opts.input, opts.location, opts.D2, num_slice=100000)
    train_x, train_y, test_x, test_y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, opts.channel_first)

    if opts.normalize:
        train_x = load_slice_IQ.normalizeData(train_x)
        test_x = load_slice_IQ.normalizeData(test_x)

    return train_x, train_y, test_x, test_y, NUM_CLASS


def loadSimuData(opts):
    simu_dict, NUM_CLASS = get_simu_data.loadData(opts.input)
    x_train, y_train = simu_dict['x_train'], simu_dict['y_train']
    x_test, y_test = simu_dict['x_test'], simu_dict['y_test']

    return x_train, y_train, x_test, y_test, NUM_CLASS


def main(opts):
    # load data
    if 'neu' == opts.dataSource:
        train_x, train_y, test_x, test_y, NUM_CLASS = loadNeuData(opts)
    elif 'simu' == opts.dataSource:
        train_x, train_y, test_x, test_y, NUM_CLASS = loadSimuData(opts)
    else:
        raise NotImplementedError()

    # setup params
    Batch_Size = 64
    Epoch_Num = 100
    saveModelPath = os.path.join(modelDir, 'best_model_{}_{}.h5'.format(opts.modelType,opts.location))
    checkpointer = ModelCheckpoint(filepath=saveModelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    earlyStopper = EarlyStopping(monitor='val_acc', mode='max', patience=10)
    callBackList = [checkpointer, earlyStopper]

    print('get the model and compile it...')
    if opts.D2:
        inp_shape = (1, train_x.shape[1], train_x.shape[2])
    else:
        inp_shape = (train_x.shape[1], train_x.shape[2])

    # pdb.set_trace()
    model = radioConv.create_model(opts.modelType, inp_shape, NUM_CLASS, D2=opts.D2, channel='first')
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
    results = [opts.modelType, opts.location, acc]

    with open('results.txt', 'a') as filehandle:
        #for listitem in results:
            #filehandle.write('%s' % listitem)
        filehandle.write('%f\n' % acc)

if __name__ == "__main__":
    opts = config.parse_args(sys.argv)
    main(opts)
