#! /usr/bin/env python3.6

import os
import sys
import argparse
import logging
import copy
import pdb
import time
import random
from statistics import mean, stdev

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras.utils import np_utils
from keras.models import load_model

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
from collections import defaultdict
import radioConv
# RootDir = os.getenv('ROOT_DIR')
# toolsDir = os.path.join(RootDir, 'tools')
# modelsDir = os.path.join(RootDir, 'models')
# sys.path.append(toolsDir)
# sys.path.append(modelsDir)
import load_slice_IQ
import augData
# import utility

import tools as mytools

thisFile = os.path.abspath(__file__)
currentDir = os.path.dirname(thisFile)
ResDir = os.path.join(currentDir, 'res_out')
modelDir = os.path.join(ResDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)

#thresholdList = 0.3 - 1 / np.logspace(0.55, 2, num=25, endpoint=True)
thresholdList = 0.4 - 1 / np.logspace(0.4, 2, num=25, endpoint=True)


class CNN():
    def __init__(self, opts):
        self.verbose = opts.verbose
        self.trainData = opts.trainData
        self.tuneData = opts.tuneData
        self.trainModelPath = os.path.join(modelDir, 'train_best_{}.h5'.format(opts.modelType))

        self.batch_size = 256
        self.trainEpochs = 50
        self.tuneEpochs = 10

        self.report = []

    def createModel(self, topK=False):
        input_shape, emb_size = self.input_shape, self.emb_size
        print("load well trained model")
        # model = DF(input_shape=input_shape, emb_size=emb_size, Classification=True)
        model = radioConv.create_model(opts.modelType, input_shape, emb_size, D2=opts.D2, channel='first')
        print('model compiling...')
        metricList = ['accuracy']
        if topK:
            metricList.append('top_k_categorical_accuracy')
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=metricList)
        return model

    def train(self, X_train, y_train, X_test, y_test, NUM_CLASS):
        '''train the cnn model'''
        model = self.createModel()

        print('Fitting model...')
        checkpointer = ModelCheckpoint(filepath=self.trainModelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        earlyStopper = EarlyStopping(monitor='val_acc', mode='max', patience=10)
        callBackList = [checkpointer, earlyStopper]

        start = time.time()
        hist = model.fit(X_train, y_train,
                         batch_size=self.batch_size,
                         epochs=self.trainEpochs,
                         validation_split=0.1,
                         verbose=self.verbose,
                         callbacks=callBackList)
        end = time.time()
        time_last = end - start
        print('Testing with best model...')
        score, acc = model.evaluate(X_test, y_test, batch_size=100)
        reportLine = 'Test accuracy with data {} is: {:f}\n'.format(self.trainData, acc)
        print(reportLine)
        return reportLine, time_last

    def tuneTheModel(self, X_train, y_train, NUM_CLASS):
        old_model = load_model(self.trainModelPath, compile=False)
        self.emb_size = NUM_CLASS
        new_model = self.createModel(topK=True)

        print("copying weights from old model to new model...")
        LayNum = len(new_model.layers) - 3
        for l1, l2 in zip(new_model.layers[:LayNum], old_model.layers[:LayNum]):
            l1.set_weights(l2.get_weights())
            l1.trainable = False

        print('Fitting model...')
        checkpointer = ModelCheckpoint(filepath=self.tuneModelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        earlyStopper = EarlyStopping(monitor='val_acc', mode='max', patience=10)
        callBackList = [checkpointer, earlyStopper]

        hist = new_model.fit(X_train, y_train,
                            batch_size=self.batch_size,
                            epochs=self.tuneEpochs,
                            validation_split=0.1,
                            verbose=0,
                            callbacks=callBackList)
        return new_model

    def test_close(self, new_model, X_test, y_test):
        print('Testing with best model...')
        score, acc, top5Acc = new_model.evaluate(X_test, y_test, batch_size=100)
        reportLine = 'Test accuracy of tune model with data {} is: {:f}, and test top 5 acc is: {:f}\n'.format(self.tuneData, acc, top5Acc)
        print(reportLine)
        return acc, top5Acc

    def tune(self, X_train, y_train, X_test, y_test, NUM_CLASS):
        new_model = self.tuneTheModel(X_train, y_train, NUM_CLASS)
        acc, top5Acc = self.test_close(new_model, X_test, y_test)
        return acc, top5Acc

    def loadData(self, opts, purpose, trsn, tesn, train_sample_num=25):
        if 'train' == purpose:
            train_sample_num = train_sample_num
            test_sample_num = 100
            opts.input = opts.trainData
        else:
            train_sample_num = trsn
            test_sample_num = tesn
            opts.input = opts.tuneData
            self.tuneModelPath = os.path.join(modelDir, 'tune_best_cnn_trsn_{}.h5'.format(trsn))

        print('loading data...')
        dataOpts = load_slice_IQ.loadDataOpts(opts.input, opts.location, opts.D2, num_slice=100000)
        train_x, train_y, test_x, test_y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, opts.channel_first)

        ############################################
        allData = np.concatenate((train_x, test_x), axis=0)
        allLabel = np.concatenate((train_y, test_y), axis=0)
        # splite the data
        dataDict = defaultdict(list)
        for i in range(len(allLabel)):
            oneLabel = allLabel[i]
            oneData = allData[i, :]
            dataDict[oneLabel].append(oneData)

        X_train, X_test, y_train, y_test = [], [], [], []
        NUM_CLASS = len(list(dataDict.keys()))
        for key in dataDict.keys():
            oneClsData = dataDict[key]
            random.shuffle(oneClsData)
            # split train and test
            train_samples = oneClsData[:train_sample_num]
            test_samples = oneClsData[train_sample_num:train_sample_num + test_sample_num]

            train_labels = np.ones(len(train_samples), dtype=np.int) * int(key)
            test_labels = np.ones(len(test_samples), dtype=np.int) * int(key)

            X_train.extend(train_samples)
            y_train.extend(train_labels)
            X_test.extend(test_samples)
            y_test.extend(test_labels)

        # shuffle data
        X_train, y_train = mytools.shuffleData(X_train, y_train)
        X_test, y_test = mytools.shuffleData(X_test, y_test)

        # limit data dim
        # X_train = X_train[:, :opts.data_dim]
        # X_test = X_test[:, :opts.data_dim]


        # delete all no use data

        del allData
        del allLabel


        if opts.D2:
            inp_shape = (1, X_train.shape[1], X_train.shape[2])
        else:
            inp_shape = (X_train.shape[1], X_train.shape[2])

        # if opts.normalize:
        #     train_x = load_slice_IQ.normalizeData(train_x)
        #     test_x = load_slice_IQ.normalizeData(test_x)

        self.input_shape = inp_shape
        self.emb_size = NUM_CLASS

        return X_train, y_train, X_test, y_test, NUM_CLASS


def prepareData(opts, model, trsn=40, test=10):
    X_train, y_train, X_test, y_test, NUM_CLASS = model.loadData(opts, purpose='tune', trsn=trsn, tesn=test)
    if opts.augData:
        print('aug data...')
        data_dim = X_train.shape[1]
        oldNum = X_train.shape[0]
        X_train, y_train = augData.data_aug(X_train, y_train, opts.augData)
        newNum = X_train.shape[0]
        print('aug data from {} to {}'.format(oldNum, newNum))

    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)
    return X_train, y_train, X_test, y_test, NUM_CLASS


def main(opts):
    model = CNN(opts)
    source = os.path.basename(opts.trainData).split('.')[0]
    target = os.path.basename(opts.tuneData).split('.')[0]
    test_times = 5

    flag = False if 'trainNum' == opts.testType else True
    if opts.trainData and flag and (not opts.modelPath):
        print('train the model once...')
        X_train, y_train, X_test, y_test, NUM_CLASS = model.loadData(opts, purpose='train')
        y_train = np_utils.to_categorical(y_train, NUM_CLASS)
        y_test = np_utils.to_categorical(y_test, NUM_CLASS)
        print('train data shape: ', X_train.shape)
        rtnLine, time_last = model.train(X_train, y_train, X_test, y_test, NUM_CLASS)
        print(rtnLine)
        del X_train, y_train, X_test, y_test, NUM_CLASS

    if opts.modelPath:
        model.trainModelPath = opts.modelPath

    if 'trainNum' == opts.testType:
        print('start run test: {}'.format(opts.testType))
        tsn = 20
        trainNum_list = [25, 50, 75, 100, 125]
        resultFile = os.path.join(ResDir, 'trainNumTest_tune_model_{}_to_{}.txt'.format(source, target))
        f = open(resultFile, 'a+')
        print('\n\n##################### test time is: {} ######################'.format(time.ctime()), file=f)
        for trNum in trainNum_list:
            # train phase
            print('train the model...')
            X_train, y_train, X_test, y_test, NUM_CLASS = model.loadData(opts, purpose='train')
            y_train = np_utils.to_categorical(y_train, NUM_CLASS)
            y_test = np_utils.to_categorical(y_test, NUM_CLASS)
            print('train data shape: ', X_train.shape)
            rtnLine, time_last = model.train(X_train, y_train, X_test, y_test, NUM_CLASS)
            print(rtnLine)
            del X_train, y_train, X_test, y_test, NUM_CLASS

            # test phase
            acc_list, acc_top5_list = [], []
            for i in range(test_times):
                X_train, y_train, X_test, y_test, NUM_CLASS = prepareData(opts, model, trsn=tsn)
                print('tune data shape: ', X_train.shape)
                print('tune the model...')
                acc, acctop5 = model.tune(X_train, y_train, X_test, y_test, NUM_CLASS)
                acc_list.append(acc)
                acc_top5_list.append(acctop5)
            rtnLine = 'Model type: {}, location: {}\n'.format(opts.modelType, opts.location)
            rtnLine = rtnLine + 'Test acc of data {} is: {:f}, stdev is: {:f}, \n'.format(model.tuneData, mean(acc_list), stdev(acc_list))
            rtnLine = rtnLine + 'test top 5 acc is: {:f}, stdev is: {:f}\n'.format(mean(acc_top5_list), stdev(acc_top5_list))
            rtnLine = rtnLine + 'training time last {}'.format(time_last)
            rtnLine = rtnLine + '\ttsn={}, \ttrain_sample_num={}'.format(tsn, trNum)
            print(rtnLine, file=f, flush=True)
        f.close()

    if 'tsn' == opts.testType:
        print('start run n_shot test...')
        trsnList = [320,640,1280]
        tesnList = [80,160,320]
        snList = zip(trsnList,tesnList)

        #tsnList = [5]
        resultFile = os.path.join(ResDir, 'tune_model_{}_to_{}.txt'.format(source, target))
        f = open(resultFile, 'a+')
        print('\n\n##################### test time is: {} ####################'.format(time.ctime()), flush=True, file=f)
        for sn in snList:
            # opts.nShot = trsn
            acc_list, acc_top5_list = [], []
            for i in range(test_times):
                X_train, y_train, X_test, y_test, NUM_CLASS = prepareData(opts, model, sn[0], sn[1])
                print('tune data shape: ', X_train.shape)
                print('tune the model...')
                acc, acctop5 = model.tune(X_train, y_train, X_test, y_test, NUM_CLASS)
                acc_list.append(acc)
                acc_top5_list.append(acctop5)
            rtnLine = 'Model type: {}, location: {}\n'.format(opts.modelType, opts.location)
            rtnLine = rtnLine + 'trsn={} test={} tune model with data {}, acc is: {:f}, and std is: {:f}\n'.format(sn[0], sn[1], model.tuneData, mean(acc_list), stdev(acc_list))
            rtnLine = rtnLine + 'Test top 5 acc is: {:f}, std is: {:f}\n\n'.format(mean(acc_top5_list), stdev(acc_top5_list))
            print(rtnLine, file=f, flush=True)
        f.close()

    elif 'aug' == opts.testType:
        print('start run test: {}'.format(opts.testType))
        augList = [10, 30, 50, 70, 90, 110]
        resultFile = os.path.join(ResDir, 'tune_model_{}_to_{}_with_aug.txt'.format(source, target))
        f = open(resultFile, 'a+')
        for aug in augList:
            opts.augData = aug
            acc_list, acc_top5_list = [], []
            for i in range(test_times):
                X_train, y_train, X_test, y_test, NUM_CLASS = prepareData(opts, model, trsn=10)
                # X_train, y_train, X_test, y_test, NUM_CLASS = prepareData(opts, model)
                print('tune the model...')
                acc, acctop5 = model.tune(X_train, y_train, X_test, y_test, NUM_CLASS)
                acc_list.append(acc)
                acc_top5_list.append(acctop5)
            rtnLine = 'Test accuracy of tune model with data {} is: {:f}, and test top 5 acc is: {:f}\n'.format(model.tuneData, mean(acc_list), mean(acc_top5_list))
            print(rtnLine, file=f)
            print('############################\n\n', file=f)
        f.close()
    elif 'trainTime' == opts.testType:
        print('training time is: ', time_last)
    else:
        pass


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--trainData', default='', help ='file path of config file')
    parser.add_argument('-tu', '--tuneData', help ='file path of config file')
    # parser.add_argument('-o', '--openData', help ='file path of open data file')
    parser.add_argument('-ns', '--nShot', type=int, help ='n shot number')
    parser.add_argument('-m', '--modelPath', default='', help ='file path of open data file')
    parser.add_argument('-d', '--data_dim', type=int, default=1500, help ='file path of config file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose or not')
    parser.add_argument('-a', '--augData', type=int, default=0, help='')
    parser.add_argument('-g', '--useGpu', action='store_true', help='')
    parser.add_argument('-tt', '--testType', default='tsn', help='choose different test: tsn/aug/trainNum/trainTime')
    parser.add_argument('-pf', '--prefix', help='')
    ######################## rf args ###############
    parser.add_argument('-i', '--input', help='input file/dir')
    parser.add_argument('-o', '--output', help='output dir')
    parser.add_argument('-mt', '--modelType', default='homegrown', help='choose from homegrown/baseline/resnet')
    parser.add_argument('-sp', '--splitType', default='random', help='choose from random/order')
    parser.add_argument('--D2', action='store_true', help='if set will return 2 dimension data')
    parser.add_argument('-n', '--normalize', action='store_true', help='')
    parser.add_argument('-ds', '--dataSource', help='choose from neu/simu')
    parser.add_argument('-cf', '--channel_first', action='store_true',
                        help='if set channel first otherwise channel last')
    parser.add_argument('-l', '--location', help='data collected location')

    opts = parser.parse_args()
    return opts

class testOpts():
    def __init__(self, trainData, tuneData, location, D2, testType, modelType ,modelPath, channel_first=True):
        self.tuneData = tuneData
        self.testType = testType
        self.modelPath = modelPath
        self.channel_first = channel_first
        self.location = location
        self.verbose = True
        self.trainData = trainData
        self.modelType = modelType
        self.D2 = D2

if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    # opts = testOpts(trainData= '/home/erc/PycharmProjects/fine-tuning/test_dataset',
    #                 tuneData='/home/erc/PycharmProjects/fine-tuning/test_dataset_2',
    #                 location='symbols',
    #                 D2=False,
    #                 testType='tsn',
    #                 modelType= 'symbols',
    #                 modelPath='/home/erc/PycharmProjects/fine-tuning/models/best_model_baseline_symbols.h5')
    # if opts.useGpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)
