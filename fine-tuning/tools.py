#!/usr/bin/env python3.6

import os
import sys
import subprocess
import tempfile
import random
import time
from collections import defaultdict
import pdb

import numpy as np
import logging


TMP_DIR = '/tmp/labtest'
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


def getVriusTotalHashKey():
    key = 'fd2d772ac104caac9b92cee4d45d9043144bf18eb4bb6df5c97b4a764345ab89'
    return key


def get_date():
    return time.strftime("%Y_%m_%d", time.localtime())


def get_time():
    return time.strftime("time_%H_%M_%S", time.localtime())


def get_pid():
    return str(os.getpid())


def makeTempFile(suffix='', prefix='', dir=None, keepfile=False):
    if not prefix:
        prefix = 'tmp{}_'.format(get_pid())
    if not dir:
        dir = TMP_DIR
    dir = os.path.join(dir, get_date())
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fd, fname = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    if not keepfile:
        os.close(fd)
    return fname


def makeTempDir(suffix='', prefix='', dir=None):
    if not prefix:
        prefix = 'tmp{}_'.format(get_pid())
    if not dir:
        dir = TMP_DIR
    dir = os.path.join(dir, get_date())
    if not os.path.isdir(dir):
        os.makedirs(dir)
    dname = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)

    return dname


def getLogger(appName='default is empty'):
    logger = logging.getLogger(appName)
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formater)
    logger.addHandler(ch)
    return logger


def getSectionList(start, end, interval):
    rangeList = [start]
    while 1:
        tmpPoint = start + interval
        if tmpPoint >= end:
            break
        rangeList.append(tmpPoint)
        start = tmpPoint
    rangeList[-1] = end
    secList = [0 for n in range(len(rangeList)-1)]
    return rangeList, secList


def computeRange(rangeList, feature):
    l = len(rangeList) - 1
    for i in range(l):
        x1 = rangeList[i]
        x2 = rangeList[i+1]
        if x1 <= feature < x2:
            return i

    raise ValueError('the value of feature == {} exceed the rangeList'.format(feature))


def shuffleData(X, y):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    assert(X.shape[0] == y.shape[0])

    # pair up
    tupList = []
    for i in range(y.shape[0]):
        if 2 == len(X.shape):
            tmp_tuple = (X[i, :], y[i])
        elif 1 == len(X.shape):
            tmp_tuple = (X[i], y[i])
        elif 3 == len(X.shape):
            tmp_tuple = (X[i, :, :], y[i])
        else:
            raise ValueError('data shape {} not supported yet'.format(X.shape))
        tupList.append(tmp_tuple)

    random.shuffle(tupList)
    X, y = [], []
    for i in range(len(tupList)):
        X.append(tupList[i][0])
        y.append(tupList[i][1])

    X = np.array(X)
    y = np.array(y)
    return X, y


def datadict2data(datadict, keys=[], shuffle=True):
    allData, allLabel = [], []

    if not keys:
        keys = list(datadict.keys())

    for key in keys:
        oneCls = datadict[key]
        oneLabel = np.ones(len(oneCls)) * int(float(key))
        allData.extend(oneCls)
        allLabel.extend(oneLabel)

    if shuffle:
        allData, allLabel = shuffleData(allData, allLabel)

    return allData, allLabel


def data2datadict(allData, allLabel, clsLimit=0, sampleLimit=0):
    '''
    expected input are numpy ndarry
    '''
    if not isinstance(allData, np.ndarray):
        allData = np.array(allData)
    if not isinstance(allLabel, np.ndarray):
        allLabel = np.array(allLabel)

    datadict = defaultdict(list)

    allCls = list(set(allLabel))
    if clsLimit:
        allCls = random.sample(allCls, clsLimit)

    for i in range(len(allLabel)):
        label = allLabel[i]
        if label in allCls:
            if len(allData.shape) == 2:
                sample = allData[i, :]
            elif len(allData.shape) == 1:
                sample = allData[i]
            else:
                raise ValueError('data shape {} not supported yet'.format(allData.shape))
            datadict[label].append(sample)

    count = 0
    new_dict = defaultdict(list)
    for key in datadict.keys():
        oneClsData = datadict[key]
        new_dict[count] = oneClsData
        count += 1

    del datadict

    if sampleLimit:
        for key in new_dict.keys():
            oneClsData = new_dict[key]
            if sampleLimit >= len(oneClsData):
                new_samp = oneClsData[:sampleLimit]
            else:
                new_samp = random.sample(oneClsData, sampleLimit)
            new_dict[key] = new_samp

    return new_dict


def limitData(allData, allLabel, clsLimit=0, sampleLimit=0):
    dataDict = data2datadict(allData, allLabel, clsLimit, sampleLimit)
    x_new, y_new = datadict2data(dataDict)
    return x_new, y_new


def divideData(allData, allLabel, train_sample_num=5, train_pool_size=20):
    data_dict = data2datadict(allData, allLabel)
    train_data, train_label, test_data, test_label = [], [], [], []
    oneClsNum = len(list(data_dict[0]))
    test_sample_num = oneClsNum - train_pool_size

    for key in data_dict.keys():
        oneCls = list(data_dict[key])
        random.shuffle(oneCls)
        train_pool = []
        for i in range(train_pool_size):
            tmp = oneCls.pop()
            train_pool.append(tmp)
        train_data.extend(train_pool[:train_sample_num])
        tmpLabels = np.ones(train_sample_num, dtype=np.int) * key
        train_label.extend(tmpLabels)

        test_data.extend(oneCls[:test_sample_num])
        tmpLabels = np.ones(test_sample_num, dtype=np.int) * key
        test_label.extend(tmpLabels)

    train_data, train_label = shuffleData(train_data, train_label)
    test_data, test_label = shuffleData(test_data, test_label)
    return train_data, train_label, test_data, test_label


def divideDataDict(data_dict, train_sample_num=5, train_pool_size=20):
    train_data, train_label, test_data, test_label = [], [], [], []
    keys = list(data_dict.keys())
    oneClsNum = len(list(data_dict[keys[0]]))
    test_sample_num = oneClsNum - train_pool_size

    for key in data_dict.keys():
        oneCls = list(data_dict[key])
        random.shuffle(oneCls)
        train_pool = []
        for i in range(train_pool_size):
            tmp = oneCls.pop()
            train_pool.append(tmp)
        train_data.extend(train_pool[:train_sample_num])
        tmpLabels = np.ones(train_sample_num, dtype=np.int) * key
        train_label.extend(tmpLabels)

        tmpData = oneCls[:test_sample_num]
        tmpLabels = np.ones(len(tmpData), dtype=np.int) * key
        test_data.extend(tmpData)
        test_label.extend(tmpLabels)

    train_data, train_label = shuffleData(train_data, train_label)
    test_data, test_label = shuffleData(test_data, test_label)
    return train_data, train_label, test_data, test_label


def highLighPrint(msg):
    print('\033[40;33m {} \033[0m'.format(msg))


if __name__ == "__main__":
    aaa = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    bbb = np.array([1, 2, 3, 4])
    xxx, yyy = shuffleData(aaa, bbb)
    print(xxx)
    print(yyy)
