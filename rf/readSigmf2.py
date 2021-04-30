#! /usr/bin python3.6

import os
import sys
import pdb
import argparse
import random
import re
import math

from collections import defaultdict
import json
import numpy as np
import itertools
from sigmf import SigMFFile, sigmffile

import mytools.tools as mytools
import config


def createSignal(metafile, binfile):
    # Load a dataset
    with open(metafile, 'r') as f:
        metadata = json.loads(f.read())
    signal = SigMFFile(metadata=metadata['_metadata'], data_file=binfile)
    return signal


def get_one_sample(raw_data, start, length):
    if not isinstance(raw_data, np.ndarray):
        raw_data = np.array(raw_data)
    return raw_data[start: start+length]


def convert2IQdata(one_raw_data):
    flag = False
    rtnList = []
    for item in one_raw_data:
        realVal = np.real(item)
        imagVal = np.imag(item)
        if math.isnan(realVal):
            realVal = 0
            flag = True
        if math.isnan(imagVal):
            imagVal = 0
            flag = True
        tmp = [realVal, imagVal]
        rtnList.append(tmp)

    #if flag:
    #    pdb.set_trace()
    #tmpMat = np.array(rtnList).T
    #rtnList = list(tmpMat)
    return rtnList


def divideIntoChucks(raw_data, chuckNum):
    dataLen = len(raw_data)
    sliceLen = dataLen // chuckNum

    chuckList = []
    start = 0
    for i in range(chuckNum):
        end = start + sliceLen
        oneSlice = raw_data[start: end]
        start = end
        chuckList.append(oneSlice)

    return chuckList


def formInpData(raw_data, sample_length, selectedNum, params):
    dataOpt = params['dataOpt']
    if 2 == dataOpt:
        selectedIndex = params['selectedIndex']
    start_range = len(raw_data) - sample_length
    raw_samples = []
    for i in range(start_range):
        tmp_sample = get_one_sample(raw_data, i, sample_length)
        pdb.set_trace()
        raw_samples.append(tmp_sample)

    if 1 == dataOpt:
        selectedSamples = random.sample(raw_samples, selectedNum)
    elif 2 == dataOpt:
        raw_samples = np.array(raw_samples)
        selectedSamples = raw_samples[selectedIndex]
        selectedSamples = list(selectedSamples)
    elif 3 == dataOpt:
        selectedSamples = raw_samples[:selectedNum]
    else:
        raise

    rtn_samples = []
    for tmp_sample in selectedSamples:
        tmp_sample = convert2IQdata(tmp_sample)
        rtn_samples.append(tmp_sample)

    return rtn_samples


def generateIndex(allDataSize):
    np.random.seed(42)
    shuffledind = np.random.permutation(allDataSize)

    return shuffledind


def getSplitIndex(allDataSize, splitRatio):
    shuffledind = generateIndex(allDataSize)

    train_set_size = int(allDataSize * splitRatio['train'])
    val_set_size = int(allDataSize * splitRatio['val'])
    test_set_size = int(allDataSize * splitRatio['test'])

    start, end = 0, train_set_size
    train_ind = shuffledind[start: end]

    start, end = train_set_size, train_set_size + val_set_size
    val_ind = shuffledind[start: end]

    start, end = train_set_size + val_set_size, train_set_size + val_set_size + test_set_size
    test_ind = shuffledind[start: end]

    return train_ind, val_ind, test_ind


def splitData(opts, splitRatio, allData, allLabel):
    if not isinstance(allData, np.ndarray):
        allData = np.array(allData)
        allLabel = np.array(allLabel)

    if opts.splitType == 'random':
        allDataSize = len(allLabel)
        train_ind, val_ind, test_ind = getSplitIndex(allDataSize, splitRatio)
        trainData, trainLabels = allData[train_ind, :, :], allLabel[train_ind]
        valData, valLabels = allData[val_ind, :, :], allLabel[val_ind]
        testData, testLabels = allData[test_ind, :, :], allLabel[test_ind]
    elif opts.splitType == 'order':
        pass
    else:
        raise

    return trainData, trainLabels, valData, valLabels, testData, testLabels


def get_signal_samples(signal, label, params):
    # Get some metadata and all annotations
    chuckNum = params['chuckNum']
    sample_length = params['sample_length']
    selectedNum = params['selectedNum']

    raw_data = signal.read_samples(0, -1)
    chuckList = divideIntoChucks(raw_data, chuckNum)
    chuckList = chuckList[0]  # only take the first chuck

    if params['dataOpt'] == 2:
        totalNum = len(chuckList)
        allRanIndex = generateIndex(totalNum)
        selectedIndex = allRanIndex[:params['selectedNum']]
        params['selectedIndex'] = selectedIndex

    oneData = formInpData(chuckList, sample_length, selectedNum, params)
    oneLabel = np.ones(len(oneData), dtype=np.int) * label
    print('raw data length is: ', len(oneData))
    return oneData, oneLabel


def searchFp(fname, metaFileList):
    for mfp in metaFileList:
        m = re.search(fname, mfp)
        if m:
            return mfp
    return ''


def getSignalList(fpTuple):
    binFileList, metaFileList = fpTuple
    signalList = []
    for bfp in binFileList:
        fname = os.path.basename(bfp).split('.')[0]
        mfp = searchFp(fname, metaFileList)
        if not mfp:
            raise ValueError('binfile {} does not have a match'.format(bfp))
        signal = createSignal(mfp, bfp)
        signalList.append(signal)
    return signalList


def getOneDevData(fpTuple, label, params):
    print('processing file: ', fpTuple)
    signalList = getSignalList(fpTuple)
    allData, allLabel = [], []
    for signal in signalList:
        oneData, oneLabel = get_signal_samples(signal, label, params)
        allData.extend(oneData)
        allLabel.extend(oneLabel)

    return allData, allLabel


def getfpTuple(strLabel, x_day_dir):
    dayDevDir = os.path.join(x_day_dir, strLabel)
    fList = os.listdir(dayDevDir)
    binFileList, metaFileList = [], []
    for fname in fList:
        fp = os.path.join(dayDevDir, fname)
        if fp.endswith('bin'):
            binFileList.append(fp)
        elif fp.endswith('sigmf-meta'):
            metaFileList.append(fp)
        else:
            raise
    return (binFileList, metaFileList)


def generate_default_params():
    params = {
            'sample_length': 288,
            'selectedNum': 100000,
            'chuckNum': 10,
            'splitRatio': {'train': 0.7, 'val': 0.2, 'test': 0.1}
            }
    return params


def getData(opts, x_day_dir):
    '''this is made to read one day data'''
    params = generate_default_params()

    # dataOpt
    #   1: take diff random number across 5 devices
    #   2: take same random number across 5 devices
    #   3: take consecutive 100k slices
    params['dataOpt'] = 1

    devList = os.listdir(x_day_dir)
    label2Data = defaultdict()
    allData, allLabel = [], []
    for i in range(len(devList)):
        strLabel = devList[i]
        fpTuple = getfpTuple(strLabel, x_day_dir)
        label2Data[i] = fpTuple

        oneData, oneLabel = getOneDevData(fpTuple, i, params)
        allData.extend(oneData)
        allLabel.extend(oneLabel)

    splitRatio = params['splitRatio']
    trainData, trainLabels, valData, valLabels, testData, testLabels = splitData(opts, splitRatio, allData, allLabel)
    return trainData, trainLabels, valData, valLabels, testData, testLabels


def test_read_one_data(opts):
    allDataSize = 1000
    splitRatio = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    train_ind, val_ind, test_ind = getSplitIndex(allDataSize, splitRatio)

    # Load a dataset
    x_day_dir = opts.input
    trainData, trainLabels, valData, valLabels, testData, testLabels = getData(opts, x_day_dir)
    print(trainData.shape, valData.shape, testData.shape)
    print(trainLabels.shape)


if __name__ == "__main__":
    opts = config.parse_args(sys.argv)
    test_read_one_data(opts)
    print('all test passed!')
