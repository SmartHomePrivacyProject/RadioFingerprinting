#! /usr/bin/env python3

import os
import sys
import argparse
import pdb
import random
import time
import copy

import numpy as np
from collections import defaultdict

import tools as mytools


def sampleWithin(oneSample, sampleList):
    for item in sampleList:
        if (oneSample==item).all():
            return True

    return False


def removeRedundancy(samplesNPList):
    afterList = []
    afterList.append(samplesNPList[0])
    for i in range(1, len(samplesNPList)):
        oneSample = samplesNPList[i]
        if not sampleWithin(oneSample, afterList):
            afterList.append(oneSample)

    return afterList


def getExchangeParams(seedNum, data_dim, start=0.1, end=0.7):
    random.seed(seedNum)
    start_point = int(random.random() * data_dim)
    exchange_len = int(random.uniform(start, end) * data_dim)
    return start_point, exchange_len


def generateNewSamples(sample_pair, start_point, exchange_len):
    if not isinstance(sample_pair[0], np.ndarray):
        sample_pair[0] = np.array(sample_pair[0])
    if not isinstance(sample_pair[1], np.ndarray):
        sample_pair[1] = np.array(sample_pair[1])

    # mind copy here
    cut_1 = copy.deepcopy(sample_pair[0][start_point: start_point+exchange_len])
    cut_2 = copy.deepcopy(sample_pair[1][start_point: start_point+exchange_len])

    sample_pair[0][start_point: start_point+exchange_len] = cut_2
    sample_pair[1][start_point: start_point+exchange_len] = cut_1

    return sample_pair


def getNewSamples(samples, newSamNum, data_dim):
    new_samples = []

    samNum = 0
    while 1:
        if samNum >= newSamNum:
            break
        sample_choose = random.sample(samples, 2)
        sample_pair = copy.deepcopy(sample_choose)
        #seed = int(time.time()) % int(13700 * random.random())
        seed = int(time.time()) % int(1000 * (random.random()+1))
        start_point, exchange_len = getExchangeParams(seed, data_dim)
        tmpSamples = generateNewSamples(sample_pair, start_point, exchange_len)
        new_samples.extend(tmpSamples)
        new_samples = removeRedundancy(new_samples)
        samNum = len(new_samples)

    new_samples = new_samples[:newSamNum]
    return new_samples


def data_aug(x, y, newSamNum):
    '''
    augument data by exchange random part of samples within class
    '''
    # form old data dict
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    data_dim = x.shape[1]
    totalNum = x.shape[0]

    data_dict = defaultdict(list)
    for i in range(totalNum):
        label = int(y[i])
        sample = x[i, :]
        data_dict[label].append(sample)

    # generate new data dict
    newDataDict = defaultdict(list)
    for key in data_dict.keys():
        samples = data_dict[key]
        new_samples = getNewSamples(samples, newSamNum, data_dim)
        newDataDict[key] = new_samples

    # convert data dict to data array
    allData, allLabel = [], []
    tmpData, tmpLabel = mytools.datadict2data(data_dict)
    allData.extend(tmpData)
    allLabel.extend(tmpLabel)

    tmpData, tmpLabel = mytools.datadict2data(newDataDict)
    allData.extend(tmpData)
    allLabel.extend(tmpLabel)

    # the shuffleData method will convert the data into numpy ndarry
    allData, allLabel = mytools.shuffleData(allData, allLabel)

    return allData, allLabel


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-n', '--newSamNum', type=int, default=50, help='')
    parser.add_argument('-l', '--limit', type=int, default=5,
                        help='limit the input sample number of input data')
    parser.add_argument('-t', '--test', action='store_true', help='')
    opts = parser.parse_args()
    return opts


def main(opts):
    '''
    standalone run, and save the data to output path
    '''
    dataName = os.path.basename(opts.input)
    wholePack = np.load(opts.input)
    x, y = wholePack['x'], wholePack['y']

    # limit data first and then augment data
    x, y = mytools.limitData(x, y, sampleLimit=opts.limit)
    allData, allLabel = data_aug(x, y, opts.newSamNum)

    savePath = os.path.join(opts.output, '{}_PartExchange_to_{}.npz'.format(dataName, opts.newSamNum))
    np.savez(savePath, x=allData, y=allLabel)
    print('data save to: {}'.format(savePath))


def test():
    print('test getNewSamples func...')
    aaa = np.array([3, 3, 3, 3, 3, 3, 3, 3])
    bbb = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    ccc = np.array([2, 2, 2, 2, 2, 2, 2, 2])
    ddd = np.array([4, 4, 4, 4, 4, 4, 4, 4])

    samplePair = [aaa, bbb]
    newSamples = getNewSamples(samplePair, 7, aaa.shape[0])
    for item in newSamples:
        print(item)

    print('test data aug func...')
    dataset = [aaa, bbb, ccc, ddd]
    label = [1, 1, 2, 2]
    newData, newLabel = data_aug(dataset, label, newSamNum=5)
    for i in range(len(newLabel)):
        print('label: ', newLabel[i], 'data: ', newData[i])


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    if opts.test:
        test()
    else:
        main(opts)
