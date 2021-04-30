#! /usr/bin python3.6

import os
import sys
import pdb

import numpy as np
import config


def generateIndex(allDataSize):
    np.random.seed(42)
    shuffledind = np.random.permutation(allDataSize)
    return shuffledind


def getSplitIndex(allDataSize, splitRatio):
    shuffledind = generateIndex(allDataSize)

    train_set_size = int(allDataSize * splitRatio['train'])
    val_set_size = int(allDataSize * splitRatio['val'])
    # test_set_size = int(allDataSize * splitRatio['test'])

    start, end = 0, train_set_size
    train_ind = shuffledind[start: end]

    start, end = train_set_size, train_set_size + val_set_size
    val_ind = shuffledind[start: end]

    '''
    start, end = train_set_size + val_set_size, train_set_size + val_set_size + test_set_size
    test_ind = shuffledind[start: end]
    '''

    #return train_ind, val_ind, test_ind
    return train_ind, val_ind


def splitData(splitRatio, allData, allLabel, splitType='random'):
    if not isinstance(allData, np.ndarray):
        allData = np.array(allData)
        allLabel = np.array(allLabel)

    if splitType == 'random':
        allDataSize = len(allLabel)
        #train_ind, val_ind, test_ind = getSplitIndex(allDataSize, splitRatio)
        train_ind, val_ind = getSplitIndex(allDataSize, splitRatio)

        trainData, trainLabels = allData[train_ind, :, :], allLabel[train_ind]
        valData, valLabels = allData[val_ind, :, :], allLabel[val_ind]
        #testData, testLabels = allData[test_ind, :, :], allLabel[test_ind]

    elif splitType == 'order':
        pass
    else:
        raise

    #return trainData, trainLabels, valData, valLabels, testData, testLabels
    return trainData, trainLabels, valData, valLabels


if __name__ == "__main__":
    opts = config.parse_args(sys.argv)
    print('all test passed!')
