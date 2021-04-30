#!/usr/bin/python3

import pdb
import numpy as np


def loadData(outfile):
    '''
    train data shape: (3800, 3200, 2)
    train label shape: (3800, 19)
    test data shape: (1900, 3200, 2)
    test label shape: (1900, 19)
    so the data is stored in real value format, we are ready
    to input these data into a real value network
    '''
    np_dict = np.load(outfile)
    dict_wifi = {}

    x_train = np_dict['arr_0.npy']
    y_train = np_dict['arr_1.npy']
    x_val = np_dict['arr_2.npy']
    y_val = np_dict['arr_3.npy']

    dict_wifi['x_test'] = np_dict['arr_4.npy']
    dict_wifi['y_test'] = np_dict['arr_5.npy']
    dict_wifi['fc_train'] = np_dict['arr_6.npy']
    dict_wifi['fc_validation'] = np_dict['arr_7.npy']
    dict_wifi['fc_test'] = np_dict['arr_8.npy']
    num_classes = dict_wifi['y_test'].shape[1]

    # do not know what fc data is
    # here we need to merge train and val data
    train_data = np.concatenate((x_train, x_val), axis=0)
    train_label = np.concatenate((y_train, y_val), axis=0)

    dict_wifi['x_train'] = train_data
    dict_wifi['y_train'] = train_label

    return dict_wifi, num_classes
