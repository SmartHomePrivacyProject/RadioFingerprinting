#!/usr/bin/python3

import os
import sys
import argparse
import pdb

import numpy as np
from sklearn.model_selection import train_test_split

import config
import load_slice_IQ
import mytools.tools as mytools

'''
data need to organized into 3 parts:
    (train_x, train_y), (val_x, val_y), (test_x, test_y)
and all data need to be normalize to (0,1) and into a mat
with shape like (sample_num, sample_dim, 2), all complex
value are stored in (real, imag) pair
'''
#'\033[{i};1m这里写需要输出打印的内容\033[0m'# i是指打印颜色的编号
print()
#print('\033[40;33m required args includes input/output/D2/normalize/ \033[0m')
msg = 'required args includes input/output/D2/normalize/'
mytools.highLighPrint(msg)
print()


def loadData2Dict(opts):
    print('loading data...')
    # train_x, train_y, val_x, val_y, test_x, test_y = readSigmf.getData(opts, x_day_dir)
    # D2 means that make it into 2 dimension data
    dataOpts = load_slice_IQ.loadDataOpts(opts.input, opts.D2, num_slice=10000)
    train_x, train_y, test_x, test_y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, opts.channel_first)

    if opts.normalize:
        train_x = load_slice_IQ.normalizeData(train_x)
        test_x = load_slice_IQ.normalizeData(test_x)

    data_dict = {}
    data_dict['x_train'] = train_x
    data_dict['y_train'] = train_y
    data_dict['x_test'] = test_x
    data_dict['y_test'] = test_y

    print('x_train shape: {}\ty_train shape: {}\tx_test shape: {}\ty_test shape: {}'.format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))

    return data_dict


def save2file(outfile, data_dict):
    x_train, y_train = data_dict['x_train'], data_dict['y_train']
    x_test, y_test = data_dict['x_test'], data_dict['y_test']

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=47, shuffle=True)
    np.savez(outfile, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
    outfile = os.path.abspath(outfile)
    print('all data save to path {}'.format(outfile))


def main(opts):
    tmp_list = opts.input.split('/')
    tmp_list = filter(lambda a: a != '', tmp_list)
    save_name = list(tmp_list)[-1]
    save_path = os.path.join(opts.output, '{}.npz'.format(save_name))
    # load all the data
    data_dict = loadData2Dict(opts)

    # save data to npz file
    save2file(save_path, data_dict)

    print('all done!')


if __name__ == "__main__":
    opts = config.parse_args(sys.argv)
    main(opts)
