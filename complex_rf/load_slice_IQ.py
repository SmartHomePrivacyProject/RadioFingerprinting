import os
import sys
import glob
import argparse
import pdb

import numpy as np
from keras.utils import np_utils
import utils


def normalizeData(v):
    # keepdims makes the result shape (1, 1, 3) instead of (3,). This doesn't matter here, but
    # would matter if you wanted to normalize over a different axis.
    v_min = v.min(axis=(0, 1), keepdims=True)
    v_max = v.max(axis=(0, 1), keepdims=True)
    v = (v - v_min)/(v_max - v_min)
    return v


def read_f32_bin(filename, start_ix, channel_first):
    with open(filename, 'rb') as bin_f:
        iq_seq = np.fromfile(bin_f, dtype='<f4')
        n_samples = iq_seq.shape[0] // 2

        IQ_data = np.zeros((2, n_samples))

        IQ_data[0, :] = iq_seq[range(0, iq_seq.shape[0]-1, 2)]
        IQ_data[1, :] = iq_seq[range(1, iq_seq.shape[0], 2)]

    del iq_seq
    rtn_data = IQ_data[:, start_ix:]
    if not channel_first:
        rtn_data = rtn_data.T
    return rtn_data


def dev_bin_dataset(glob_dat_path, n_samples, start_ix, channel_first, uniform):
    filelist = sorted(glob.glob(glob_dat_path))
    num_tran = len(filelist)
    axis = 1 if channel_first else 0
    all_IQ_data = []

    if uniform:
        samples_per_tran = n_samples // num_tran + 1
        for f in filelist:
            IQ_per_tran = read_f32_bin(f, start_ix, channel_first)
            if channel_first:
                IQ_per_tran = IQ_per_tran[:, 0:samples_per_tran]
            else:
                IQ_per_tran = IQ_per_tran[0:samples_per_tran, :]
            if len(all_IQ_data):
                all_IQ_data = np.concatenate((all_IQ_data, IQ_per_tran), axis=axis)
            else:
                all_IQ_data = IQ_per_tran
    else:
        all_IQ_data = read_f32_bin(filelist[0], start_ix, channel_first)
        isEnoughSample = False

        if all_IQ_data.shape[axis] < n_samples:
            for f in filelist[1:]:
                all_IQ_data = np.concatenate((all_IQ_data, read_f32_bin(f, start_ix, channel_first)), axis=axis)
                if all_IQ_data.shape[axis] >= n_samples:
                    isEnoughSample = True
                    break
        else:
            isEnoughSample = True

        if not isEnoughSample:
            print("ERROR! There are not enough samples to satisfy dataset parameters. Aborting...")
            sys.exit(-1)

    if channel_first:
        all_IQ_data = all_IQ_data[:, 0:n_samples]
    else:
        all_IQ_data = all_IQ_data[0:n_samples, :]
    return all_IQ_data, num_tran


def loadData(args, channel_first):
    n_slices_per_dev = args.num_slice
    start_ix = args.start_ix
    file_key = args.file_key

    dev_dir_list = []
    dev_dir_names = os.listdir(args.root_dir)
    for n in dev_dir_names:
        tmp = os.path.join(args.root_dir, n)
        dev_dir_list.append(tmp)

    stride = args.stride
    n_devices = len(dev_dir_list)
    # locations = ["after_fft","before_fft", "output_equ", "symbols"]
    # locations = ["output_equ"]
    if channel_first:
        slice_dims = (2, args.slice_len)
        samps_to_retrieve = (n_slices_per_dev - 1) * stride + slice_dims[1]
    else:
        slice_dims = (args.slice_len, 2)
        samps_to_retrieve = (n_slices_per_dev - 1) * stride + slice_dims[0]

    x_train, y_train, x_test, y_test = [], [], [], []
    split_ratio = {'train': 0.8, 'val': 0.2}
    for i, d in enumerate(dev_dir_list):

        p = os.path.join(d, args.location)
        pre_X_data, num_tran = dev_bin_dataset(os.path.join(p, file_key), samps_to_retrieve, start_ix, channel_first, uniform=True)

        X_data_pd = []
        count_s = 0
        for j in range(0, samps_to_retrieve, stride):
            if channel_first:
                X_data_pd.append(pre_X_data[:, j:j+slice_dims[1]])
            else:
                X_data_pd.append(pre_X_data[j:j+slice_dims[0], :])
            count_s += 1
            if count_s == n_slices_per_dev:
                break
        a = X_data_pd[-1]
        X_data_pd = np.array(X_data_pd)
        y_data_pd = i * np.ones(n_slices_per_dev, )

        # split one class data
        uniform = True
        x_train_pd, x_test_pd, y_train_pd, y_test_pd = [],[],[],[]
        if uniform:
            samples_per_tran = n_slices_per_dev // num_tran
            idx = 0
            while idx + samples_per_tran <= n_slices_per_dev:
                x_train_per_tran, y_train_per_tran, x_test_per_tran, y_test_per_tran = utils.splitData(split_ratio, X_data_pd[i:i+samples_per_tran, :, :], y_data_pd[i:i+samples_per_tran])
                if idx == 0:
                    x_train_pd, x_test_pd = x_train_per_tran, x_test_per_tran
                    y_train_pd, y_test_pd = y_train_per_tran, y_test_per_tran
                else:
                    x_train_pd = np.concatenate((x_train_pd, x_train_per_tran), axis=0)
                    x_test_pd = np.concatenate((x_test_pd, x_test_per_tran), axis=0)
                    y_train_pd = np.concatenate((y_train_pd, y_train_per_tran), axis=0)
                    y_test_pd = np.concatenate((y_test_pd, y_test_per_tran), axis=0)
                idx += samples_per_tran
        else:

            x_train_pd, y_train_pd, x_test_pd, y_test_pd = utils.splitData(split_ratio, X_data_pd, y_data_pd)

        if i == 0:
            x_train, x_test = x_train_pd, x_test_pd
            y_train, y_test = y_train_pd, y_test_pd
        else:
            x_train = np.concatenate((x_train, x_train_pd), axis=0)
            x_test = np.concatenate((x_test, x_test_pd), axis=0)
            y_train = np.concatenate((y_train, y_train_pd), axis=0)
            y_test = np.concatenate((y_test, y_test_pd), axis=0)
        del pre_X_data
        del X_data_pd

    if args.D2:
        if channel_first:
            x_train = x_train[:, :, np.newaxis, :]
            x_test = x_test[:, :, np.newaxis, :]
        else:
            x_train = x_train[:, np.newaxis, :, :]
            x_test = x_test[:, np.newaxis, :, :]

    y_train = np_utils.to_categorical(y_train, n_devices)
    y_test = np_utils.to_categorical(y_test, n_devices)
    return x_train, y_train, x_test, y_test, n_devices


class loadDataOpts():
    def __init__(self, root_dir, location, D2, file_key='*.bin', num_slice=100000, start_ix=0, slice_len=288, stride=1):
        self.root_dir = root_dir
        self.num_slice = num_slice
        self.start_ix = start_ix
        self.slice_len = slice_len
        self.stride = stride
        self.file_key = file_key
        self.D2 = D2
        self.location = location


def parseArgs(argv):
    Desc = 'Read and slice the collected I/Q samples'
    parser = argparse.ArgumentParser(description=Desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--root_dir', required=True, help='Root directory for the devices\' folders.')
    parser.add_argument('-n', '--num_slice', required=True, type=int, help='Number of slices to be generated for each device.')
    parser.add_argument('-i', '--start_ix', type=int, default=0, help='Starting read index in .bin files.')
    parser.add_argument('-l', '--slice_len', type=int, default=288, help='Lenght of slices.')
    parser.add_argument('-s', '--stride', type=int, default=1, help='Stride used for windowing.')
    parser.add_argument('-f', '--file_key', default='*.bin', help='used to choose different filetype, choose from *.bin/*.sigmf-meta')
    parser.add_argument('--D2', action='store_true', help='')
    parser.add_argument('-cf', '--channel_first', action='store_true', help='if set channel first otherwise channel last')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    # opts = parseArgs(sys.argv)
    opts = loadDataOpts(root_dir="/home/erc/PycharmProjects/rf/test_dataset/", D2=False, location='after_fft')
    channel_first = True
    x_train, y_train, x_test, y_test, NUM_CLASS = loadData(opts, channel_first)
    print('train data shape: ', x_train.shape, 'train label shape: ', y_train.shape)
    print('test data shape: ', x_test.shape, 'test label shape: ', y_test.shape)
    print('all test done!')
