#! /usr/bin python3.6

import os
import sys
import pdb
import math

import json
import numpy as np
from sigmf import SigMFFile, sigmffile

import config


class OurWay():
    def __init__(self, droot):
        fnames = os.listdir(droot)
        for fn in fnames:
            if fn.endswith('bin'):
                self.binfile = os.path.join(droot, fn)
            elif fn.endswith('sigmf-meta'):
                self.metafile = os.path.join(droot, fn)

    def createSignal(self, metafile, binfile):
        # Load a dataset
        with open(metafile, 'r') as f:
            metadata = json.loads(f.read())
        signal = SigMFFile(metadata=metadata['_metadata'], data_file=binfile)
        return signal

    def convert2IQdata(self, one_raw_data):
        realList, imagList = [], []
        for item in one_raw_data:
            realVal = np.real(item)
            imagVal = np.imag(item)
            if math.isnan(realVal):
                realVal = 0
            if math.isnan(imagVal):
                imagVal = 0
            realList.append(realVal)
            imagList.append(imagVal)
        rtnList = [realList, imagList]
        rtnList = np.array(rtnList)
        return rtnList

    def getOneData(self):
        # Get some metadata and all annotations
        sample_length = 288
        signal = self.createSignal(self.metafile, self.binfile)
        raw_data = signal.read_samples(0, -1)
        one_data = self.convert2IQdata(raw_data)
        pdb.set_trace()
        return one_data[:, :sample_length]


class AuthorWay():
    def __init__(self, droot):
        fnames = os.listdir(droot)
        for fn in fnames:
            if fn.endswith('bin'):
                fp = os.path.join(droot, fn)
                self.filename = fp

    def read_f32_bin(self, filename, start_ix=0):
        with open(filename, 'rb') as bin_f:
            iq_seq = np.fromfile(bin_f, dtype='<f4')
            n_samples = iq_seq.shape[0] // 2

            IQ_data = np.zeros((2, n_samples))

            IQ_data[0, :] = iq_seq[range(0, iq_seq.shape[0]-1, 2)]
            IQ_data[1, :] = iq_seq[range(1, iq_seq.shape[0], 2)]
        del iq_seq
        return IQ_data[:, start_ix:]

    def getOneData(self):
        sample_length = 288
        raw_data = self.read_f32_bin(self.filename)
        return raw_data[:, :sample_length]


if __name__ == "__main__":
    opts = config.parse_args(sys.argv)
    ourWay = OurWay(opts.input)
    ourWayData = ourWay.getOneData()
    print('our way data...')
    print(ourWayData)

    print('\n#######################################\n')

    authorWay = AuthorWay(opts.input)
    authorWayData = authorWay.getOneData()
    print('author way data...')
    print(authorWayData)

    print('all test passed!')
