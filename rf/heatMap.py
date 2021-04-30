#encoding=utf-8

import os
import sys
import argparse

import matplotlib
matplotlib.use('Agg')
'''
# The purpose of these  following three lines is to generate figures without type 3 fonts
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts']
matplotlib.rcParams['text.usetex']
'''
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams.update({'font.family': 'Times New Roman'})

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import defaultdict

import mytools.fileUtils as fileUtils
import readSigmf


def reshape2OneDim(oldData):
    if not isinstance(oldData, list):
        oldData = list(oldData)
    newData = []
    for item in oldData:
        item = np.array(item).T
        item = list(item)
        tmp = []
        tmp.extend(item[0])
        tmp.extend(item[1])
        newData.append(tmp)

    return newData


def loadData(x_day_dir, data_dim=40):
    rtnClass = defaultdict()
    rcParams = readSigmf.generate_default_params()
    devList = os.listdir(x_day_dir)
    for i in range(len(devList)):
        strLabel = devList[i]
        fpTuple = readSigmf.getfpTuple(strLabel, x_day_dir)
        oneData, oneLabel = readSigmf.getOneDevData(fpTuple, i, rcParams)
        oneDimData = reshape2OneDim(oneData)
        oneDimData = np.array(oneDimData)
        oneDimData = oneDimData[:data_dim,:]
        print('one dim data shape is: ', oneDimData.shape)
        rtnClass[i] = oneDimData

    return rtnClass


def main(opts):
    if not os.path.isdir(opts.output):
        os.makedirs(opts.output)
    classData = loadData(opts.input)
    count = 1
    for key in classData.keys():
        sns.set(style='whitegrid', color_codes=True, font_scale=1.4)
        #sns.palplot(sns.color_palette("Blues"))
        print('({:d}/{:d}) now generate figure for class {}...'.format(count, len(list(classData.keys())), key))
        count = count + 1
        fpath = os.path.join(opts.output, '{}.{}'.format(key, opts.dataFormat))
        data = classData[key]
        #ax = sns.heatmap(data, center=0, xticklabels=50, cmap="cubehelix")
        #sns.color_palette()

        # 组合图的框架
        xSize, ySize = 10, 3
        fig = plt.figure(figsize=(xSize, ySize))
        # 使用gs参数，确定了子图的句柄，可以通过对ax进行操作，以达到不同位置显示不同的图像的目的
        #gs = gridspec.GridSpec(data.shape[0], 1, height_ratios=[3,1])
        gs = gridspec.GridSpec(data.shape[0], 1, wspace=0.4, hspace=0.4)
        ax = defaultdict()
        for i in range(data.shape[0]):
            ax[i] = plt.subplot(gs[i])

        allAx = []
        for i in range(data.shape[0]):
            # heatmap requires 2D input
            sample = data[i, :]
            sample = sample.reshape(1, sample.shape[0])
            heatplt = sns.heatmap(sample, center=0, xticklabels=100, cmap="RdBu_r", ax=ax[i], cbar=False, vmin=-1500, vmax=1500)
            allAx.append(ax[i])
            heatplt.axes.get_xaxis().set_visible(False)
            heatplt.axes.get_yaxis().set_visible(False)
            lastOne = i

        #ax[lastOne].set_xlabel('No. of Packets', fontsize=16)
        #ax.set_ylabel('Trace Index', fontsize=16)

        #label_y = ax.get_yticklabels()
        #plt.setp(label_y, rotation=360, horizontalalignment='right')
        #label_x = ax[lastOne].get_xticklabels()
        #plt.setp(label_x, rotation=45, horizontalalignment='right')

        #tmp = key.split('_')
        #new_title = ' '.join(tmp)
        #fig.suptitle(new_title, fontsize=16)
        fig.colorbar(ax[0].collections[0], ax=allAx)
        # 加入tight_layout后，会导致colorbar和子图重合
        #gs.tight_layout(fig)
        #plt.show()
        plt.savefig(fpath, bbox_inches='tight')
        #import pdb
        #pdb.set_trace()
        plt.pause(1)
        plt.close('all')


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-df', '--dataFormat', default='pdf', help='file format to store, jpg, pdf or png')
    parser.add_argument('-o', '--output', default='heatMap', help='')
    parser.add_argument('-c', '--cbar', action='store_true', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
