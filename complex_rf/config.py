#!/usr/bin/python3
'''
This module is for universal arg parse
'''

import argparse


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file/dir')
    parser.add_argument('-o', '--output', help='output dir')
    parser.add_argument('-m', '--modelType', default='homegrown', help='choose from homegrown/baseline/resnet')
    parser.add_argument('-sp', '--splitType', default='random', help='choose from random/order')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('--D2', action='store_true', help='if set will return 2 dimension data')
    parser.add_argument('-n', '--normalize', action='store_true', help='')
    parser.add_argument('-ds', '--dataSource', help='choose from neu/simu')
    parser.add_argument('-cf', '--channel_first', action='store_true', help='if set channel first otherwise channel last')
    parser.add_argument('-l','--location',help='data collected location')
    opts = parser.parse_args()
    return opts
